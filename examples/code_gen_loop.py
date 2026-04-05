#!/usr/bin/env python3
"""Code-generation loop demo — generate → test → fix → re-test.

A stub "LLM" generates Python code with an intentional bug.  The test
step **actually executes** the code via ``subprocess`` and captures
stdout/stderr.  If the test fails, a stub "fix" agent patches the bug
and the loop re-tests (up to ``MAX_FIX_ATTEMPTS`` times).

Usage:
    python examples/code_gen_loop.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import anyio

from agentwire.engine import Engine, Step, Workflow
from agentwire.observability import TraceRenderer, trace_to_dict
from agentwire.types import Context, ExecutionTrace

MAX_FIX_ATTEMPTS = 3

# ── The "code under test" at each stage ──────────────────────────────

# Round 0: intentional bug — off-by-one in range()
BUGGY_CODE = textwrap.dedent("""\
    def fizzbuzz(n: int) -> list[str]:
        result: list[str] = []
        for i in range(1, n):  # BUG: should be n + 1
            if i % 15 == 0:
                result.append("FizzBuzz")
            elif i % 3 == 0:
                result.append("Fizz")
            elif i % 5 == 0:
                result.append("Buzz")
            else:
                result.append(str(i))
        return result
""")

# Round 1: still buggy — wrong modulo order
FIX_ATTEMPT_1 = textwrap.dedent("""\
    def fizzbuzz(n: int) -> list[str]:
        result: list[str] = []
        for i in range(1, n + 1):
            if i % 3 == 0:
                result.append("Fizz")  # BUG: should check 15 first
            elif i % 5 == 0:
                result.append("Buzz")
            elif i % 15 == 0:
                result.append("FizzBuzz")
            else:
                result.append(str(i))
        return result
""")

# Round 2: correct implementation
FIXED_CODE = textwrap.dedent("""\
    def fizzbuzz(n: int) -> list[str]:
        result: list[str] = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                result.append("FizzBuzz")
            elif i % 3 == 0:
                result.append("Fizz")
            elif i % 5 == 0:
                result.append("Buzz")
            else:
                result.append(str(i))
        return result
""")

# The test harness that gets appended to whatever code we're testing.
TEST_HARNESS = textwrap.dedent("""\

    # ── test harness ──────────────────────────────────────
    import sys

    expected_15 = [
        "1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz",
        "11","Fizz","13","14","FizzBuzz",
    ]
    got = fizzbuzz(15)
    if got != expected_15:
        print(f"FAIL: fizzbuzz(15)\\n  expected: {expected_15}\\n       got: {got}",
              file=sys.stderr)
        sys.exit(1)

    # Edge cases
    assert fizzbuzz(1) == ["1"], f"fizzbuzz(1) = {fizzbuzz(1)}"
    assert fizzbuzz(3) == ["1", "2", "Fizz"], f"fizzbuzz(3) = {fizzbuzz(3)}"

    print("ALL TESTS PASSED")
""")

# Ordered list of code versions the "fix" agent cycles through.
_CODE_VERSIONS = [BUGGY_CODE, FIX_ATTEMPT_1, FIXED_CODE]


# ── Helpers ──────────────────────────────────────────────────────────


def _run_code(source: str) -> subprocess.CompletedProcess[str]:
    """Write *source* to a temp file, execute it, and return the result."""
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False,
    ) as tmp:
        tmp.write(source)
        tmp_path = tmp.name
    try:
        return subprocess.run(  # noqa: S603
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Agents ───────────────────────────────────────────────────────────


class GenerateAgent:
    """Stub LLM that emits buggy fizzbuzz code."""

    async def run(self, ctx: Context) -> Context:
        await anyio.sleep(0.05)
        return ctx.set("code", BUGGY_CODE).set("fix_attempt", 0)


class TestAgent:
    """Runs the current code in a real subprocess and captures output."""

    async def run(self, ctx: Context) -> Context:
        code = ctx.get("code", "")
        full_source = code + TEST_HARNESS
        proc = await anyio.to_thread.run_sync(
            lambda: _run_code(full_source),
        )
        return (
            ctx.set("test_passed", proc.returncode == 0)
            .set("test_stdout", proc.stdout.strip())
            .set("test_stderr", proc.stderr.strip())
            .set("test_returncode", proc.returncode)
        )


class FixAgent:
    """Stub LLM that cycles through progressively better code versions."""

    async def run(self, ctx: Context) -> Context:
        await anyio.sleep(0.05)
        attempt = ctx.get("fix_attempt", 0) + 1
        # Pick the next code version (clamped to the last one).
        code_idx = min(attempt, len(_CODE_VERSIONS) - 1)
        new_code = _CODE_VERSIONS[code_idx]
        return ctx.set("code", new_code).set("fix_attempt", attempt)


# ── Routing ──────────────────────────────────────────────────────────


def _test_router(ctx: Context) -> list[str]:
    """After a test run, decide: done or fix → re-test."""
    if ctx.get("test_passed"):
        return []  # success — stop

    attempt = ctx.get("fix_attempt", 0)
    if attempt >= MAX_FIX_ATTEMPTS:
        return ["fail"]  # exhausted retries
    return ["fix"]


class FailAgent:
    """Terminal agent that records structured failure information."""

    async def run(self, ctx: Context) -> Context:
        stderr = ctx.get("test_stderr", "")
        attempt = ctx.get("fix_attempt", 0)
        error_msg = (
            f"Code generation failed after {attempt} fix attempts. "
            f"Last error:\n{stderr}"
        )
        return ctx.set("final_error", error_msg)


# ── Workflow ─────────────────────────────────────────────────────────


def build_workflow() -> Workflow:
    """Construct the code-generation-loop workflow."""
    return Workflow(
        name="code-gen-loop",
        steps={
            "generate": Step(
                name="generate",
                agent=GenerateAgent(),
                next_steps=["test"],
            ),
            "test": Step(
                name="test",
                agent=TestAgent(),
                next_steps=_test_router,
            ),
            "fix": Step(
                name="fix",
                agent=FixAgent(),
                next_steps=["test"],
            ),
            "fail": Step(
                name="fail",
                agent=FailAgent(),
            ),
        },
        start="generate",
    )


# ── Main ─────────────────────────────────────────────────────────────


async def main() -> None:
    """Run the code-generation workflow and display results."""
    workflow = build_workflow()
    engine = Engine()
    ctx = Context(data={"task": "Implement fizzbuzz"})

    print(f"\n{'=' * 60}")
    print("  Code Generation Loop Demo")
    print(f"{'=' * 60}\n")

    trace = await engine.run(workflow, ctx)

    # ── Trace tree ───────────────────────────────────────────
    print("Execution trace:")
    TraceRenderer().print(trace)
    print()

    _print_step_details(trace)
    _print_verdict(trace)


def _print_step_details(trace: ExecutionTrace) -> None:
    """Print per-step detail lines."""
    for result in trace.results:
        out = result.output_context
        if out is None:
            continue
        if result.step_name == "test":
            _print_test_detail(result.step_name, out)
        elif result.step_name == "fix":
            print(f"  [{result.step_name}] producing fix attempt {out.get('fix_attempt', 0)}")
        elif result.step_name == "fail":
            print(f"  [{result.step_name}] {out.get('final_error', '')}")


def _print_test_detail(step_name: str, out: Context) -> None:
    status = "PASS" if out.get("test_passed") else "FAIL"
    print(f"  [{step_name}] attempt={out.get('fix_attempt', 0)}  {status}")
    if out.get("test_stdout"):
        print(f"    stdout: {out.get('test_stdout')}")
    if out.get("test_stderr"):
        for line in str(out.get("test_stderr", "")).splitlines():
            print(f"    stderr: {line}")


def _print_verdict(trace: ExecutionTrace) -> None:
    last_ctx = None
    for r in reversed(trace.results):
        if r.output_context is not None:
            last_ctx = r.output_context
            break

    print()
    if last_ctx is not None and last_ctx.get("test_passed"):
        print(f"Result: PASSED after {last_ctx.get('fix_attempt', 0)} fix attempt(s)")
    elif last_ctx is not None and last_ctx.get("final_error"):
        print(f"Result: FAILED\n  {last_ctx.get('final_error')}")
    else:
        print("Result: FAILED (unknown)")

    d = trace_to_dict(trace)
    print(
        f"\nSerialised trace: {len(d['results'])} step(s), "
        f"success={d['success']}",
    )


if __name__ == "__main__":
    anyio.run(main)
