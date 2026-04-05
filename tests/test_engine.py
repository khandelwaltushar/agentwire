"""Tests for agentwire.engine — workflow execution."""

from __future__ import annotations

import anyio
import pytest

from agentwire.engine import Agent, Engine, RetryConfig, Step, Workflow
from agentwire.types import Context, StepStatus

# ── Helpers ──────────────────────────────────────────────────────────


class Echo(Agent):
    """Agent that copies a key into data."""

    def __init__(self, key: str, value: object) -> None:
        self.key = key
        self.value = value

    async def run(self, ctx: Context) -> Context:
        return ctx.set(self.key, self.value)


class Boom(Agent):
    """Agent that always raises."""

    def __init__(self, msg: str = "boom") -> None:
        self.msg = msg

    async def run(self, _ctx: Context) -> Context:
        raise RuntimeError(self.msg)


class FlakeyAgent(Agent):
    """Fails *n* times then succeeds."""

    def __init__(self, fail_times: int, key: str, value: object) -> None:
        self.remaining = fail_times
        self.key = key
        self.value = value

    async def run(self, ctx: Context) -> Context:
        if self.remaining > 0:
            self.remaining -= 1
            msg = "transient"
            raise RuntimeError(msg)
        return ctx.set(self.key, self.value)


# ── Tests ────────────────────────────────────────────────────────────


class TestLinearExecution:
    @pytest.mark.anyio
    async def test_single_step(self):
        wf = Workflow(
            name="single",
            steps={"a": Step(name="a", agent=Echo("x", 1))},
            start="a",
        )
        trace = await Engine().run(wf, Context())
        assert trace.success
        assert len(trace.results) == 1
        assert trace.results[0].output_context is not None
        assert trace.results[0].output_context.get("x") == 1

    @pytest.mark.anyio
    async def test_two_step_chain(self):
        wf = Workflow(
            name="chain",
            steps={
                "a": Step(name="a", agent=Echo("x", 1), next_steps=["b"]),
                "b": Step(name="b", agent=Echo("y", 2)),
            },
            start="a",
        )
        trace = await Engine().run(wf, Context())
        assert trace.success
        assert len(trace.results) == 2
        last = trace.results[-1]
        assert last.output_context is not None
        assert last.output_context.get("x") == 1
        assert last.output_context.get("y") == 2

    @pytest.mark.anyio
    async def test_three_step_chain(self):
        wf = Workflow(
            name="long",
            steps={
                "a": Step(name="a", agent=Echo("a", 1), next_steps=["b"]),
                "b": Step(name="b", agent=Echo("b", 2), next_steps=["c"]),
                "c": Step(name="c", agent=Echo("c", 3)),
            },
            start="a",
        )
        trace = await Engine().run(wf, Context())
        assert trace.success
        assert len(trace.results) == 3
        final = trace.results[-1].output_context
        assert final is not None
        assert final.data == {"a": 1, "b": 2, "c": 3}


class TestFanOutFanIn:
    @pytest.mark.anyio
    async def test_parallel_steps(self):
        wf = Workflow(
            name="fanout",
            steps={
                "start": Step(
                    name="start",
                    agent=Echo("init", True),
                    next_steps=["left", "right"],
                ),
                "left": Step(
                    name="left",
                    agent=Echo("l", "left"),
                    next_steps=["join"],
                ),
                "right": Step(
                    name="right",
                    agent=Echo("r", "right"),
                    next_steps=["join"],
                ),
                "join": Step(name="join", agent=Echo("done", True)),
            },
            start="start",
        )
        trace = await Engine().run(wf, Context())
        assert trace.success
        names = [r.step_name for r in trace.results]
        assert names[0] == "start"
        assert set(names[1:3]) == {"left", "right"}
        assert names[3] == "join"
        final = trace.results[-1].output_context
        assert final is not None
        assert final.get("l") == "left"
        assert final.get("r") == "right"
        assert final.get("done") is True

    @pytest.mark.anyio
    async def test_fan_out_runs_concurrently(self):
        """Verify parallel steps actually overlap in time."""
        order: list[str] = []

        class SlowAgent(Agent):
            def __init__(self, tag: str) -> None:
                self.tag = tag

            async def run(self, ctx: Context) -> Context:
                order.append(f"{self.tag}_start")
                await anyio.sleep(0.05)
                order.append(f"{self.tag}_end")
                return ctx.set(self.tag, True)

        wf = Workflow(
            name="concurrent",
            steps={
                "root": Step(
                    name="root",
                    agent=Echo("go", True),
                    next_steps=["p", "q"],
                ),
                "p": Step(name="p", agent=SlowAgent("p")),
                "q": Step(name="q", agent=SlowAgent("q")),
            },
            start="root",
        )
        trace = await Engine().run(wf, Context())
        assert trace.success
        # Both should start before either ends (true concurrency)
        assert set(order[:2]) == {"p_start", "q_start"}


class TestConditionalBranching:
    @pytest.mark.anyio
    async def test_branch_on_context(self):
        def router(ctx: Context) -> list[str]:
            if ctx.get("route") == "left":
                return ["left"]
            return ["right"]

        wf = Workflow(
            name="branch",
            steps={
                "start": Step(
                    name="start",
                    agent=Echo("route", "left"),
                    next_steps=router,
                ),
                "left": Step(name="left", agent=Echo("chosen", "left")),
                "right": Step(name="right", agent=Echo("chosen", "right")),
            },
            start="start",
        )
        trace = await Engine().run(wf, Context())
        assert trace.success
        assert len(trace.results) == 2
        assert trace.results[1].step_name == "left"
        final = trace.results[-1].output_context
        assert final is not None
        assert final.get("chosen") == "left"

    @pytest.mark.anyio
    async def test_branch_takes_other_path(self):
        def router(ctx: Context) -> list[str]:
            return ["b"] if ctx.get("flag") else ["a"]

        wf = Workflow(
            name="branch2",
            steps={
                "entry": Step(
                    name="entry",
                    agent=Echo("flag", True),
                    next_steps=router,
                ),
                "a": Step(name="a", agent=Echo("path", "a")),
                "b": Step(name="b", agent=Echo("path", "b")),
            },
            start="entry",
        )
        trace = await Engine().run(wf, Context())
        assert trace.success
        assert trace.results[-1].step_name == "b"


class TestRetryWithBackoff:
    @pytest.mark.anyio
    async def test_succeeds_after_retries(self):
        agent = FlakeyAgent(fail_times=2, key="val", value="ok")
        wf = Workflow(
            name="retry",
            steps={
                "flaky": Step(
                    name="flaky",
                    agent=agent,
                    retry_config=RetryConfig(
                        max_attempts=3, backoff_seconds=0.01,
                    ),
                ),
            },
            start="flaky",
        )
        trace = await Engine().run(wf, Context())
        assert trace.success
        assert trace.results[0].retries == 2
        assert trace.results[0].output_context is not None
        assert trace.results[0].output_context.get("val") == "ok"

    @pytest.mark.anyio
    async def test_exhausts_retries(self):
        agent = FlakeyAgent(fail_times=5, key="v", value="x")
        wf = Workflow(
            name="exhaust",
            steps={
                "flaky": Step(
                    name="flaky",
                    agent=agent,
                    retry_config=RetryConfig(
                        max_attempts=3, backoff_seconds=0.01,
                    ),
                ),
            },
            start="flaky",
        )
        trace = await Engine().run(wf, Context())
        assert not trace.success
        assert trace.results[0].status == StepStatus.FAILED
        assert trace.results[0].retries == 2
        assert trace.results[0].error == "transient"

    @pytest.mark.anyio
    async def test_no_retry_by_default(self):
        wf = Workflow(
            name="noretry",
            steps={"x": Step(name="x", agent=Boom())},
            start="x",
        )
        trace = await Engine().run(wf, Context())
        assert not trace.success
        assert trace.results[0].retries == 0


class TestFailedStepHandling:
    @pytest.mark.anyio
    async def test_failure_stops_execution(self):
        wf = Workflow(
            name="stop",
            steps={
                "a": Step(name="a", agent=Boom("fail"), next_steps=["b"]),
                "b": Step(name="b", agent=Echo("x", 1)),
            },
            start="a",
        )
        trace = await Engine().run(wf, Context())
        assert not trace.success
        assert len(trace.results) == 1
        assert trace.results[0].status == StepStatus.FAILED
        assert trace.results[0].error == "fail"

    @pytest.mark.anyio
    async def test_failure_preserves_earlier_results(self):
        wf = Workflow(
            name="partial",
            steps={
                "ok": Step(
                    name="ok", agent=Echo("x", 1), next_steps=["bad"],
                ),
                "bad": Step(name="bad", agent=Boom()),
            },
            start="ok",
        )
        trace = await Engine().run(wf, Context())
        assert not trace.success
        assert len(trace.results) == 2
        assert trace.results[0].status == StepStatus.SUCCESS
        assert trace.results[1].status == StepStatus.FAILED

    @pytest.mark.anyio
    async def test_trace_has_finished_timestamp(self):
        wf = Workflow(
            name="ts",
            steps={"a": Step(name="a", agent=Echo("k", "v"))},
            start="a",
        )
        trace = await Engine().run(wf, Context())
        assert trace.finished_at is not None


class TestWorkflowValidation:
    def test_invalid_start_raises(self):
        with pytest.raises(ValueError, match="start step"):
            Workflow(name="bad", steps={}, start="missing")
