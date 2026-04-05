#!/usr/bin/env python3
"""Research-agent demo — plan → search → summarise → critique loop.

Runs end-to-end with **no API keys**.  The DuckDuckGo search step makes
a real HTTP request; everything else is a stubbed "LLM" that returns
plausible strings after a short delay (simulating latency).

Usage:
    python examples/research_agent.py [topic]
"""

from __future__ import annotations

import random
import sys
import textwrap

import anyio
import httpx

from agentwire.engine import Engine, Step, Workflow
from agentwire.observability import TraceRenderer, trace_to_dict
from agentwire.types import Context

SEARCH_URL = "https://html.duckduckgo.com/html/"
MAX_REVISIONS = 2


# ── Agents ───────────────────────────────────────────────────────────


class PlanAgent:
    """Stubbed LLM that turns a topic into a research plan."""

    async def run(self, ctx: Context) -> Context:
        topic = ctx.get("topic", "AI agents")
        await anyio.sleep(0.1)  # simulate LLM latency
        plan = (
            f"Research plan for '{topic}':\n"
            f"  1. Search for recent developments in {topic}\n"
            f"  2. Identify key themes and players\n"
            f"  3. Synthesise findings into a concise summary"
        )
        return ctx.set("plan", plan)


class SearchAgent:
    """Real DuckDuckGo search via httpx — no API key required."""

    async def run(self, ctx: Context) -> Context:
        topic = ctx.get("topic", "AI agents")
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
        ) as client:
            resp = await client.post(
                SEARCH_URL,
                data={"q": topic, "b": ""},
                headers={"User-Agent": "agentwire-demo/0.1"},
            )
            resp.raise_for_status()

        # Extract snippet text from the HTML result blobs.
        snippets = _extract_snippets(resp.text, max_results=5)
        return ctx.set("search_results", snippets)


class SummariseAgent:
    """Stubbed LLM that pretends to summarise search results."""

    async def run(self, ctx: Context) -> Context:
        results = ctx.get("search_results", [])
        revision = ctx.get("revision", 0)
        await anyio.sleep(0.1)

        bullet_points = "\n".join(
            f"  - {s[:120]}" for s in results[:5]
        )
        summary = (
            f"Summary (revision {revision}):\n"
            f"Based on {len(results)} search results, "
            f"key findings include:\n{bullet_points}\n"
            f"[End of summary]"
        )
        return ctx.set("summary", summary)


class CritiqueAgent:
    """Stubbed LLM critic — randomly approves or requests revision."""

    async def run(self, ctx: Context) -> Context:
        await anyio.sleep(0.05)
        revision = ctx.get("revision", 0)

        # Force approval after max revisions to prevent infinite loops.
        if revision >= MAX_REVISIONS:
            verdict = "approved"
        else:
            verdict = random.choice(["approved", "needs_revision"])  # noqa: S311

        feedback = (
            "Looks comprehensive — approved."
            if verdict == "approved"
            else "Summary lacks depth on recent trends. Revise."
        )
        return ctx.set("verdict", verdict).set("feedback", feedback)


class RefineAgent:
    """Stubbed LLM that 'improves' the summary based on feedback."""

    async def run(self, ctx: Context) -> Context:
        await anyio.sleep(0.1)
        revision = ctx.get("revision", 0) + 1
        old_summary = ctx.get("summary", "")
        feedback = ctx.get("feedback", "")
        refined = (
            f"{old_summary}\n"
            f"  [Revision {revision} — addressed: {feedback}]"
        )
        return ctx.set("summary", refined).set("revision", revision)


# ── HTML helpers ─────────────────────────────────────────────────────


def _extract_snippets(html: str, *, max_results: int = 5) -> list[str]:
    """Pull snippet text from DuckDuckGo HTML results (no lxml needed)."""
    snippets: list[str] = []
    marker = 'class="result__snippet"'
    pos = 0
    while len(snippets) < max_results:
        idx = html.find(marker, pos)
        if idx == -1:
            break
        tag_end = html.find(">", idx)
        close = html.find("</", tag_end)
        if tag_end == -1 or close == -1:
            break
        raw = html[tag_end + 1 : close]
        text = _strip_tags(raw).strip()
        if text:
            snippets.append(text)
        pos = close
    if not snippets:
        snippets = ["(No snippets extracted — DuckDuckGo may have changed format)"]
    return snippets


def _strip_tags(html: str) -> str:
    """Naïve tag stripper — good enough for plain-text snippet extraction."""
    out: list[str] = []
    inside = False
    for ch in html:
        if ch == "<":
            inside = True
        elif ch == ">":
            inside = False
        elif not inside:
            out.append(ch)
    return "".join(out)


# ── Workflow wiring ──────────────────────────────────────────────────


def _critique_router(ctx: Context) -> list[str]:
    """Route from critique → refine (loop) or → done (exit)."""
    if ctx.get("verdict") == "needs_revision":
        return ["refine"]
    return []  # approved — stop


def build_workflow() -> Workflow:
    """Construct the research-agent workflow graph."""
    return Workflow(
        name="research-agent",
        steps={
            "plan": Step(
                name="plan",
                agent=PlanAgent(),
                next_steps=["search"],
            ),
            "search": Step(
                name="search",
                agent=SearchAgent(),
                next_steps=["summarise"],
            ),
            "summarise": Step(
                name="summarise",
                agent=SummariseAgent(),
                next_steps=["critique"],
            ),
            "critique": Step(
                name="critique",
                agent=CritiqueAgent(),
                next_steps=_critique_router,
            ),
            "refine": Step(
                name="refine",
                agent=RefineAgent(),
                next_steps=["summarise"],
            ),
        },
        start="plan",
    )


# ── Main ─────────────────────────────────────────────────────────────


async def main(topic: str) -> None:
    """Run the research workflow and display results."""
    ctx = Context(data={"topic": topic})
    workflow = build_workflow()
    engine = Engine()

    print(f"\n{'=' * 60}")
    print(f"  Research Agent — topic: {topic!r}")
    print(f"{'=' * 60}\n")

    trace = await engine.run(workflow, ctx)

    # ── Trace tree ───────────────────────────────────────────────
    print("Execution trace:")
    TraceRenderer().print(trace)
    print()

    # ── Final output ─────────────────────────────────────────────
    last_ctx = None
    for r in reversed(trace.results):
        if r.output_context is not None:
            last_ctx = r.output_context
            break

    if last_ctx is not None:
        verdict = last_ctx.get("verdict", "n/a")
        summary = last_ctx.get("summary", "(no summary)")
        revision = last_ctx.get("revision", 0)

        print(f"Verdict : {verdict}")
        print(f"Revisions: {revision}")
        print(f"\nFinal summary:\n{textwrap.indent(summary, '  ')}\n")

    # ── Serialised trace (first 3 keys) ─────────────────────────
    d = trace_to_dict(trace)
    print(
        f"Serialised trace: {len(d['results'])} step(s), "
        f"success={d['success']}",
    )


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI agent orchestration"
    anyio.run(main, topic)
