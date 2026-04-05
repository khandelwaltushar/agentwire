"""Example: a multi-step research workflow using agentwire.

This demonstrates a realistic pipeline that:
1. Validates a research query
2. Fetches sources in parallel (web + academic)
3. Synthesizes findings into a summary
4. Formats the final report

Run with::

    python examples/research_agent.py
"""

from __future__ import annotations

import anyio

from agentwire import Context, Engine, Step, Workflow
from agentwire.step import RetryPolicy


# ---------------------------------------------------------------------------
# Agent implementations (no real LLM calls — pure simulation for the example)
# ---------------------------------------------------------------------------


class ValidateQueryAgent:
    """Checks that the query is non-empty and well-formed."""

    async def run(self, ctx: Context) -> Context:
        query = ctx.get("query", "")
        if not query:
            raise ValueError("Query must not be empty")
        return ctx.set("validated_query", query.strip().lower())


class WebSearchAgent:
    """Simulates a web search returning article titles."""

    async def run(self, ctx: Context) -> Context:
        query = ctx.get("validated_query", "")
        # Simulate latency
        await anyio.sleep(0.05)
        results = [
            f"Web result 1 for '{query}'",
            f"Web result 2 for '{query}'",
            f"Web result 3 for '{query}'",
        ]
        return ctx.set("web_results", results)


class AcademicSearchAgent:
    """Simulates an academic paper search."""

    async def run(self, ctx: Context) -> Context:
        query = ctx.get("validated_query", "")
        await anyio.sleep(0.03)
        papers = [
            f"Paper: 'A Survey of {query}' (2024)",
            f"Paper: 'Advances in {query}' (2025)",
        ]
        return ctx.set("academic_results", papers)


class SynthesizeAgent:
    """Merges web and academic results into key findings."""

    async def run(self, ctx: Context) -> Context:
        web: list[str] = ctx.get("web_results", [])
        academic: list[str] = ctx.get("academic_results", [])
        findings = [
            f"Finding 1: Combined insight from {len(web)} web sources",
            f"Finding 2: {len(academic)} academic papers reviewed",
            f"Finding 3: Topic '{ctx.get('validated_query')}' is well-covered",
        ]
        return ctx.set("findings", findings)


class FormatReportAgent:
    """Produces a final text report from findings."""

    async def run(self, ctx: Context) -> Context:
        findings: list[str] = ctx.get("findings", [])
        query = ctx.get("validated_query", "unknown topic")
        lines = [
            f"=== Research Report: {query} ===",
            "",
            *[f"  • {f}" for f in findings],
            "",
            f"Sources: {len(ctx.get('web_results', []))} web, "
            f"{len(ctx.get('academic_results', []))} academic",
            "=" * 40,
        ]
        report = "\n".join(lines)
        return ctx.set("report", report)


# ---------------------------------------------------------------------------
# Workflow definition
# ---------------------------------------------------------------------------


def build_research_workflow() -> Workflow:
    """Build and return the research pipeline workflow.

    DAG structure::

        validate ──┬── web_search ───┬── synthesize ── format_report
                   └── acad_search ──┘
    """
    return Workflow(
        name="research_pipeline",
        steps=[
            Step(
                name="validate",
                agent=ValidateQueryAgent(),
            ),
            Step(
                name="web_search",
                agent=WebSearchAgent(),
                retry=RetryPolicy(max_retries=2, backoff_seconds=0.1),
                deps={"validate"},
            ),
            Step(
                name="acad_search",
                agent=AcademicSearchAgent(),
                retry=RetryPolicy(max_retries=2, backoff_seconds=0.1),
                deps={"validate"},
            ),
            Step(
                name="synthesize",
                agent=SynthesizeAgent(),
                deps={"web_search", "acad_search"},
            ),
            Step(
                name="format_report",
                agent=FormatReportAgent(),
                deps={"synthesize"},
            ),
        ],
    )


async def main() -> None:
    workflow = build_research_workflow()
    engine = Engine()

    ctx = Context(data={"query": "  Large Language Model Agents  "})

    print("Running research pipeline...\n")
    trace = await engine.run(workflow, ctx)
    trace.print_summary()

    # Retrieve final report from state store (re-run to show it works)
    print()
    if trace.success:
        # Re-run to show the report output
        # In production you'd read from the state store or capture the final context
        final_ctx = ctx
        for step in workflow.steps:
            result = await step.agent.run(final_ctx)
            final_ctx = result
        print(final_ctx.get("report"))


if __name__ == "__main__":
    anyio.run(main)
