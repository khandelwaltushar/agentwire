#!/usr/bin/env python3
"""Intelligent router demo — classify a request then route to a specialist.

A keyword-based classifier inspects the user's message and routes it to
one of three stub agents: **billing**, **technical**, or **general**.
Each agent returns a structured response.  No LLM, no API keys.

Usage:
    python examples/intelligent_router.py "I can't connect to the API"
    python examples/intelligent_router.py "Where is my invoice?"
    python examples/intelligent_router.py "What are your office hours?"
"""

from __future__ import annotations

import sys

import anyio

from agentwire.engine import Engine, Step, Workflow
from agentwire.observability import TraceRenderer
from agentwire.types import Context

# ── Keyword sets for classification ──────────────────────────────────

BILLING_KEYWORDS = {
    "invoice", "charge", "billing", "payment", "refund",
    "subscription", "plan", "price", "cost", "receipt", "credit",
}
TECHNICAL_KEYWORDS = {
    "error", "bug", "crash", "api", "connect", "timeout", "login",
    "password", "install", "deploy", "server", "latency", "ssl",
    "certificate", "endpoint", "404", "500", "debug",
}


# ── Agents ───────────────────────────────────────────────────────────


class ClassifyAgent:
    """Keyword-based intent classifier — no LLM required."""

    async def run(self, ctx: Context) -> Context:
        message = str(ctx.get("message", "")).lower()
        tokens = set(message.split())

        billing_hits = tokens & BILLING_KEYWORDS
        tech_hits = tokens & TECHNICAL_KEYWORDS

        if len(billing_hits) >= len(tech_hits) and billing_hits:
            category = "billing"
        elif tech_hits:
            category = "technical"
        else:
            category = "general"

        return (
            ctx.set("category", category)
            .set("matched_keywords", sorted(billing_hits | tech_hits))
        )


class BillingAgent:
    """Stub billing-support agent."""

    async def run(self, ctx: Context) -> Context:
        message = ctx.get("message", "")
        return ctx.set("response", {
            "department": "billing",
            "reply": (
                f"Thank you for your billing enquiry. "
                f"I've reviewed your message: \"{message}\". "
                f"A billing specialist will follow up within 24 hours."
            ),
            "actions": ["opened_ticket", "flagged_billing_team"],
        })


class TechnicalAgent:
    """Stub technical-support agent."""

    async def run(self, ctx: Context) -> Context:
        message = ctx.get("message", "")
        keywords = ctx.get("matched_keywords", [])
        return ctx.set("response", {
            "department": "technical",
            "reply": (
                f"I see you're experiencing a technical issue "
                f"(detected: {', '.join(keywords)}). "
                f'Your message: "{message}". '
                f"Escalating to engineering on-call."
            ),
            "actions": ["opened_ticket", "paged_oncall"],
        })


class GeneralAgent:
    """Stub general-enquiry agent."""

    async def run(self, ctx: Context) -> Context:
        message = ctx.get("message", "")
        return ctx.set("response", {
            "department": "general",
            "reply": (
                f"Thanks for reaching out! "
                f'Regarding "{message}" — '
                f"a team member will respond shortly."
            ),
            "actions": ["opened_ticket"],
        })


# ── Routing ──────────────────────────────────────────────────────────


def _classify_router(ctx: Context) -> list[str]:
    """Route to the specialist agent matching the classified category."""
    category = ctx.get("category", "general")
    return [f"handle_{category}"]


# ── Workflow ─────────────────────────────────────────────────────────


def build_workflow() -> Workflow:
    """Construct the intelligent-router workflow."""
    return Workflow(
        name="intelligent-router",
        steps={
            "classify": Step(
                name="classify",
                agent=ClassifyAgent(),
                next_steps=_classify_router,
            ),
            "handle_billing": Step(
                name="handle_billing",
                agent=BillingAgent(),
            ),
            "handle_technical": Step(
                name="handle_technical",
                agent=TechnicalAgent(),
            ),
            "handle_general": Step(
                name="handle_general",
                agent=GeneralAgent(),
            ),
        },
        start="classify",
    )


# ── Main ─────────────────────────────────────────────────────────────


async def main(message: str) -> None:
    """Classify and route a support message."""
    workflow = build_workflow()
    ctx = Context(data={"message": message})

    print(f"\n{'=' * 60}")
    print(f"  Intelligent Router — message: {message!r}")
    print(f"{'=' * 60}\n")

    trace = await Engine().run(workflow, ctx)

    # ── Trace tree ───────────────────────────────────────────
    print("Execution trace:")
    TraceRenderer().print(trace)
    print()

    # ── Classification detail ────────────────────────────────
    classify_result = trace.results[0]
    cls_ctx = classify_result.output_context
    if cls_ctx is not None:
        print(f"  Category : {cls_ctx.get('category')}")
        print(f"  Keywords : {cls_ctx.get('matched_keywords')}")

    # ── Response ─────────────────────────────────────────────
    handler_result = trace.results[-1]
    out = handler_result.output_context
    if out is not None:
        resp = out.get("response", {})
        print(f"\n  Department: {resp.get('department')}")
        print(f"  Reply     : {resp.get('reply')}")
        print(f"  Actions   : {resp.get('actions')}")

    print()


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "I can't connect to the API"
    anyio.run(main, msg)
