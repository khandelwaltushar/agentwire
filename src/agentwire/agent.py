"""Agent protocol — the unit of work in agentwire.

An Agent is anything with an async ``run`` method that transforms a Context.
agentwire never calls an LLM itself; your Agent implementation decides what
to do (call an API, run a local model, do pure computation, etc.).

Example::

    from agentwire.agent import Agent
    from agentwire.context import Context

    class Upper(Agent):
        async def run(self, ctx: Context) -> Context:
            text = ctx.get("text", "")
            return ctx.set("text", text.upper())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentwire.context import Context


@runtime_checkable
class Agent(Protocol):
    """Protocol that all agents must satisfy.

    Implement ``run`` to transform a Context into a new Context.

    Example::

        class MyAgent:
            async def run(self, ctx: Context) -> Context:
                return ctx.set("done", True)

        assert isinstance(MyAgent(), Agent)
    """

    async def run(self, ctx: Context) -> Context: ...
