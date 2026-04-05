"""Step — a node in the workflow DAG.

A Step wraps an Agent with execution metadata: retry policy, timeout,
and dependency edges.  Steps are composed into a Workflow.

Example::

    from agentwire.step import Step, RetryPolicy

    step = Step(
        name="fetch_articles",
        agent=my_agent,
        retry=RetryPolicy(max_retries=3, backoff_seconds=1.0),
        deps={"validate_query"},
    )
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from agentwire.context import Context

if TYPE_CHECKING:
    from agentwire.agent import Agent


class RetryPolicy(BaseModel):
    """Per-step retry configuration with exponential backoff.

    Example::

        policy = RetryPolicy(max_retries=3, backoff_seconds=0.5, backoff_multiplier=2.0)
        # Retries after 0.5s, 1.0s, 2.0s
    """

    model_config = ConfigDict(frozen=True)

    max_retries: int = 0
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0


class Step:
    """A named node in a workflow DAG that wraps an Agent.

    Args:
        name: Unique identifier for this step within a workflow.
        agent: The Agent instance that executes this step's logic.
        retry: Optional retry policy. Defaults to no retries.
        deps: Names of steps that must complete before this one runs.
        condition: Optional callable that decides whether to run this step
            based on the current context. Returns True to run, False to skip.

    Example::

        class FetchAgent:
            async def run(self, ctx: Context) -> Context:
                return ctx.set("data", [1, 2, 3])

        step = Step(
            name="fetch",
            agent=FetchAgent(),
            retry=RetryPolicy(max_retries=2),
            deps=set(),
        )
    """

    __slots__ = ("name", "agent", "retry", "deps", "condition")

    def __init__(
        self,
        *,
        name: str,
        agent: Agent,
        retry: RetryPolicy | None = None,
        deps: set[str] | None = None,
        condition: StepCondition | None = None,
    ) -> None:
        self.name = name
        self.agent = agent
        self.retry = retry or RetryPolicy()
        self.deps = deps or set()
        self.condition = condition

    def __repr__(self) -> str:
        return f"Step(name={self.name!r}, deps={self.deps!r})"


StepCondition = Callable[[Context], bool]
"""Type alias for step condition callables — ``(Context) -> bool``."""
