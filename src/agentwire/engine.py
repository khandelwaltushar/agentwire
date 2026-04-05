"""Orchestration engine — executes a workflow graph of steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import anyio
import structlog

from agentwire.types import Context, ExecutionTrace, StepResult, StepStatus

log: structlog.stdlib.BoundLogger = structlog.get_logger()


# ── Protocols & config ───────────────────────────────────────────────


@runtime_checkable
class Agent(Protocol):
    """Any object with an async ``run`` method qualifies."""

    async def run(self, ctx: Context) -> Context:
        """Execute the agent's logic and return a new context."""
        ...


@dataclass(frozen=True)
class RetryConfig:
    """Per-step retry policy."""

    max_attempts: int = 1
    backoff_seconds: float = 0.0


# ── Step & Workflow ──────────────────────────────────────────────────

# next_steps is either a static list (fan-out) or a callable that
# inspects the output context and returns the list (conditional branching).
NextResolver = list[str] | Callable[[Context], list[str]]


@dataclass(frozen=True)
class Step:
    """A named unit of work inside a workflow."""

    name: str
    agent: Agent
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    next_steps: NextResolver = field(default_factory=list)


@dataclass(frozen=True)
class Workflow:
    """A directed graph of steps with a single entry point."""

    name: str
    steps: dict[str, Step]
    start: str

    def __post_init__(self) -> None:  # noqa: D105
        if self.start not in self.steps:
            msg = f"start step {self.start!r} not found in steps"
            raise ValueError(msg)


# ── Engine ───────────────────────────────────────────────────────────


class Engine:
    """Execute a :class:`Workflow` and return an :class:`ExecutionTrace`."""

    async def run(self, workflow: Workflow, ctx: Context) -> ExecutionTrace:
        """Run *workflow* starting from *ctx*, return the full trace."""
        trace = ExecutionTrace(workflow_name=workflow.name)
        frontier: list[str] = [workflow.start]

        while frontier:
            if len(frontier) == 1:
                ctx, trace = await self._run_single(
                    workflow, frontier[0], ctx, trace,
                )
                if not trace.success:
                    break
                frontier = self._resolve_next(
                    workflow.steps[frontier[0]], ctx,
                )
            else:
                ctx, trace, frontier = await self._run_parallel(
                    workflow, frontier, ctx, trace,
                )
                if not trace.success:
                    break

        return trace.finish()

    # ── internals ────────────────────────────────────────────────────

    async def _run_single(
        self,
        workflow: Workflow,
        step_name: str,
        ctx: Context,
        trace: ExecutionTrace,
    ) -> tuple[Context, ExecutionTrace]:
        step = workflow.steps[step_name]
        result = await self._execute_step(step, ctx)
        trace = trace.add(result)
        out = result.output_context if result.output_context is not None else ctx
        return out, trace

    async def _run_parallel(
        self,
        workflow: Workflow,
        step_names: list[str],
        ctx: Context,
        trace: ExecutionTrace,
    ) -> tuple[Context, ExecutionTrace, list[str]]:
        results: dict[str, StepResult] = {}

        async with anyio.create_task_group() as tg:
            for name in step_names:

                async def _go(s: Step = workflow.steps[name]) -> None:
                    results[s.name] = await self._execute_step(s, ctx)

                tg.start_soon(_go)

        # Merge outputs and collect next frontier.
        merged = ctx
        all_next: list[str] = []
        for name in step_names:
            result = results[name]
            trace = trace.add(result)
            if result.status == StepStatus.SUCCESS and result.output_context:
                merged = Context(
                    data={**merged.data, **result.output_context.data},
                    metadata={
                        **merged.metadata,
                        **result.output_context.metadata,
                    },
                )
                for n in self._resolve_next(workflow.steps[name], merged):
                    if n not in all_next:
                        all_next.append(n)

        return merged, trace, all_next

    async def _execute_step(self, step: Step, ctx: Context) -> StepResult:
        """Run a step's agent with retry/backoff."""
        started = datetime.now(UTC)
        last_error = ""

        for attempt in range(step.retry_config.max_attempts):
            try:
                await log.adebug(
                    "step.attempt",
                    step=step.name,
                    attempt=attempt + 1,
                )
                output = await step.agent.run(ctx)
                return StepResult(
                    step_name=step.name,
                    status=StepStatus.SUCCESS,
                    input_context=ctx,
                    output_context=output,
                    started_at=started,
                    finished_at=datetime.now(UTC),
                    retries=attempt,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                await log.awarning(
                    "step.error",
                    step=step.name,
                    attempt=attempt + 1,
                    error=last_error,
                )
                if attempt < step.retry_config.max_attempts - 1:
                    await anyio.sleep(step.retry_config.backoff_seconds)

        return StepResult(
            step_name=step.name,
            status=StepStatus.FAILED,
            input_context=ctx,
            error=last_error,
            started_at=started,
            finished_at=datetime.now(UTC),
            retries=step.retry_config.max_attempts - 1,
        )

    @staticmethod
    def _resolve_next(step: Step, ctx: Context) -> list[str]:
        if callable(step.next_steps):
            return step.next_steps(ctx)
        return list(step.next_steps)
