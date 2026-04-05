"""Engine — the workflow executor.

The Engine runs a Workflow against a starting Context, producing an
ExecutionTrace.  It handles retry/backoff per-step, parallel fan-out via
anyio task groups, state persistence, and structured logging.

Example::

    from agentwire.engine import Engine
    from agentwire.state import InMemoryStore

    engine = Engine(store=InMemoryStore())
    trace = await engine.run(workflow, Context(data={"query": "AI safety"}))
    trace.print_summary()
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

import anyio
import structlog

from agentwire.state import InMemoryStore, StateStore
from agentwire.trace import ExecutionTrace, StepResult, StepStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentwire.context import Context
    from agentwire.step import Step
    from agentwire.workflow import Workflow

logger: structlog.stdlib.BoundLogger = structlog.get_logger()


class EngineError(Exception):
    """Raised when the engine encounters an unrecoverable execution error.

    Example::

        try:
            await engine.run(workflow, ctx)
        except EngineError as e:
            print(f"Execution failed: {e}")
    """


class Engine:
    """Execute a Workflow, producing an ExecutionTrace.

    Args:
        store: StateStore implementation for persisting step snapshots.
            Defaults to InMemoryStore.
        execution_id_factory: Callable that produces unique execution IDs.
            Defaults to uuid4 hex strings.

    Example::

        engine = Engine()
        trace = await engine.run(my_workflow, Context(data={"input": "hello"}))
        assert trace.success
    """

    def __init__(
        self,
        *,
        store: StateStore | None = None,
        execution_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._store: StateStore = store or InMemoryStore()
        self._id_factory = execution_id_factory or (lambda: uuid.uuid4().hex)

    async def run(
        self,
        workflow: Workflow,
        ctx: Context,
        *,
        execution_id: str | None = None,
    ) -> ExecutionTrace:
        """Execute *workflow* starting from *ctx* and return the trace.

        Args:
            workflow: The Workflow DAG to execute.
            ctx: Initial context.
            execution_id: Optional explicit execution ID (auto-generated if omitted).

        Returns:
            An ExecutionTrace recording every step's outcome.

        Example::

            trace = await engine.run(wf, Context(data={"x": 1}))
            for step_result in trace.steps:
                print(step_result.step_name, step_result.status)
        """
        eid = execution_id or self._id_factory()
        log = logger.bind(execution_id=eid, workflow=workflow.name)
        log.info("workflow.start")

        start = time.monotonic()
        all_results: list[StepResult] = []
        current_ctx = ctx

        tiers = workflow.execution_order()

        for tier in tiers:
            tier_results, current_ctx = await self._run_tier(tier, current_ctx, eid, log)
            all_results.extend(tier_results)

            # Stop on first failure
            if any(r.status == StepStatus.FAILED for r in tier_results):
                log.error("workflow.failed", tier=[s.name for s in tier])
                break

        elapsed = (time.monotonic() - start) * 1000
        trace = ExecutionTrace(
            workflow_name=workflow.name,
            steps=all_results,
            total_duration_ms=elapsed,
        )
        log.info("workflow.done", success=trace.success, duration_ms=f"{elapsed:.1f}")
        return trace

    async def _run_tier(
        self,
        tier: list[Step],
        ctx: Context,
        execution_id: str,
        log: Any,
    ) -> tuple[list[StepResult], Context]:
        """Run all steps in a tier concurrently, merge contexts."""
        if len(tier) == 1:
            result, new_ctx = await self._run_step(tier[0], ctx, execution_id, log)
            return [result], new_ctx

        results: list[StepResult] = []
        contexts: list[Context] = []

        async with anyio.create_task_group() as tg:
            for step in tier:

                async def _execute(s: Step = step) -> None:
                    r, c = await self._run_step(s, ctx, execution_id, log)
                    results.append(r)
                    contexts.append(c)

                tg.start_soon(_execute)

        merged = ctx
        for c in contexts:
            # Merge only new/changed keys from each parallel branch
            diff = {k: v for k, v in c.data.items() if k not in ctx.data or ctx.data[k] != v}
            merged = merged.merge(diff)

        return results, merged

    async def _run_step(
        self,
        step: Step,
        ctx: Context,
        execution_id: str,
        log: Any,
    ) -> tuple[StepResult, Context]:
        """Run a single step with retry logic."""
        log = log.bind(step=step.name)

        # Check condition
        if step.condition is not None and not step.condition(ctx):
            log.info("step.skipped", reason="condition_false")
            return (
                StepResult(
                    step_name=step.name,
                    status=StepStatus.SKIPPED,
                ),
                ctx,
            )

        attempts = step.retry.max_retries + 1
        last_error: Exception | None = None
        step_start = time.monotonic()

        for attempt in range(1, attempts + 1):
            log.info("step.start", attempt=attempt)
            attempt_start = time.monotonic()
            try:
                new_ctx = await step.agent.run(ctx)
                elapsed = (time.monotonic() - attempt_start) * 1000

                # Persist state
                await self._store.save(execution_id, step.name, new_ctx.data)

                new_keys = [k for k in new_ctx.data if k not in ctx.data]
                log.info("step.success", duration_ms=f"{elapsed:.1f}")
                return (
                    StepResult(
                        step_name=step.name,
                        status=StepStatus.SUCCESS,
                        duration_ms=elapsed,
                        output_keys=new_keys,
                    ),
                    new_ctx,
                )
            except Exception as exc:
                last_error = exc
                elapsed = (time.monotonic() - attempt_start) * 1000
                log.warning(
                    "step.error",
                    attempt=attempt,
                    error=str(exc),
                    duration_ms=f"{elapsed:.1f}",
                )
                if attempt < attempts:
                    backoff = step.retry.backoff_seconds * (
                        step.retry.backoff_multiplier ** (attempt - 1)
                    )
                    await anyio.sleep(backoff)

        elapsed_total = (time.monotonic() - step_start) * 1000
        error_msg = str(last_error) if last_error else "unknown error"
        log.error("step.failed", error=error_msg)
        return (
            StepResult(
                step_name=step.name,
                status=StepStatus.FAILED,
                duration_ms=elapsed_total,
                error=error_msg,
            ),
            ctx,
        )
