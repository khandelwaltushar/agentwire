"""Observability utilities — trace rendering, structured logging, serialisation."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import structlog

from agentwire.types import StepStatus

if TYPE_CHECKING:
    from datetime import datetime

    from agentwire.types import Context, ExecutionTrace, StepResult


# ── Serialisation ────────────────────────────────────────────────────


def _ctx_to_dict(ctx: Context) -> dict[str, Any]:
    return {"data": dict(ctx.data), "metadata": dict(ctx.metadata)}


def _dt_to_iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def step_result_to_dict(result: StepResult) -> dict[str, Any]:
    """Serialise a single :class:`StepResult` to a plain dict."""
    d: dict[str, Any] = {
        "step_name": result.step_name,
        "status": str(result.status),
        "input_context": _ctx_to_dict(result.input_context),
        "started_at": _dt_to_iso(result.started_at),
        "finished_at": _dt_to_iso(result.finished_at),
        "retries": result.retries,
    }
    if result.output_context is not None:
        d["output_context"] = _ctx_to_dict(result.output_context)
    if result.error is not None:
        d["error"] = result.error
    return d


def trace_to_dict(trace: ExecutionTrace) -> dict[str, Any]:
    """Serialise an :class:`ExecutionTrace` to a JSON-ready dict."""
    return {
        "workflow_name": trace.workflow_name,
        "started_at": trace.started_at.isoformat(),
        "finished_at": _dt_to_iso(trace.finished_at),
        "success": trace.success,
        "results": [step_result_to_dict(r) for r in trace.results],
    }


# ── Tree renderer ────────────────────────────────────────────────────

_STATUS_ICONS: dict[StepStatus, str] = {
    StepStatus.SUCCESS: "[OK]",
    StepStatus.FAILED: "[FAIL]",
    StepStatus.SKIPPED: "[SKIP]",
    StepStatus.RUNNING: "[RUN]",
    StepStatus.PENDING: "[..]",
}


class TraceRenderer:
    """Render an :class:`ExecutionTrace` as a ``tree``-style text view."""

    def render(self, trace: ExecutionTrace) -> str:
        """Return a multi-line tree string for *trace*."""
        buf = io.StringIO()
        status_tag = "OK" if trace.success else "FAIL"
        buf.write(f"{trace.workflow_name} [{status_tag}]\n")

        results = trace.results
        for idx, result in enumerate(results):
            is_last = idx == len(results) - 1
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            icon = _STATUS_ICONS.get(result.status, "[??]")
            line = f"{connector}{result.step_name} {icon}"
            if result.retries > 0:
                line += f" (retries={result.retries})"
            if result.error is not None:
                line += f" err={result.error!r}"
            buf.write(line + "\n")

        return buf.getvalue()

    def print(self, trace: ExecutionTrace) -> None:
        """Print the tree to stdout."""
        print(self.render(trace), end="")  # noqa: T201


# ── Structured logger ───────────────────────────────────────────────


class StructuredLogger:
    """Emit JSON-structured logs for step lifecycle events via structlog."""

    def __init__(self) -> None:  # noqa: D107
        self._log: structlog.stdlib.BoundLogger = structlog.get_logger()

    async def step_start(self, step_name: str, ctx: Context) -> None:
        """Log that a step is starting."""
        await self._log.ainfo(
            "step.start",
            step=step_name,
            context_keys=list(ctx.data.keys()),
        )

    async def step_end(self, result: StepResult) -> None:
        """Log that a step has finished (success or skip)."""
        await self._log.ainfo(
            "step.end",
            step=result.step_name,
            status=str(result.status),
            retries=result.retries,
        )

    async def step_retry(
        self, step_name: str, attempt: int, error: str,
    ) -> None:
        """Log a retry attempt."""
        await self._log.awarning(
            "step.retry",
            step=step_name,
            attempt=attempt,
            error=error,
        )

    async def step_failure(self, result: StepResult) -> None:
        """Log a terminal step failure."""
        await self._log.aerror(
            "step.failure",
            step=result.step_name,
            error=result.error,
            retries=result.retries,
        )
