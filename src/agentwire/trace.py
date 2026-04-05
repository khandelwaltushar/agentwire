"""Execution trace types for workflow observability.

Every workflow execution produces an `ExecutionTrace` — a tree of `StepResult`
objects that records what happened, how long it took, and whether it succeeded.

Example::

    from agentwire.trace import StepResult, ExecutionTrace, StepStatus

    result = StepResult(
        step_name="summarize",
        status=StepStatus.SUCCESS,
        duration_ms=120.5,
    )
    trace = ExecutionTrace(steps=[result])
    trace.print_summary()
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class StepStatus(enum.StrEnum):
    """Outcome of a single step execution.

    Example::

        assert StepStatus.SUCCESS == "success"
    """

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepResult(BaseModel):
    """Record of a single step's execution.

    Example::

        result = StepResult(
            step_name="fetch",
            status=StepStatus.SUCCESS,
            duration_ms=45.2,
            output_keys=["articles"],
        )
    """

    model_config = ConfigDict(frozen=True)

    step_name: str
    status: StepStatus
    duration_ms: float = 0.0
    error: str | None = None
    output_keys: list[str] = []
    children: list[StepResult] = []
    metadata: dict[str, Any] = {}


class ExecutionTrace(BaseModel):
    """Complete trace of a workflow execution.

    Example::

        trace = ExecutionTrace(
            workflow_name="research",
            steps=[
                StepResult(step_name="fetch", status=StepStatus.SUCCESS, duration_ms=100),
                StepResult(step_name="summarize", status=StepStatus.SUCCESS, duration_ms=200),
            ],
        )
        trace.print_summary()
    """

    model_config = ConfigDict(frozen=True)

    workflow_name: str = ""
    steps: list[StepResult] = []
    total_duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        """True if every step succeeded.

        Example::

            assert trace.success  # all steps passed
        """
        return all(s.status != StepStatus.FAILED for s in self.steps)

    def print_summary(self) -> None:
        """Print a human-readable execution summary to stdout.

        Example::

            trace.print_summary()
            # Output:
            # Workflow: research (300.0ms) ✓
            #   fetch ............. 100.0ms ✓
            #   summarize ......... 200.0ms ✓
        """
        status_icon = "\u2713" if self.success else "\u2717"
        print(f"Workflow: {self.workflow_name} ({self.total_duration_ms:.1f}ms) {status_icon}")
        for step in self.steps:
            _print_step(step, indent=2)


def _print_step(step: StepResult, indent: int = 0) -> None:
    icon = "\u2713" if step.status == StepStatus.SUCCESS else "\u2717"
    pad = " " * indent
    dots = "." * max(1, 30 - len(step.step_name))
    print(f"{pad}{step.step_name} {dots} {step.duration_ms:.1f}ms {icon}")
    if step.error:
        print(f"{pad}  error: {step.error}")
    for child in step.children:
        _print_step(child, indent + 2)
