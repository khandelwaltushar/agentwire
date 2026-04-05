"""Core types for the agentwire orchestration engine."""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Context(BaseModel, frozen=True):
    """Immutable execution context passed between steps.

    Each step receives a Context and returns a new one — the original
    is never mutated, which makes replay and debugging trivial.
    """

    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def set(self, key: str, value: Any) -> Context:
        """Return a new Context with *key* set to *value* in data."""
        return self.model_copy(update={"data": {**self.data, key: value}})

    def get(self, key: str, default: Any = None) -> Any:
        """Return a value from data, or *default* if missing."""
        return self.data.get(key, default)

    def set_meta(self, key: str, value: Any) -> Context:
        """Return a new Context with *key* set in metadata."""
        return self.model_copy(
            update={"metadata": {**self.metadata, key: value}},
        )


class StepStatus(enum.StrEnum):
    """Lifecycle status of a single step execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepResult(BaseModel, frozen=True):
    """Outcome of executing a single step."""

    step_name: str
    status: StepStatus
    input_context: Context
    output_context: Context | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    retries: int = 0


class ExecutionTrace(BaseModel, frozen=True):
    """Ordered record of every step result in a workflow run."""

    workflow_name: str
    results: tuple[StepResult, ...] = ()
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    finished_at: datetime | None = None

    @property
    def success(self) -> bool:
        """True when no step failed."""
        return all(r.status != StepStatus.FAILED for r in self.results)

    def add(self, result: StepResult) -> ExecutionTrace:
        """Return a new trace with *result* appended."""
        return self.model_copy(
            update={"results": (*self.results, result)},
        )

    def finish(self) -> ExecutionTrace:
        """Return a new trace with *finished_at* stamped."""
        return self.model_copy(
            update={"finished_at": datetime.now(UTC)},
        )
