"""Pluggable state persistence for workflow executions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agentwire.engine import Engine, Workflow

if TYPE_CHECKING:
    from agentwire.types import Context, ExecutionTrace


@runtime_checkable
class StateStore(Protocol):
    """Protocol for persisting and retrieving execution traces."""

    async def save(
        self,
        execution_id: str,
        trace: ExecutionTrace,
        *,
        initial_context: Context | None = None,
        workflow: Workflow | None = None,
    ) -> None:
        """Persist *trace* under *execution_id*."""
        ...

    async def load(self, execution_id: str) -> ExecutionTrace:
        """Return the trace for *execution_id*, or raise ``KeyError``."""
        ...

    async def list_executions(self) -> list[str]:
        """Return all stored execution IDs."""
        ...


class InMemoryStateStore:
    """In-process :class:`StateStore` backed by a plain dict."""

    def __init__(self) -> None:  # noqa: D107
        self._traces: dict[str, ExecutionTrace] = {}
        self._contexts: dict[str, Context] = {}
        self._workflows: dict[str, Workflow] = {}

    async def save(
        self,
        execution_id: str,
        trace: ExecutionTrace,
        *,
        initial_context: Context | None = None,
        workflow: Workflow | None = None,
    ) -> None:
        """Persist *trace* (and optional initial context / workflow)."""
        self._traces[execution_id] = trace
        if initial_context is not None:
            self._contexts[execution_id] = initial_context
        if workflow is not None:
            self._workflows[execution_id] = workflow

    async def load(self, execution_id: str) -> ExecutionTrace:
        """Return the trace for *execution_id*, or raise ``KeyError``."""
        try:
            return self._traces[execution_id]
        except KeyError:
            msg = f"No execution found with id {execution_id!r}"
            raise KeyError(msg) from None

    async def list_executions(self) -> list[str]:
        """Return all stored execution IDs."""
        return list(self._traces)

    async def replay(
        self,
        execution_id: str,
        engine: Engine | None = None,
    ) -> ExecutionTrace:
        """Re-run the workflow from the saved initial context.

        Requires that both ``initial_context`` and ``workflow`` were
        provided when the execution was originally saved.
        """
        if execution_id not in self._contexts:
            msg = f"No initial context saved for {execution_id!r}"
            raise KeyError(msg)
        if execution_id not in self._workflows:
            msg = f"No workflow saved for {execution_id!r}"
            raise KeyError(msg)

        ctx = self._contexts[execution_id]
        workflow = self._workflows[execution_id]
        runner = engine or Engine()
        return await runner.run(workflow, ctx)
