"""Pluggable state storage for workflow executions.

The engine persists Context snapshots via a ``StateStore`` protocol so that
executions can be inspected, replayed, or recovered.  ``InMemoryStore`` ships
as the default implementation.

Example::

    from agentwire.state import InMemoryStore

    store = InMemoryStore()
    await store.save("run-1", "fetch", {"articles": [...]})
    snapshot = await store.load("run-1", "fetch")
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StateStore(Protocol):
    """Protocol for persisting step-level state snapshots.

    Example::

        class RedisStore:
            async def save(self, execution_id: str, step_name: str, data: dict[str, Any]) -> None:
                ...
            async def load(self, execution_id: str, step_name: str) -> dict[str, Any] | None:
                ...
            async def list_steps(self, execution_id: str) -> list[str]:
                ...
    """

    async def save(self, execution_id: str, step_name: str, data: dict[str, Any]) -> None: ...

    async def load(self, execution_id: str, step_name: str) -> dict[str, Any] | None: ...

    async def list_steps(self, execution_id: str) -> list[str]: ...


class InMemoryStore:
    """In-memory implementation of ``StateStore`` for development and testing.

    Example::

        store = InMemoryStore()
        await store.save("run-1", "step-a", {"x": 1})
        data = await store.load("run-1", "step-a")
        assert data == {"x": 1}
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict[str, Any]]] = {}

    async def save(self, execution_id: str, step_name: str, data: dict[str, Any]) -> None:
        """Persist a snapshot for the given execution and step.

        Example::

            await store.save("run-1", "fetch", {"results": [1, 2, 3]})
        """
        self._data.setdefault(execution_id, {})[step_name] = data

    async def load(self, execution_id: str, step_name: str) -> dict[str, Any] | None:
        """Load a previously saved snapshot, or ``None`` if not found.

        Example::

            data = await store.load("run-1", "fetch")
        """
        return self._data.get(execution_id, {}).get(step_name)

    async def list_steps(self, execution_id: str) -> list[str]:
        """List all step names that have saved state for an execution.

        Example::

            steps = await store.list_steps("run-1")
            assert steps == ["fetch", "summarize"]
        """
        return list(self._data.get(execution_id, {}).keys())
