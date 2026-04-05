"""Tests for agentwire.state."""

import pytest

from agentwire.state import InMemoryStore, StateStore


class TestInMemoryStore:
    @pytest.fixture()
    def store(self) -> InMemoryStore:
        return InMemoryStore()

    @pytest.mark.anyio()
    async def test_save_and_load(self, store: InMemoryStore) -> None:
        await store.save("run-1", "step-a", {"x": 1})
        data = await store.load("run-1", "step-a")
        assert data == {"x": 1}

    @pytest.mark.anyio()
    async def test_load_missing(self, store: InMemoryStore) -> None:
        result = await store.load("nope", "nope")
        assert result is None

    @pytest.mark.anyio()
    async def test_list_steps(self, store: InMemoryStore) -> None:
        await store.save("run-1", "a", {"x": 1})
        await store.save("run-1", "b", {"y": 2})
        steps = await store.list_steps("run-1")
        assert set(steps) == {"a", "b"}

    @pytest.mark.anyio()
    async def test_list_steps_empty(self, store: InMemoryStore) -> None:
        steps = await store.list_steps("nope")
        assert steps == []

    @pytest.mark.anyio()
    async def test_overwrite(self, store: InMemoryStore) -> None:
        await store.save("run-1", "a", {"v": 1})
        await store.save("run-1", "a", {"v": 2})
        data = await store.load("run-1", "a")
        assert data == {"v": 2}

    def test_protocol_conformance(self) -> None:
        assert isinstance(InMemoryStore(), StateStore)
