"""Tests for agentwire.state — StateStore protocol and InMemoryStateStore."""

from __future__ import annotations

import pytest

from agentwire.engine import Agent, Engine, Step, Workflow
from agentwire.state import InMemoryStateStore, StateStore
from agentwire.types import Context, ExecutionTrace

# ── Helpers ──────────────────────────────────────────────────────────


class Adder(Agent):
    """Deterministic agent: adds a fixed value to a key."""

    def __init__(self, key: str, value: int) -> None:
        self.key = key
        self.value = value

    async def run(self, ctx: Context) -> Context:
        current = ctx.get(self.key, 0)
        return ctx.set(self.key, current + self.value)


class Concatenator(Agent):
    """Deterministic agent: appends a suffix to a string key."""

    def __init__(self, key: str, suffix: str) -> None:
        self.key = key
        self.suffix = suffix

    async def run(self, ctx: Context) -> Context:
        current = ctx.get(self.key, "")
        return ctx.set(self.key, current + self.suffix)


# ── Protocol conformance ─────────────────────────────────────────────


class TestProtocol:
    def test_inmemory_is_statestore(self):
        assert isinstance(InMemoryStateStore(), StateStore)


# ── Basic CRUD ───────────────────────────────────────────────────────


class TestSaveAndLoad:
    @pytest.mark.anyio
    async def test_save_and_load_roundtrip(self):
        store = InMemoryStateStore()
        trace = ExecutionTrace(workflow_name="w")
        await store.save("run-1", trace)
        loaded = await store.load("run-1")
        assert loaded.workflow_name == "w"

    @pytest.mark.anyio
    async def test_load_missing_raises_keyerror(self):
        store = InMemoryStateStore()
        with pytest.raises(KeyError, match="no-such-id"):
            await store.load("no-such-id")

    @pytest.mark.anyio
    async def test_list_executions_empty(self):
        store = InMemoryStateStore()
        assert await store.list_executions() == []

    @pytest.mark.anyio
    async def test_list_executions_returns_ids(self):
        store = InMemoryStateStore()
        trace = ExecutionTrace(workflow_name="w")
        await store.save("a", trace)
        await store.save("b", trace)
        ids = await store.list_executions()
        assert set(ids) == {"a", "b"}

    @pytest.mark.anyio
    async def test_save_overwrites(self):
        store = InMemoryStateStore()
        await store.save("x", ExecutionTrace(workflow_name="old"))
        await store.save("x", ExecutionTrace(workflow_name="new"))
        loaded = await store.load("x")
        assert loaded.workflow_name == "new"


# ── Replay ───────────────────────────────────────────────────────────


class TestReplay:
    @pytest.mark.anyio
    async def test_replay_produces_identical_data(self):
        """Deterministic agents must yield the same output data on replay."""
        wf = Workflow(
            name="add-chain",
            steps={
                "a": Step(name="a", agent=Adder("total", 10), next_steps=["b"]),
                "b": Step(name="b", agent=Adder("total", 5)),
            },
            start="a",
        )
        initial = Context()
        engine = Engine()

        # First run
        trace1 = await engine.run(wf, initial)
        assert trace1.success

        # Save with context + workflow
        store = InMemoryStateStore()
        await store.save(
            "run-1", trace1, initial_context=initial, workflow=wf,
        )

        # Replay
        trace2 = await store.replay("run-1", engine=engine)
        assert trace2.success

        # Same step names, same statuses, same output data
        assert len(trace2.results) == len(trace1.results)
        for r1, r2 in zip(trace1.results, trace2.results, strict=True):
            assert r1.step_name == r2.step_name
            assert r1.status == r2.status
            assert r1.output_context is not None
            assert r2.output_context is not None
            assert r1.output_context.data == r2.output_context.data

    @pytest.mark.anyio
    async def test_replay_three_step_chain(self):
        """Replay a longer chain and verify final accumulated state."""
        wf = Workflow(
            name="concat",
            steps={
                "s1": Step(
                    name="s1",
                    agent=Concatenator("msg", "hello"),
                    next_steps=["s2"],
                ),
                "s2": Step(
                    name="s2",
                    agent=Concatenator("msg", "-world"),
                    next_steps=["s3"],
                ),
                "s3": Step(name="s3", agent=Concatenator("msg", "!")),
            },
            start="s1",
        )
        initial = Context()
        engine = Engine()
        store = InMemoryStateStore()

        trace1 = await engine.run(wf, initial)
        await store.save("r1", trace1, initial_context=initial, workflow=wf)

        trace2 = await store.replay("r1", engine=engine)

        final1 = trace1.results[-1].output_context
        final2 = trace2.results[-1].output_context
        assert final1 is not None
        assert final2 is not None
        assert final1.data == final2.data
        assert final1.data["msg"] == "hello-world!"

    @pytest.mark.anyio
    async def test_replay_with_initial_data(self):
        """Replay preserves the initial context that was saved."""
        wf = Workflow(
            name="inc",
            steps={"x": Step(name="x", agent=Adder("n", 1))},
            start="x",
        )
        initial = Context(data={"n": 100})
        engine = Engine()
        store = InMemoryStateStore()

        trace1 = await engine.run(wf, initial)
        await store.save("r1", trace1, initial_context=initial, workflow=wf)
        trace2 = await store.replay("r1", engine=engine)

        out1 = trace1.results[0].output_context
        out2 = trace2.results[0].output_context
        assert out1 is not None
        assert out2 is not None
        assert out1.data == out2.data
        assert out1.data["n"] == 101

    @pytest.mark.anyio
    async def test_replay_fan_out_deterministic(self):
        """Parallel fan-out with deterministic agents produces same data."""
        wf = Workflow(
            name="fan",
            steps={
                "root": Step(
                    name="root",
                    agent=Adder("base", 1),
                    next_steps=["left", "right"],
                ),
                "left": Step(name="left", agent=Adder("l", 10)),
                "right": Step(name="right", agent=Adder("r", 20)),
            },
            start="root",
        )
        initial = Context()
        engine = Engine()
        store = InMemoryStateStore()

        trace1 = await engine.run(wf, initial)
        await store.save("r1", trace1, initial_context=initial, workflow=wf)
        trace2 = await store.replay("r1", engine=engine)

        # Collect final data from last two results (parallel steps)
        data1 = {}
        data2 = {}
        for r in trace1.results:
            if r.output_context:
                data1.update(r.output_context.data)
        for r in trace2.results:
            if r.output_context:
                data2.update(r.output_context.data)

        assert data1 == data2
        assert data1["l"] == 10
        assert data1["r"] == 20

    @pytest.mark.anyio
    async def test_replay_missing_context_raises(self):
        store = InMemoryStateStore()
        await store.save("x", ExecutionTrace(workflow_name="w"))
        with pytest.raises(KeyError, match="initial context"):
            await store.replay("x")

    @pytest.mark.anyio
    async def test_replay_missing_workflow_raises(self):
        store = InMemoryStateStore()
        await store.save(
            "x",
            ExecutionTrace(workflow_name="w"),
            initial_context=Context(),
        )
        with pytest.raises(KeyError, match="workflow"):
            await store.replay("x")

    @pytest.mark.anyio
    async def test_replay_uses_default_engine(self):
        """Replay works without explicitly passing an engine."""
        wf = Workflow(
            name="simple",
            steps={"a": Step(name="a", agent=Adder("v", 1))},
            start="a",
        )
        store = InMemoryStateStore()
        initial = Context()
        trace = await Engine().run(wf, initial)
        await store.save("r1", trace, initial_context=initial, workflow=wf)

        replayed = await store.replay("r1")
        assert replayed.success
        assert replayed.results[0].output_context is not None
        assert replayed.results[0].output_context.data["v"] == 1
