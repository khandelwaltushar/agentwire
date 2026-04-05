"""Tests for agentwire.engine."""

import pytest

from agentwire.context import Context
from agentwire.engine import Engine
from agentwire.state import InMemoryStore
from agentwire.step import RetryPolicy, Step
from agentwire.trace import StepStatus
from agentwire.workflow import Workflow


class _AppendAgent:
    """Agent that appends a value to a list in context."""

    def __init__(self, key: str, value: str) -> None:
        self._key = key
        self._value = value

    async def run(self, ctx: Context) -> Context:
        current: list[str] = list(ctx.get(self._key, []))
        current.append(self._value)
        return ctx.set(self._key, current)


class _SetAgent:
    """Agent that sets a key in context."""

    def __init__(self, key: str, value: object) -> None:
        self._key = key
        self._value = value

    async def run(self, ctx: Context) -> Context:
        return ctx.set(self._key, self._value)


class _FailAgent:
    """Agent that always raises."""

    def __init__(self, msg: str = "boom") -> None:
        self._msg = msg

    async def run(self, ctx: Context) -> Context:
        raise RuntimeError(self._msg)


class _FailThenSucceedAgent:
    """Agent that fails N times then succeeds."""

    def __init__(self, fail_count: int) -> None:
        self._fail_count = fail_count
        self._attempts = 0

    async def run(self, ctx: Context) -> Context:
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise RuntimeError(f"fail #{self._attempts}")
        return ctx.set("recovered", True)


class TestEngine:
    @pytest.fixture()
    def engine(self) -> Engine:
        return Engine(store=InMemoryStore(), execution_id_factory=lambda: "test-run")

    @pytest.mark.anyio()
    async def test_single_step(self, engine: Engine) -> None:
        wf = Workflow(
            name="simple",
            steps=[Step(name="greet", agent=_SetAgent("msg", "hello"))],
        )
        trace = await engine.run(wf, Context())
        assert trace.success
        assert len(trace.steps) == 1
        assert trace.steps[0].step_name == "greet"
        assert trace.steps[0].status == StepStatus.SUCCESS

    @pytest.mark.anyio()
    async def test_linear_chain(self, engine: Engine) -> None:
        wf = Workflow(
            name="chain",
            steps=[
                Step(name="a", agent=_AppendAgent("log", "a")),
                Step(name="b", agent=_AppendAgent("log", "b"), deps={"a"}),
                Step(name="c", agent=_AppendAgent("log", "c"), deps={"b"}),
            ],
        )
        trace = await engine.run(wf, Context())
        assert trace.success
        assert len(trace.steps) == 3

    @pytest.mark.anyio()
    async def test_parallel_fanout(self, engine: Engine) -> None:
        wf = Workflow(
            name="fanout",
            steps=[
                Step(name="root", agent=_SetAgent("base", True)),
                Step(name="left", agent=_SetAgent("left", True), deps={"root"}),
                Step(name="right", agent=_SetAgent("right", True), deps={"root"}),
            ],
        )
        trace = await engine.run(wf, Context())
        assert trace.success
        assert len(trace.steps) == 3

    @pytest.mark.anyio()
    async def test_step_failure_stops_execution(self, engine: Engine) -> None:
        wf = Workflow(
            name="fail",
            steps=[
                Step(name="ok", agent=_SetAgent("x", 1)),
                Step(name="bad", agent=_FailAgent(), deps={"ok"}),
                Step(name="after", agent=_SetAgent("y", 2), deps={"bad"}),
            ],
        )
        trace = await engine.run(wf, Context())
        assert not trace.success
        names = {s.step_name for s in trace.steps}
        assert "after" not in names  # stopped before reaching "after"

    @pytest.mark.anyio()
    async def test_retry_success(self) -> None:
        engine = Engine(
            store=InMemoryStore(),
            execution_id_factory=lambda: "retry-run",
        )
        agent = _FailThenSucceedAgent(fail_count=2)
        wf = Workflow(
            name="retry",
            steps=[
                Step(
                    name="flaky",
                    agent=agent,
                    retry=RetryPolicy(max_retries=3, backoff_seconds=0.01),
                ),
            ],
        )
        trace = await engine.run(wf, Context())
        assert trace.success
        assert agent._attempts == 3  # 2 fails + 1 success

    @pytest.mark.anyio()
    async def test_retry_exhausted(self) -> None:
        engine = Engine(
            store=InMemoryStore(),
            execution_id_factory=lambda: "exhaust-run",
        )
        wf = Workflow(
            name="exhaust",
            steps=[
                Step(
                    name="fail",
                    agent=_FailAgent("always"),
                    retry=RetryPolicy(max_retries=2, backoff_seconds=0.01),
                ),
            ],
        )
        trace = await engine.run(wf, Context())
        assert not trace.success
        assert trace.steps[0].error == "always"

    @pytest.mark.anyio()
    async def test_condition_skip(self, engine: Engine) -> None:
        wf = Workflow(
            name="cond",
            steps=[
                Step(
                    name="skipped",
                    agent=_SetAgent("x", 1),
                    condition=lambda ctx: False,
                ),
            ],
        )
        trace = await engine.run(wf, Context())
        assert trace.success
        assert trace.steps[0].status == StepStatus.SKIPPED

    @pytest.mark.anyio()
    async def test_condition_run(self, engine: Engine) -> None:
        wf = Workflow(
            name="cond",
            steps=[
                Step(
                    name="runs",
                    agent=_SetAgent("x", 1),
                    condition=lambda ctx: True,
                ),
            ],
        )
        trace = await engine.run(wf, Context())
        assert trace.success
        assert trace.steps[0].status == StepStatus.SUCCESS

    @pytest.mark.anyio()
    async def test_state_persistence(self) -> None:
        store = InMemoryStore()
        engine = Engine(store=store, execution_id_factory=lambda: "persist-run")
        wf = Workflow(
            name="persist",
            steps=[Step(name="s1", agent=_SetAgent("val", 42))],
        )
        await engine.run(wf, Context())
        data = await store.load("persist-run", "s1")
        assert data is not None
        assert data["val"] == 42

    @pytest.mark.anyio()
    async def test_execution_trace_duration(self, engine: Engine) -> None:
        wf = Workflow(
            name="timed",
            steps=[Step(name="s1", agent=_SetAgent("x", 1))],
        )
        trace = await engine.run(wf, Context())
        assert trace.total_duration_ms > 0

    @pytest.mark.anyio()
    async def test_custom_execution_id(self) -> None:
        store = InMemoryStore()
        engine = Engine(store=store)
        wf = Workflow(
            name="custom_id",
            steps=[Step(name="s1", agent=_SetAgent("x", 1))],
        )
        await engine.run(wf, Context(), execution_id="my-custom-id")
        data = await store.load("my-custom-id", "s1")
        assert data is not None
