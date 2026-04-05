"""Tests for agentwire.step."""

from agentwire.context import Context
from agentwire.step import RetryPolicy, Step


class _DummyAgent:
    async def run(self, ctx: Context) -> Context:
        return ctx


class TestRetryPolicy:
    def test_defaults(self) -> None:
        p = RetryPolicy()
        assert p.max_retries == 0
        assert p.backoff_seconds == 1.0
        assert p.backoff_multiplier == 2.0

    def test_custom(self) -> None:
        p = RetryPolicy(max_retries=5, backoff_seconds=0.1, backoff_multiplier=3.0)
        assert p.max_retries == 5


class TestStep:
    def test_basic(self) -> None:
        agent = _DummyAgent()
        step = Step(name="s1", agent=agent)
        assert step.name == "s1"
        assert step.deps == set()
        assert step.retry.max_retries == 0
        assert step.condition is None

    def test_with_deps(self) -> None:
        step = Step(name="s2", agent=_DummyAgent(), deps={"s1"})
        assert step.deps == {"s1"}

    def test_repr(self) -> None:
        step = Step(name="s1", agent=_DummyAgent(), deps={"s0"})
        assert "s1" in repr(step)
        assert "s0" in repr(step)

    def test_with_condition(self) -> None:
        cond = lambda ctx: True  # noqa: E731
        step = Step(name="s1", agent=_DummyAgent(), condition=cond)
        assert step.condition is cond
