"""Tests for agentwire.workflow."""

import pytest

from agentwire.context import Context
from agentwire.step import Step
from agentwire.workflow import Workflow, WorkflowError


class _DummyAgent:
    async def run(self, ctx: Context) -> Context:
        return ctx


def _step(name: str, deps: set[str] | None = None) -> Step:
    return Step(name=name, agent=_DummyAgent(), deps=deps)


class TestWorkflow:
    def test_single_step(self) -> None:
        wf = Workflow(name="w", steps=[_step("a")])
        assert wf.name == "w"
        assert len(wf.steps) == 1

    def test_empty_raises(self) -> None:
        with pytest.raises(WorkflowError, match="at least one step"):
            Workflow(name="w", steps=[])

    def test_duplicate_name_raises(self) -> None:
        with pytest.raises(WorkflowError, match="Duplicate"):
            Workflow(name="w", steps=[_step("a"), _step("a")])

    def test_missing_dep_raises(self) -> None:
        with pytest.raises(WorkflowError, match="unknown"):
            Workflow(name="w", steps=[_step("a", deps={"missing"})])

    def test_cycle_raises(self) -> None:
        s1 = _step("a", deps={"b"})
        s2 = _step("b", deps={"a"})
        with pytest.raises(WorkflowError, match="Cycle"):
            Workflow(name="w", steps=[s1, s2])

    def test_execution_order_linear(self) -> None:
        # a -> b -> c
        wf = Workflow(
            name="w",
            steps=[_step("a"), _step("b", {"a"}), _step("c", {"b"})],
        )
        tiers = wf.execution_order()
        assert len(tiers) == 3
        assert [t[0].name for t in tiers] == ["a", "b", "c"]

    def test_execution_order_parallel(self) -> None:
        # a -> b, a -> c (b and c can run in parallel)
        wf = Workflow(
            name="w",
            steps=[_step("a"), _step("b", {"a"}), _step("c", {"a"})],
        )
        tiers = wf.execution_order()
        assert len(tiers) == 2
        tier1_names = {s.name for s in tiers[1]}
        assert tier1_names == {"b", "c"}

    def test_get_step(self) -> None:
        wf = Workflow(name="w", steps=[_step("a"), _step("b", {"a"})])
        assert wf.get_step("a").name == "a"
        assert wf.get_step("b").name == "b"

    def test_diamond(self) -> None:
        # a -> b, a -> c, b -> d, c -> d
        wf = Workflow(
            name="w",
            steps=[
                _step("a"),
                _step("b", {"a"}),
                _step("c", {"a"}),
                _step("d", {"b", "c"}),
            ],
        )
        tiers = wf.execution_order()
        assert len(tiers) == 3
        assert tiers[0][0].name == "a"
        assert {s.name for s in tiers[1]} == {"b", "c"}
        assert tiers[2][0].name == "d"
