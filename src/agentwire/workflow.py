"""Workflow — a validated directed acyclic graph of Steps.

A Workflow is constructed from a list of Steps.  On construction it validates
that the graph is a DAG (no cycles, no missing dependencies) and computes the
topological execution order.

Example::

    from agentwire.workflow import Workflow
    from agentwire.step import Step

    wf = Workflow(
        name="research",
        steps=[fetch_step, summarize_step, report_step],
    )
    order = wf.execution_order()  # topologically sorted groups
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentwire.step import Step


class WorkflowError(Exception):
    """Raised when a workflow definition is invalid.

    Example::

        try:
            Workflow(name="bad", steps=[])
        except WorkflowError:
            pass
    """


class Workflow:
    """A validated DAG of Steps ready for execution.

    Args:
        name: Human-readable workflow name (used in traces).
        steps: List of Step objects forming the DAG.

    Raises:
        WorkflowError: If the graph has cycles, missing deps, or duplicate names.

    Example::

        wf = Workflow(name="pipeline", steps=[step_a, step_b])
        for group in wf.execution_order():
            print([s.name for s in group])
    """

    __slots__ = ("name", "steps", "_step_map")

    def __init__(self, *, name: str, steps: list[Step]) -> None:
        self.name = name
        self.steps = list(steps)
        self._step_map: dict[str, Step] = {}
        self._validate()

    def _validate(self) -> None:
        if not self.steps:
            raise WorkflowError("Workflow must have at least one step")

        names: set[str] = set()
        for step in self.steps:
            if step.name in names:
                raise WorkflowError(f"Duplicate step name: {step.name!r}")
            names.add(step.name)
            self._step_map[step.name] = step

        for step in self.steps:
            missing = step.deps - names
            if missing:
                raise WorkflowError(f"Step {step.name!r} depends on unknown steps: {missing}")

        self._check_cycles()

    def _check_cycles(self) -> None:
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(name: str) -> None:
            if name in in_stack:
                raise WorkflowError(f"Cycle detected involving step {name!r}")
            if name in visited:
                return
            in_stack.add(name)
            for dep in self._step_map[name].deps:
                dfs(dep)
            in_stack.discard(name)
            visited.add(name)

        for step in self.steps:
            dfs(step.name)

    def get_step(self, name: str) -> Step:
        """Look up a step by name.

        Example::

            step = wf.get_step("fetch")
        """
        return self._step_map[name]

    def execution_order(self) -> list[list[Step]]:
        """Return steps grouped into parallelizable tiers (Kahn's algorithm).

        Each inner list contains steps that can execute concurrently.
        The outer list is ordered: tier N must finish before tier N+1 starts.

        Example::

            tiers = wf.execution_order()
            # tiers[0] = steps with no deps (can run in parallel)
            # tiers[1] = steps whose deps are all in tier 0, etc.
        """
        in_degree: dict[str, int] = {s.name: 0 for s in self.steps}
        dependents: dict[str, list[str]] = {s.name: [] for s in self.steps}

        for step in self.steps:
            for dep in step.deps:
                dependents[dep].append(step.name)
                in_degree[step.name] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        tiers: list[list[Step]] = []

        while queue:
            tier = [self._step_map[n] for n in queue]
            tiers.append(tier)
            next_queue: list[str] = []
            for name in queue:
                for dep_name in dependents[name]:
                    in_degree[dep_name] -= 1
                    if in_degree[dep_name] == 0:
                        next_queue.append(dep_name)
            queue = next_queue

        return tiers
