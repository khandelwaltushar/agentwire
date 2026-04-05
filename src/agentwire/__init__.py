"""agentwire — lightweight, deterministic multi-agent orchestration.

AWS Step Functions for LLM agents: transparent, debuggable, zero magic.

Example::

    import anyio
    from agentwire import Context, Engine, Step, Workflow

    class HelloAgent:
        async def run(self, ctx: Context) -> Context:
            return ctx.set("greeting", f"Hello, {ctx.get('name')}!")

    async def main() -> None:
        wf = Workflow(
            name="greet",
            steps=[Step(name="hello", agent=HelloAgent())],
        )
        engine = Engine()
        trace = await engine.run(wf, Context(data={"name": "World"}))
        trace.print_summary()

    anyio.run(main)
"""

from agentwire.agent import Agent
from agentwire.context import Context
from agentwire.engine import Engine, EngineError
from agentwire.state import InMemoryStore, StateStore
from agentwire.step import RetryPolicy, Step
from agentwire.trace import ExecutionTrace, StepResult, StepStatus
from agentwire.workflow import Workflow, WorkflowError

__all__ = [
    "Agent",
    "Context",
    "Engine",
    "EngineError",
    "ExecutionTrace",
    "InMemoryStore",
    "RetryPolicy",
    "StateStore",
    "Step",
    "StepResult",
    "StepStatus",
    "Workflow",
    "WorkflowError",
]

__version__ = "0.1.0"
