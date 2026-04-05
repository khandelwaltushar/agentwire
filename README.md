# agentwire

**Lightweight, deterministic multi-agent orchestration engine — AWS Step Functions for LLM agents.**

Zero magic. No hidden registries. No decorators that silently rewrite your code.
Just a typed DAG of steps, immutable context, and structured traces you can replay.

---

## Why this exists

If you've built production agent systems with the current crop of frameworks,
you've hit at least one of these:

### 1. "Where did that value come from?"

LangChain's shared mutable state means any node can mutate anything at any time.
Good luck debugging a 12-step chain where step 7 silently overwrote step 3's output:

```python
# LangChain — mutable shared state
class MyChain(Chain):
    def _call(self, inputs: dict) -> dict:
        inputs["result"] = self.llm(inputs["query"])  # mutates in place
        return inputs  # hope nobody else touched "result"
```

agentwire's Context is **frozen** (pydantic `frozen=True`). Each step gets
a context and returns a new one. The original is never touched:

```python
# agentwire — immutable context
async def run(self, ctx: Context) -> Context:
    result = await self.llm(ctx.get("query"))
    return ctx.set("result", result)  # returns new Context
```

### 2. "I can't replay this failure"

LangGraph checkpoints require you to wire up a persistence backend, configure
serializers, and hope your custom objects are picklable. Replaying a failed
run means reconstructing the exact graph state by hand:

```python
# LangGraph — checkpoint wiring just to get replay
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=memory)  # extra compile step
config = {"configurable": {"thread_id": "abc123"}}
# ... now manually re-invoke with the right config to "replay"
```

agentwire's `InMemoryStateStore.replay()` re-runs the saved workflow from the
saved initial context in one call:

```python
# agentwire — one-call replay
store = InMemoryStateStore()
await store.save("run-1", trace, initial_context=ctx, workflow=wf)
replayed = await store.replay("run-1")  # that's it
```

### 3. "What just happened?"

Most frameworks give you logs. Structured, sure — but logs.
You have to reconstruct the execution flow by correlating timestamps.
agentwire gives you an `ExecutionTrace` — a typed, immutable object you can
render, serialize, diff, or assert against in tests:

```
research-agent [OK]
├── plan [OK]
├── search [OK]
├── summarise [OK]
├── critique [OK]
├── refine [OK]
├── summarise [OK]
└── critique [OK]
```

---

## Architecture

```
                         ┌──────────────────────────────────────────┐
                         │              Workflow (DAG)              │
                         │                                         │
     Context ──────►     │   ┌──────┐    ┌──────┐    ┌──────┐     │  ──────► ExecutionTrace
    (immutable)          │   │Step A│───►│Step B│──┬►│Step D│     │    (typed, replayable)
                         │   └──────┘    └──────┘  │ └──────┘     │
                         │                         │              │
                         │                         └►┌──────┐     │
                         │                           │Step C│     │
                         │                           └──────┘     │
                         └──────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
              ┌──────────┐       ┌──────────────┐     ┌────────────┐
              │  Engine   │       │  StateStore  │     │Observability│
              │           │       │  (Protocol)  │     │            │
              │• retry    │       │• save/load   │     │• TraceTree │
              │• backoff  │       │• replay      │     │• JSON logs │
              │• fan-out  │       │• list        │     │• serialize │
              │  (anyio)  │       └──────────────┘     └────────────┘
              └──────────┘
```

**Core types** (`types.py`): `Context`, `StepStatus`, `StepResult`, `ExecutionTrace`

**Engine** (`engine.py`): `Agent` protocol, `Step`, `Workflow`, `Engine` — DAG walker
with per-step retry/backoff and anyio task-group fan-out

**State** (`state.py`): `StateStore` protocol, `InMemoryStateStore` — save, load,
list, replay

**Observability** (`observability.py`): `TraceRenderer` (tree view),
`StructuredLogger` (structlog JSON), `trace_to_dict()` (serialisation)

---

## Quickstart

```bash
pip install -e ".[dev]"

# Run the research agent (real DuckDuckGo search, no API keys)
python examples/research_agent.py "AI agent orchestration"

# Run the code generation loop (real subprocess execution)
python examples/code_gen_loop.py

# Run the intelligent router (keyword classifier → branching)
python examples/intelligent_router.py "I can't connect to the API"
```

### Minimal workflow in 30 lines

```python
import anyio
from agentwire.engine import Engine, Step, Workflow
from agentwire.observability import TraceRenderer
from agentwire.types import Context


class Greet:
    async def run(self, ctx: Context) -> Context:
        return ctx.set("greeting", f"Hello, {ctx.get('name')}!")


class Shout:
    async def run(self, ctx: Context) -> Context:
        return ctx.set("greeting", ctx.get("greeting", "").upper())


async def main() -> None:
    wf = Workflow(
        name="hello",
        steps={
            "greet": Step(name="greet", agent=Greet(), next_steps=["shout"]),
            "shout": Step(name="shout", agent=Shout()),
        },
        start="greet",
    )
    trace = await Engine().run(wf, Context(data={"name": "world"}))
    TraceRenderer().print(trace)

anyio.run(main)
```

Output:

```
hello [OK]
├── greet [OK]
└── shout [OK]
```

---

## Comparison

| | **agentwire** | **LangGraph** | **Prefect** | **Raw asyncio** |
|---|---|---|---|---|
| **Target** | LLM agent DAGs | LLM agent graphs | General data pipelines | Anything |
| **Graph model** | DAG of Steps | StateGraph with edges | Task DAG with decorators | Manual |
| **Context** | Immutable, typed | Mutable shared dict | Parameter passing | Manual |
| **Parallelism** | anyio task groups | Conditional edges | Dask/concurrent | `asyncio.gather` |
| **Replay** | Built-in (one call) | Checkpoint-based | Re-run from UI | Manual |
| **Observability** | Typed ExecutionTrace | LangSmith (paid) | Prefect UI (hosted) | Logging |
| **Dependencies** | 4 (pydantic, anyio, structlog, httpx) | 10+ | 30+ | 0 |
| **Typing** | mypy strict | Partial | Partial | Up to you |
| **Learning curve** | Low — just implement `run()` | Medium — StateGraph API | Medium — decorators, server | None |
| **Production readiness** | Alpha | Mature | Mature | Depends |
| **Distributed execution** | Not yet | Not built-in | Yes (workers) | Manual |
| **Human-in-the-loop** | Not yet | Built-in | Via UI | Manual |
| **Ecosystem** | Minimal | Large (LangChain) | Large | Python stdlib |

**Where agentwire wins**: transparency, debuggability, minimal dependencies, type safety.

**Where others win**: LangGraph has a larger ecosystem and built-in human-in-the-loop.
Prefect has distributed workers and a production UI. Raw asyncio has zero overhead
and zero opinions. If you need any of those today, use them.

agentwire is for teams that want to understand exactly what their agent
orchestrator is doing, and are willing to trade ecosystem size for clarity.

---

## Design

See [docs/design.md](docs/design.md) for architectural decisions, tradeoffs,
replay semantics, and the roadmap.

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type check
mypy

# Lint
ruff check .
ruff format --check .
```

---

## License

MIT
