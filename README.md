# agentwire

**Lightweight, deterministic multi-agent orchestration engine.**

AWS Step Functions for LLM agents — transparent, debuggable, zero magic.

## Features

- **DAG-based workflows** — define steps as a directed acyclic graph with explicit dependencies
- **Immutable context** — each step returns a new context, enabling safe replay and debugging
- **Parallel fan-out** — steps with no dependency conflicts run concurrently via anyio task groups
- **Per-step retry** — configurable retry count, backoff, and multiplier per step
- **Execution traces** — every run produces a structured trace tree for observability
- **Pluggable state store** — swap InMemoryStore for Redis, Postgres, S3, etc.
- **Fully typed** — passes `mypy --strict` with zero errors
- **No magic** — no hidden registries, no decorators, no implicit LLM calls

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import anyio
from agentwire import Context, Engine, Step, Workflow


class GreetAgent:
    async def run(self, ctx: Context) -> Context:
        name = ctx.get("name", "World")
        return ctx.set("greeting", f"Hello, {name}!")


async def main() -> None:
    wf = Workflow(
        name="greet",
        steps=[Step(name="hello", agent=GreetAgent())],
    )
    engine = Engine()
    trace = await engine.run(wf, Context(data={"name": "agentwire"}))
    trace.print_summary()


anyio.run(main)
```

Output:

```
Workflow: greet (0.2ms) ✓
  hello .............................. 0.1ms ✓
```

## Architecture

```
Context ──► Step(Agent) ──► new Context
               │
               ├── retry policy
               ├── dependencies (DAG edges)
               └── optional condition

Workflow = validated DAG of Steps
Engine   = executor (parallel tiers, retry, state persistence, tracing)
```

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Context** | Immutable key-value bag passed between steps |
| **Agent** | Protocol: `async def run(ctx: Context) -> Context` |
| **Step** | Named DAG node wrapping an Agent + retry config + deps |
| **Workflow** | Validated DAG of Steps with topological ordering |
| **Engine** | Executes a Workflow, producing an ExecutionTrace |
| **StateStore** | Protocol for persisting step snapshots (pluggable) |
| **ExecutionTrace** | Tree of StepResult objects for observability |

### Parallel Execution

Steps with no dependency conflicts are grouped into tiers and run concurrently:

```python
Workflow(
    name="pipeline",
    steps=[
        Step(name="fetch", agent=FetchAgent()),
        Step(name="search_web", agent=WebAgent(), deps={"fetch"}),
        Step(name="search_db", agent=DbAgent(), deps={"fetch"}),
        Step(name="merge", agent=MergeAgent(), deps={"search_web", "search_db"}),
    ],
)
# Tier 0: [fetch]
# Tier 1: [search_web, search_db]  ← parallel
# Tier 2: [merge]
```

### Retry Policy

```python
from agentwire import Step, RetryPolicy

Step(
    name="flaky_api",
    agent=my_agent,
    retry=RetryPolicy(
        max_retries=3,
        backoff_seconds=1.0,
        backoff_multiplier=2.0,  # 1s, 2s, 4s
    ),
)
```

### Conditional Steps

```python
Step(
    name="notify",
    agent=SlackAgent(),
    condition=lambda ctx: ctx.get("severity") == "critical",
    deps={"analyze"},
)
```

## Example

See [examples/research_agent.py](examples/research_agent.py) for a complete multi-step research pipeline with parallel search and synthesis.

```bash
python examples/research_agent.py
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest --cov=agentwire

# Type checking
mypy --strict src/

# Linting
ruff check src/ tests/
ruff format src/ tests/
```

## Dependencies

- **pydantic** v2 — data models and validation
- **anyio** — structured concurrency for parallel fan-out
- **structlog** — structured logging
- **httpx** — async HTTP (available for agent implementations)

## License

MIT
