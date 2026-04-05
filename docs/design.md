# Design Document

Architectural decisions, tradeoffs, and roadmap for agentwire.

---

## Why immutable Context

The most common source of bugs in multi-agent systems is shared mutable state.
When agent B reads a value that agent A set, then agent C overwrites it before
agent D runs, you get a debugging session that involves correlating log
timestamps across four concurrent tasks. We've been there. It's bad.

agentwire's `Context` is a pydantic model with `frozen=True`. Every mutation
method (`.set()`, `.set_meta()`) returns a **new** Context:

```python
ctx = Context(data={"x": 1})
ctx2 = ctx.set("x", 2)

assert ctx.data["x"] == 1    # original untouched
assert ctx2.data["x"] == 2   # new copy
```

This gives us three properties for free:

1. **Deterministic replay.** If you save the initial context and re-run the
   same workflow with the same agents, you get the same output data. There's
   no hidden mutation to miss.

2. **Safe parallel fan-out.** When steps B and C run concurrently from step A's
   output, they both receive the same frozen context. Neither can corrupt the
   other's input. The engine merges their outputs afterward with explicit dict
   merging — you can see exactly what happened.

3. **Debuggable traces.** Every `StepResult` in the `ExecutionTrace` stores
   both `input_context` and `output_context`. You can diff them to see exactly
   what a step changed, without log correlation.

### The tradeoff

Immutable contexts create more objects. Each `.set()` copies the data dict.
For typical agent workflows (5-20 steps, kilobytes of data), this is
negligible. If you're passing megabytes of data between steps, you should
be storing that data externally (S3, database) and passing a reference in
the context — which is the right architecture anyway.

---

## Why anyio over asyncio directly

We considered three options:

| | `asyncio` | `trio` | `anyio` |
|---|---|---|---|
| Standard library | Yes | No | No |
| Structured concurrency | No (`gather` is not structured) | Yes | Yes |
| Cancellation semantics | Fragile | Strict | Strict |
| Backend flexibility | asyncio only | trio only | Both |

We chose anyio because:

1. **Structured concurrency via task groups.** The engine's parallel fan-out
   uses `anyio.create_task_group()`. If any step in a fan-out raises, the
   task group cancels the siblings and propagates the exception. With raw
   `asyncio.gather`, you get `return_exceptions=True` and have to manually
   check which futures failed — or you get an unhandled exception that kills
   the event loop.

2. **Backend flexibility.** Users can run agentwire on asyncio (the default)
   or trio without changing their code. This matters for teams that have
   an existing trio codebase.

3. **Clean sleep/timeout APIs.** `anyio.sleep()` and `anyio.fail_after()`
   are cleaner than `asyncio.sleep()` + `asyncio.wait_for()` with their
   various edge cases around cancellation.

### The tradeoff

anyio is a dependency. It's well-maintained, has no transitive dependencies
beyond `sniffio`, and is used by httpx and Starlette. But it is not the
standard library. If you need zero dependencies beyond stdlib, use raw asyncio.

---

## Why the StateStore is a Protocol, not an ABC

```python
@runtime_checkable
class StateStore(Protocol):
    async def save(self, execution_id: str, trace: ExecutionTrace, ...) -> None: ...
    async def load(self, execution_id: str) -> ExecutionTrace: ...
    async def list_executions(self) -> list[str]: ...
```

We use a `typing.Protocol` (structural subtyping) instead of an
`abc.ABC` (nominal subtyping) for two reasons:

1. **No inheritance required.** If you have an existing class with `save`,
   `load`, and `list_executions` methods that match the signatures, it
   satisfies `StateStore` automatically. You don't need to modify it to
   inherit from our base class. This is important when integrating with
   existing infrastructure — your Redis wrapper or DynamoDB client shouldn't
   need to know about agentwire.

2. **Runtime checking.** The `@runtime_checkable` decorator means you can
   write `isinstance(my_store, StateStore)` in tests and assertions. This
   is a nice property that plain duck typing doesn't give you.

3. **No diamond inheritance.** ABCs in Python create coupling. If we have
   `StateStore(ABC)` and someone also inherits from another ABC (e.g., a
   framework's `BaseRepository`), MRO conflicts become possible. Protocols
   avoid this entirely because there's no inheritance relationship.

### The tradeoff

Protocols don't provide default method implementations. If we wanted a
`replay()` method on the base protocol with a default implementation, we'd
need to either make it a mixin or put it in the concrete class. We chose
to put `replay()` on `InMemoryStateStore` directly rather than on the
protocol, because replay semantics depend on having the initial context
and workflow stored — not all backends will choose to do this.

---

## Replay semantics and limitations

### How it works

```python
store = InMemoryStateStore()
trace = await engine.run(workflow, initial_ctx)
await store.save("run-1", trace, initial_context=initial_ctx, workflow=workflow)

replayed = await store.replay("run-1")
```

`replay()` calls `engine.run(saved_workflow, saved_initial_context)`. The
engine walks the same DAG from scratch. For deterministic agents (same input
always produces same output), the replayed trace will have identical step
results and output data.

### What "identical" means

Replay guarantees identical **output data** for deterministic agents. It does
**not** guarantee identical:

- **Timestamps.** `started_at` / `finished_at` will differ because the code
  runs at a different wall-clock time.
- **Execution order of parallel steps.** If steps B and C fan out in parallel,
  the trace may record them in a different order on replay. The merged context
  will be the same, but the ordering of `trace.results` for parallel tiers
  is not deterministic.
- **Non-deterministic agents.** If an agent calls an LLM, makes an HTTP
  request to a changing endpoint, or reads the current time, replay will
  produce different output. This is inherent — replay re-executes the code,
  it doesn't replay recorded responses.

### What you need for replay

Both `initial_context` and `workflow` must be provided when calling `save()`.
If either is missing, `replay()` raises `KeyError`. This is deliberate —
rather than silently returning a partial replay, we fail loudly.

### Limitations

- **No response recording.** We don't intercept and record HTTP responses
  or LLM outputs. True "tape replay" (replaying recorded responses without
  re-executing) is not implemented. This would require wrapping the agent's
  I/O layer, which conflicts with our "no magic" principle.
- **No partial replay.** You can't replay from step 5 of a 10-step workflow.
  Replay always starts from the beginning. Partial replay would require
  checkpointing intermediate contexts, which is a future consideration.
- **Workflow must be the same object.** The saved workflow reference includes
  the agent instances. If the agent code has changed between save and replay,
  replay runs the new code — not the code that produced the original trace.

---

## What agentwire is NOT good for

Being honest about limitations saves everyone time:

### Long-running workflows (hours/days)

agentwire runs workflows in a single process. There's no distributed task
queue, no worker pool, no persistence across process restarts. If your
workflow takes 6 hours and the process dies at hour 4, you start over.
**Use Prefect, Temporal, or Airflow** for long-running pipelines.

### High-throughput data processing

agentwire is optimized for agent orchestration (tens of steps, seconds to
minutes per step). It's not designed for processing millions of records
with fine-grained parallelism. **Use Dask, Ray, or Spark** for data
processing.

### Workflows that need human approval gates

There's no built-in mechanism for pausing a workflow, waiting for human
input, and resuming. The engine runs to completion in a single `await`.
**Use LangGraph's interrupt mechanism or Prefect's UI** if you need
human-in-the-loop today.

### Multi-language agent systems

agentwire is Python-only. If your agents are written in TypeScript, Go,
and Python, you need an orchestrator that speaks HTTP/gRPC between
services. **Use Temporal or a custom API gateway.**

### Teams that want a visual workflow builder

There's no UI, no drag-and-drop, no web dashboard. Workflows are defined
in Python code. If your team includes non-developers who need to build
workflows, **use Prefect, n8n, or Windmill**.

---

## Future roadmap

These are concrete next steps, not aspirational hand-waving:

### Redis StateStore

`InMemoryStateStore` doesn't survive process restarts. A Redis-backed
implementation would store serialized traces via `trace_to_dict()` /
`model_validate()`. The Protocol-based design means this is a new class,
not a modification to existing code.

```python
class RedisStateStore:
    def __init__(self, redis: Redis) -> None: ...
    async def save(self, execution_id, trace, **kw) -> None:
        await self.redis.set(f"trace:{execution_id}", trace_to_dict(trace))
    async def load(self, execution_id) -> ExecutionTrace:
        data = await self.redis.get(f"trace:{execution_id}")
        return ExecutionTrace.model_validate(json.loads(data))
```

### Distributed execution

Replace the in-process anyio task group with a task queue (e.g., Redis
streams, SQS). Each step becomes a message. A pool of workers picks up
messages and posts results back. The engine becomes a coordinator that
tracks frontier state rather than executing steps directly.

This is a significant architectural change but the core abstractions
(Step, Workflow, Context, ExecutionTrace) don't change — only the
Engine internals.

### Human-in-the-loop

Add a `PauseStep` type that serializes the current context to the
StateStore and returns a `StepStatus.PAUSED` result. A separate
`Engine.resume(execution_id, human_input)` method loads the paused
context, merges the human input, and continues from the next step.

This requires:

1. A new `StepStatus.PAUSED` enum value
2. Context serialization in the StateStore (already possible via
   `model_dump()`)
3. A `resume()` method on Engine that reconstructs the frontier

### Step-level timeouts

Wrap each step execution in `anyio.fail_after(step.timeout_seconds)`.
This is straightforward but needs careful interaction with the retry
logic — a timeout should count as a retryable failure, not an
unrecoverable crash.

### Conditional step skipping

Add an optional `condition: Callable[[Context], bool]` field to `Step`.
If the condition returns `False`, the step is recorded as `SKIPPED` in
the trace and the engine moves to `next_steps` with the input context
unchanged. This is useful for feature flags and A/B testing in agent
pipelines.
