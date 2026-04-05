"""Immutable execution context passed between steps.

Context is the primary data carrier in agentwire. Each step receives a Context
and returns a new one — the original is never mutated, enabling safe replay
and debugging.

Example::

    ctx = Context(data={"query": "climate change"})
    next_ctx = ctx.set("results", ["paper1", "paper2"])
    assert "results" not in ctx.data  # original unchanged
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Context(BaseModel):
    """Immutable bag of key-value data threaded through a workflow.

    Example::

        ctx = Context(data={"user_id": 42})
        ctx2 = ctx.set("score", 0.95)
        ctx3 = ctx2.merge({"label": "positive", "score": 0.99})
    """

    model_config = ConfigDict(frozen=True)

    data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> Context:
        """Return a new Context with *key* set to *value*.

        Example::

            ctx = Context().set("x", 1)
            assert ctx.data["x"] == 1
        """
        return Context(data={**self.data, key: value})

    def merge(self, other: dict[str, Any]) -> Context:
        """Return a new Context with *other* merged in (other wins on conflict).

        Example::

            ctx = Context(data={"a": 1}).merge({"b": 2})
            assert ctx.data == {"a": 1, "b": 2}
        """
        return Context(data={**self.data, **other})

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key, with an optional default.

        Example::

            ctx = Context(data={"x": 10})
            assert ctx.get("x") == 10
            assert ctx.get("missing", 42) == 42
        """
        return self.data.get(key, default)
