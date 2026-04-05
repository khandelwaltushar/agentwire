"""Tests for agentwire.context."""

from agentwire.context import Context


class TestContext:
    def test_default_empty(self) -> None:
        ctx = Context()
        assert ctx.data == {}

    def test_init_with_data(self) -> None:
        ctx = Context(data={"x": 1})
        assert ctx.data == {"x": 1}

    def test_set_returns_new_context(self) -> None:
        ctx = Context(data={"a": 1})
        ctx2 = ctx.set("b", 2)
        assert ctx2.data == {"a": 1, "b": 2}
        assert ctx.data == {"a": 1}  # original unchanged

    def test_set_overwrites(self) -> None:
        ctx = Context(data={"a": 1})
        ctx2 = ctx.set("a", 99)
        assert ctx2.data["a"] == 99
        assert ctx.data["a"] == 1

    def test_merge(self) -> None:
        ctx = Context(data={"a": 1, "b": 2})
        ctx2 = ctx.merge({"b": 20, "c": 30})
        assert ctx2.data == {"a": 1, "b": 20, "c": 30}
        assert ctx.data == {"a": 1, "b": 2}

    def test_get_existing(self) -> None:
        ctx = Context(data={"key": "value"})
        assert ctx.get("key") == "value"

    def test_get_missing_default(self) -> None:
        ctx = Context()
        assert ctx.get("missing") is None
        assert ctx.get("missing", 42) == 42

    def test_frozen(self) -> None:
        ctx = Context(data={"a": 1})
        try:
            ctx.data = {"b": 2}  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except Exception:
            pass
