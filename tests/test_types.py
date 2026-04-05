"""Tests for agentwire.types — Context immutability and core types."""

from datetime import UTC, datetime

from agentwire.types import Context, ExecutionTrace, StepResult, StepStatus


class TestContextImmutability:
    def test_set_returns_new_object(self):
        ctx = Context(data={"a": 1})
        new_ctx = ctx.set("b", 2)
        assert new_ctx is not ctx
        assert new_ctx.data == {"a": 1, "b": 2}
        assert ctx.data == {"a": 1}  # original unchanged

    def test_set_overwrites_existing_key(self):
        ctx = Context(data={"x": "old"})
        new_ctx = ctx.set("x", "new")
        assert new_ctx.data["x"] == "new"
        assert ctx.data["x"] == "old"

    def test_set_meta_returns_new_object(self):
        ctx = Context(metadata={"run_id": "abc"})
        new_ctx = ctx.set_meta("run_id", "xyz")
        assert new_ctx is not ctx
        assert new_ctx.metadata["run_id"] == "xyz"
        assert ctx.metadata["run_id"] == "abc"

    def test_frozen_rejects_direct_assignment(self):
        import pytest  # noqa: PLC0415

        ctx = Context()
        with pytest.raises(Exception):  # noqa: B017, PT011
            ctx.data = {"hacked": True}  # type: ignore[misc]

    def test_get_returns_default(self):
        ctx = Context()
        assert ctx.get("missing") is None
        assert ctx.get("missing", 42) == 42

    def test_get_returns_value(self):
        ctx = Context(data={"key": "val"})
        assert ctx.get("key") == "val"

    def test_empty_context(self):
        ctx = Context()
        assert ctx.data == {}
        assert ctx.metadata == {}


class TestStepStatus:
    def test_values(self):
        assert StepStatus.PENDING == "pending"
        assert StepStatus.RUNNING == "running"
        assert StepStatus.SUCCESS == "success"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"


class TestStepResult:
    def test_minimal_result(self):
        ctx = Context()
        result = StepResult(
            step_name="step1",
            status=StepStatus.SUCCESS,
            input_context=ctx,
            output_context=ctx.set("done", True),
        )
        assert result.step_name == "step1"
        assert result.error is None
        assert result.retries == 0

    def test_failed_result(self):
        ctx = Context()
        result = StepResult(
            step_name="bad",
            status=StepStatus.FAILED,
            input_context=ctx,
            error="boom",
            retries=3,
        )
        assert result.status == StepStatus.FAILED
        assert result.error == "boom"
        assert result.retries == 3


class TestExecutionTrace:
    def test_empty_trace_is_successful(self):
        trace = ExecutionTrace(workflow_name="w")
        assert trace.success is True

    def test_add_returns_new_trace(self):
        ctx = Context()
        trace = ExecutionTrace(workflow_name="w")
        result = StepResult(
            step_name="s1",
            status=StepStatus.SUCCESS,
            input_context=ctx,
            output_context=ctx,
        )
        new_trace = trace.add(result)
        assert new_trace is not trace
        assert len(new_trace.results) == 1
        assert len(trace.results) == 0

    def test_success_false_when_any_failed(self):
        ctx = Context()
        trace = ExecutionTrace(workflow_name="w")
        trace = trace.add(
            StepResult(
                step_name="ok",
                status=StepStatus.SUCCESS,
                input_context=ctx,
                output_context=ctx,
            ),
        )
        trace = trace.add(
            StepResult(
                step_name="bad",
                status=StepStatus.FAILED,
                input_context=ctx,
                error="fail",
            ),
        )
        assert trace.success is False

    def test_skipped_does_not_count_as_failure(self):
        ctx = Context()
        trace = ExecutionTrace(workflow_name="w")
        trace = trace.add(
            StepResult(
                step_name="skip",
                status=StepStatus.SKIPPED,
                input_context=ctx,
            ),
        )
        assert trace.success is True

    def test_finish_stamps_time(self):
        trace = ExecutionTrace(workflow_name="w")
        assert trace.finished_at is None
        finished = trace.finish()
        assert finished.finished_at is not None
        assert finished.finished_at.tzinfo == UTC

    def test_started_at_defaults_to_utc(self):
        trace = ExecutionTrace(workflow_name="w")
        assert trace.started_at.tzinfo == UTC
        assert (datetime.now(UTC) - trace.started_at).total_seconds() < 2
