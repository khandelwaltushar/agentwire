"""Tests for agentwire.trace."""

from agentwire.trace import ExecutionTrace, StepResult, StepStatus


class TestStepStatus:
    def test_values(self) -> None:
        assert StepStatus.SUCCESS == "success"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"


class TestStepResult:
    def test_basic(self) -> None:
        r = StepResult(step_name="s1", status=StepStatus.SUCCESS, duration_ms=10.0)
        assert r.step_name == "s1"
        assert r.error is None
        assert r.children == []

    def test_with_error(self) -> None:
        r = StepResult(step_name="s1", status=StepStatus.FAILED, error="boom")
        assert r.error == "boom"

    def test_with_children(self) -> None:
        child = StepResult(step_name="c1", status=StepStatus.SUCCESS)
        parent = StepResult(step_name="p1", status=StepStatus.SUCCESS, children=[child])
        assert len(parent.children) == 1


class TestExecutionTrace:
    def test_success(self) -> None:
        trace = ExecutionTrace(
            workflow_name="test",
            steps=[
                StepResult(step_name="a", status=StepStatus.SUCCESS),
                StepResult(step_name="b", status=StepStatus.SUCCESS),
            ],
        )
        assert trace.success is True

    def test_failure(self) -> None:
        trace = ExecutionTrace(
            workflow_name="test",
            steps=[
                StepResult(step_name="a", status=StepStatus.SUCCESS),
                StepResult(step_name="b", status=StepStatus.FAILED, error="oops"),
            ],
        )
        assert trace.success is False

    def test_empty_trace(self) -> None:
        trace = ExecutionTrace(workflow_name="empty", steps=[])
        assert trace.success is True  # vacuously true

    def test_print_summary(self, capsys: object) -> None:
        trace = ExecutionTrace(
            workflow_name="demo",
            steps=[
                StepResult(step_name="fetch", status=StepStatus.SUCCESS, duration_ms=50.0),
                StepResult(
                    step_name="fail",
                    status=StepStatus.FAILED,
                    duration_ms=10.0,
                    error="timeout",
                ),
            ],
            total_duration_ms=60.0,
        )
        trace.print_summary()
        # Just verify it doesn't crash — output tested by eye
