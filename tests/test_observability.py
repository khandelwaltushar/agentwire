"""Tests for agentwire.observability — rendering, logging, serialisation."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from agentwire.observability import (
    StructuredLogger,
    TraceRenderer,
    trace_to_dict,
)
from agentwire.types import Context, ExecutionTrace, StepResult, StepStatus

# ── Helpers ──────────────────────────────────────────────────────────


def _make_result(
    name: str,
    status: StepStatus,
    *,
    retries: int = 0,
    error: str | None = None,
) -> StepResult:
    ctx = Context()
    return StepResult(
        step_name=name,
        status=status,
        input_context=ctx,
        output_context=ctx if status == StepStatus.SUCCESS else None,
        started_at=datetime(2025, 1, 1, tzinfo=UTC),
        finished_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        retries=retries,
        error=error,
    )


# ── TraceRenderer ────────────────────────────────────────────────────


class TestTraceRenderer:
    def test_single_success(self):
        trace = ExecutionTrace(
            workflow_name="my-wf",
            results=(_make_result("step-a", StepStatus.SUCCESS),),
        )
        got = TraceRenderer().render(trace)
        expected = (
            "my-wf [OK]\n"
            "\u2514\u2500\u2500 step-a [OK]\n"
        )
        assert got == expected

    def test_multiple_steps(self):
        trace = ExecutionTrace(
            workflow_name="pipeline",
            results=(
                _make_result("fetch", StepStatus.SUCCESS),
                _make_result("transform", StepStatus.SUCCESS),
                _make_result("load", StepStatus.SUCCESS),
            ),
        )
        got = TraceRenderer().render(trace)
        expected = (
            "pipeline [OK]\n"
            "\u251c\u2500\u2500 fetch [OK]\n"
            "\u251c\u2500\u2500 transform [OK]\n"
            "\u2514\u2500\u2500 load [OK]\n"
        )
        assert got == expected

    def test_failed_step(self):
        trace = ExecutionTrace(
            workflow_name="broken",
            results=(
                _make_result("ok-step", StepStatus.SUCCESS),
                _make_result(
                    "bad-step", StepStatus.FAILED, error="timeout",
                ),
            ),
        )
        got = TraceRenderer().render(trace)
        expected = (
            "broken [FAIL]\n"
            "\u251c\u2500\u2500 ok-step [OK]\n"
            "\u2514\u2500\u2500 bad-step [FAIL] err='timeout'\n"
        )
        assert got == expected

    def test_retries_shown(self):
        trace = ExecutionTrace(
            workflow_name="retry-wf",
            results=(
                _make_result("flaky", StepStatus.SUCCESS, retries=3),
            ),
        )
        got = TraceRenderer().render(trace)
        assert "(retries=3)" in got

    def test_skipped_step(self):
        trace = ExecutionTrace(
            workflow_name="skip-wf",
            results=(_make_result("skipped", StepStatus.SKIPPED),),
        )
        got = TraceRenderer().render(trace)
        assert "[SKIP]" in got

    def test_empty_trace(self):
        trace = ExecutionTrace(workflow_name="empty")
        got = TraceRenderer().render(trace)
        assert got == "empty [OK]\n"

    def test_print_writes_to_stdout(self, capsys):
        trace = ExecutionTrace(
            workflow_name="w",
            results=(_make_result("a", StepStatus.SUCCESS),),
        )
        TraceRenderer().print(trace)
        captured = capsys.readouterr()
        assert "w [OK]" in captured.out
        assert "a [OK]" in captured.out


# ── trace_to_dict ────────────────────────────────────────────────────


class TestTraceToDict:
    def test_basic_structure(self):
        ctx = Context(data={"k": "v"})
        result = StepResult(
            step_name="s1",
            status=StepStatus.SUCCESS,
            input_context=Context(),
            output_context=ctx,
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
            finished_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        )
        trace = ExecutionTrace(
            workflow_name="w",
            results=(result,),
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
            finished_at=datetime(2025, 1, 1, 0, 0, 2, tzinfo=UTC),
        )
        d = trace_to_dict(trace)
        assert d["workflow_name"] == "w"
        assert d["success"] is True
        assert d["started_at"] == "2025-01-01T00:00:00+00:00"
        assert d["finished_at"] == "2025-01-01T00:00:02+00:00"
        assert len(d["results"]) == 1

    def test_result_fields(self):
        result = _make_result("s", StepStatus.FAILED, error="oops", retries=2)
        trace = ExecutionTrace(workflow_name="w", results=(result,))
        d = trace_to_dict(trace)
        r = d["results"][0]
        assert r["step_name"] == "s"
        assert r["status"] == "failed"
        assert r["error"] == "oops"
        assert r["retries"] == 2

    def test_missing_output_context_omitted(self):
        result = _make_result("s", StepStatus.FAILED, error="e")
        trace = ExecutionTrace(workflow_name="w", results=(result,))
        d = trace_to_dict(trace)
        assert "output_context" not in d["results"][0]

    def test_unfinished_trace(self):
        trace = ExecutionTrace(workflow_name="w")
        d = trace_to_dict(trace)
        assert d["finished_at"] is None

    def test_context_data_preserved(self):
        ctx = Context(data={"a": 1, "b": [2, 3]}, metadata={"run": "x"})
        result = StepResult(
            step_name="s",
            status=StepStatus.SUCCESS,
            input_context=Context(),
            output_context=ctx,
        )
        trace = ExecutionTrace(workflow_name="w", results=(result,))
        d = trace_to_dict(trace)
        out = d["results"][0]["output_context"]
        assert out["data"] == {"a": 1, "b": [2, 3]}
        assert out["metadata"] == {"run": "x"}


# ── StructuredLogger ─────────────────────────────────────────────────


class TestStructuredLogger:
    @pytest.mark.anyio
    async def test_step_start_logs(self):
        logger = StructuredLogger()
        with patch.object(logger, "_log", new_callable=AsyncMock) as mock:
            mock.ainfo = AsyncMock()
            await logger.step_start("s1", Context(data={"k": "v"}))
            mock.ainfo.assert_called_once()
            _args, kwargs = mock.ainfo.call_args
            assert kwargs["step"] == "s1"
            assert kwargs["context_keys"] == ["k"]

    @pytest.mark.anyio
    async def test_step_end_logs(self):
        logger = StructuredLogger()
        result = _make_result("s1", StepStatus.SUCCESS, retries=1)
        with patch.object(logger, "_log", new_callable=AsyncMock) as mock:
            mock.ainfo = AsyncMock()
            await logger.step_end(result)
            mock.ainfo.assert_called_once()
            _args, kwargs = mock.ainfo.call_args
            assert kwargs["status"] == "success"
            assert kwargs["retries"] == 1

    @pytest.mark.anyio
    async def test_step_retry_logs(self):
        logger = StructuredLogger()
        with patch.object(logger, "_log", new_callable=AsyncMock) as mock:
            mock.awarning = AsyncMock()
            await logger.step_retry("s1", attempt=2, error="boom")
            mock.awarning.assert_called_once()
            _args, kwargs = mock.awarning.call_args
            assert kwargs["attempt"] == 2
            assert kwargs["error"] == "boom"

    @pytest.mark.anyio
    async def test_step_failure_logs(self):
        logger = StructuredLogger()
        result = _make_result("s1", StepStatus.FAILED, error="fatal", retries=3)
        with patch.object(logger, "_log", new_callable=AsyncMock) as mock:
            mock.aerror = AsyncMock()
            await logger.step_failure(result)
            mock.aerror.assert_called_once()
            _args, kwargs = mock.aerror.call_args
            assert kwargs["error"] == "fatal"
            assert kwargs["retries"] == 3
