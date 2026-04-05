"""Tests for agentwire.agent."""

from agentwire.agent import Agent
from agentwire.context import Context


class _ValidAgent:
    async def run(self, ctx: Context) -> Context:
        return ctx


class TestAgentProtocol:
    def test_isinstance_check(self) -> None:
        assert isinstance(_ValidAgent(), Agent)

    def test_non_agent(self) -> None:
        assert not isinstance("not an agent", Agent)
