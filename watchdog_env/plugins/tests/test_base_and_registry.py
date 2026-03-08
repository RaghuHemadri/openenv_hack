"""Tests for base plugin interface and registry. No API key required."""

from __future__ import annotations

import pytest

from watchdog_env.plugins.base import (
    AgentTurn,
    MultiAgentConfig,
    MultiAgentState,
    MultiAgentStep,
    MultiAgentSystemPlugin,
    append_to_conversation_log,
    clear_conversation_log,
    get_conversation_log,
)
from watchdog_env.plugins.registry import get_plugin, get_registry, list_game_ids, register


class ContextAwarePlugin(MultiAgentSystemPlugin):
    """Plugin that uses append_to_conversation_log in generate_step to accumulate context."""

    def __init__(self) -> None:
        self._state = MultiAgentState()

    def get_game_id(self) -> str:
        return "context_aware"

    def reset(
        self,
        seed: int | None = None,
        config: MultiAgentConfig | None = None,
    ) -> None:
        self._state = MultiAgentState(
            step_index=0,
            turns_so_far=[],
            config=config,
            done=False,
            conversation_log=[],
        )

    def generate_step(self, seed: int | None, step_index: int) -> MultiAgentStep:
        append_to_conversation_log(
            self._state,
            speaker_id="agent_1",
            speaker_display="Agent 1",
            message=f"Step {step_index} response",
        )
        turn = AgentTurn(
            agent_id="agent_1",
            action_text=f"Step {step_index}",
            step_index=step_index,
        )
        done = step_index >= 1
        self._state.step_index = step_index + 1
        self._state.turns_so_far.append(turn)
        self._state.done = done
        return MultiAgentStep(
            turns=[turn],
            done=done,
            step_index=step_index,
            game_id=self.get_game_id(),
            state=MultiAgentState(
                step_index=self._state.step_index,
                turns_so_far=list(self._state.turns_so_far),
                config=self._state.config,
                done=self._state.done,
                conversation_log=list(self._state.conversation_log),
            ),
        )

    def get_state(self) -> MultiAgentState:
        return self._state

    def get_display_name(self) -> str:
        return "Context-Aware Test Plugin"

    def list_agent_ids(self) -> list[str]:
        return ["agent_1"]


class MinimalPlugin(MultiAgentSystemPlugin):
    """Minimal plugin that implements all base methods for testing."""

    def __init__(self) -> None:
        self._state = MultiAgentState()

    def get_game_id(self) -> str:
        return "minimal"

    def reset(
        self,
        seed: int | None = None,
        config: MultiAgentConfig | None = None,
    ) -> None:
        self._state = MultiAgentState(
            step_index=0,
            turns_so_far=[],
            config=config,
            done=False,
            conversation_log=[],
        )

    def generate_step(self, seed: int | None, step_index: int) -> MultiAgentStep:
        turn = AgentTurn(
            agent_id="agent_1",
            action_text=f"Step {step_index} message.",
            step_index=step_index,
        )
        done = step_index >= 1
        self._state.step_index = step_index + 1
        self._state.turns_so_far.append(turn)
        self._state.done = done
        return MultiAgentStep(
            turns=[turn],
            done=done,
            step_index=step_index,
            game_id=self.get_game_id(),
            state=MultiAgentState(
                step_index=self._state.step_index,
                turns_so_far=list(self._state.turns_so_far),
                config=self._state.config,
                done=self._state.done,
                conversation_log=list(self._state.conversation_log),
            ),
        )

    def get_state(self) -> MultiAgentState:
        return self._state

    def get_display_name(self) -> str:
        return "Minimal Test Plugin"

    def list_agent_ids(self) -> list[str]:
        return ["agent_1"]


def test_register_and_get_plugin():
    """Register a plugin and retrieve it by game_id."""
    plugin = MinimalPlugin()
    register(plugin)
    assert get_plugin("minimal") is plugin
    assert "minimal" in list_game_ids()


def test_get_plugin_nonexistent():
    """get_plugin returns None for unknown game_id."""
    assert get_plugin("nonexistent") is None


def test_reset_then_generate_step():
    """After reset, generate_step(seed, 0) returns the first step."""
    register(MinimalPlugin())
    plugin = get_plugin("minimal")
    assert plugin is not None
    plugin.reset(seed=42, config=None)
    step = plugin.generate_step(seed=42, step_index=0)
    assert step.step_index == 0
    assert len(step.turns) == 1
    assert step.turns[0].agent_id == "agent_1"
    assert "Step 0" in step.turns[0].action_text
    assert step.state is not None
    assert plugin.get_state().step_index == 1


def test_conversation_log_entry_structure():
    """Conversation log entries have speaker_id, speaker_display, message."""
    state = MultiAgentState()
    append_to_conversation_log(
        state, speaker_id="a1", speaker_display="Agent 1", message="Hello"
    )
    log = get_conversation_log(state)
    assert len(log) == 1
    assert log[0]["speaker_id"] == "a1"
    assert log[0]["speaker_display"] == "Agent 1"
    assert log[0]["message"] == "Hello"


def test_conversation_log_access_and_clear():
    """get_conversation_log, append_to_conversation_log, clear_conversation_log work."""
    state = MultiAgentState()
    assert get_conversation_log(state) == []

    append_to_conversation_log(
        state, speaker_id="a1", speaker_display="Agent 1", message="Hello"
    )
    append_to_conversation_log(
        state, speaker_id="a2", speaker_display="Agent 2", message="Hi there!"
    )
    log = get_conversation_log(state)
    assert len(log) == 2
    assert log[0]["message"] == "Hello"
    assert log[1]["message"] == "Hi there!"

    clear_conversation_log(state)
    assert get_conversation_log(state) == []


def test_append_to_conversation_log_mutates_state():
    """append_to_conversation_log mutates state.conversation_log in place."""
    state = MultiAgentState()
    append_to_conversation_log(
        state, speaker_id="a1", speaker_display="A1", message="msg1"
    )
    assert state.conversation_log is get_conversation_log(state)
    assert len(state.conversation_log) == 1
    append_to_conversation_log(
        state, speaker_id="a2", speaker_display="A2", message="msg2"
    )
    assert len(state.conversation_log) == 2


def test_clear_conversation_log():
    """clear_conversation_log empties log; idempotent on empty."""
    state = MultiAgentState()
    append_to_conversation_log(
        state, speaker_id="a1", speaker_display="A1", message="x"
    )
    clear_conversation_log(state)
    assert get_conversation_log(state) == []
    clear_conversation_log(state)
    assert get_conversation_log(state) == []


def test_reset_clears_conversation_log():
    """Plugin reset creates fresh state with empty conversation_log."""
    register(MinimalPlugin())
    plugin = get_plugin("minimal")
    assert plugin is not None
    plugin.reset(seed=1, config=None)
    assert len(get_conversation_log(plugin.get_state())) == 0


def test_step_state_includes_conversation_log():
    """MultiAgentStep.state includes conversation_log when plugin returns it."""
    register(MinimalPlugin())
    plugin = get_plugin("minimal")
    assert plugin is not None
    plugin.reset(seed=1, config=None)
    step = plugin.generate_step(seed=1, step_index=0)
    assert step.state is not None
    assert hasattr(step.state, "conversation_log")
    assert isinstance(step.state.conversation_log, list)


def test_conversation_log_accumulates_across_steps():
    """Plugin that uses append_to_conversation_log accumulates context across steps."""
    register(ContextAwarePlugin())
    plugin = get_plugin("context_aware")
    assert plugin is not None
    plugin.reset(seed=1, config=None)
    assert len(get_conversation_log(plugin.get_state())) == 0

    plugin.generate_step(seed=1, step_index=0)
    log = get_conversation_log(plugin.get_state())
    assert len(log) == 1
    assert log[0]["speaker_id"] == "agent_1"
    assert "Step 0" in log[0]["message"]

    plugin.generate_step(seed=1, step_index=1)
    log = get_conversation_log(plugin.get_state())
    assert len(log) == 2
    assert "Step 1" in log[1]["message"]


def test_conversation_log_cleared_on_reset():
    """Context accumulated across steps is cleared when reset is called."""
    register(ContextAwarePlugin())
    plugin = get_plugin("context_aware")
    assert plugin is not None
    plugin.reset(seed=1, config=None)
    plugin.generate_step(seed=1, step_index=0)
    assert len(get_conversation_log(plugin.get_state())) == 1

    plugin.reset(seed=99, config=None)
    assert len(get_conversation_log(plugin.get_state())) == 0


def test_config_used_after_reset():
    """Plugin uses config passed to reset."""
    from dataclasses import dataclass

    @dataclass
    class ConfigWithNum(MultiAgentConfig):
        num: int = 2

    register(MinimalPlugin())
    plugin = get_plugin("minimal")
    assert plugin is not None
    plugin.reset(seed=1, config=ConfigWithNum(num=5))
    state = plugin.get_state()
    assert state.config is not None
    assert getattr(state.config, "num", None) == 5
