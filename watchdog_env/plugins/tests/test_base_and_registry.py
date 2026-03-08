"""Tests for base plugin interface and registry. No API key required."""

from __future__ import annotations

import pytest

from watchdog_env.plugins.base import (
    AgentTurn,
    ContextMessage,
    MultiAgentConfig,
    MultiAgentState,
    MultiAgentStep,
    MultiAgentSystemPlugin,
    append_to_context,
    clear_system_context,
    get_system_context,
)
from watchdog_env.plugins.registry import get_plugin, get_registry, list_game_ids, register


class ContextAwarePlugin(MultiAgentSystemPlugin):
    """Plugin that uses append_to_context in generate_step to accumulate context."""

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
            system_context=[],
        )

    def generate_step(self, seed: int | None, step_index: int) -> MultiAgentStep:
        append_to_context(self._state, "user", f"user_step_{step_index}")
        append_to_context(self._state, "assistant", f"assistant_step_{step_index}")
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
                system_context=list(self._state.system_context),
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
            system_context=[],
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
                system_context=list(self._state.system_context),
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


def test_context_message_dataclass():
    """ContextMessage has role and content; supports all three roles."""
    for role in ("system", "user", "assistant"):
        msg = ContextMessage(role=role, content=f"content_{role}")
        assert msg.role == role
        assert msg.content == f"content_{role}"


def test_system_context_access_and_clear():
    """get_system_context, append_to_context, clear_system_context work."""
    state = MultiAgentState()
    assert get_system_context(state) == []

    append_to_context(state, "system", "You are a helpful assistant.")
    append_to_context(state, "user", "Hello")
    append_to_context(state, "assistant", "Hi there!")
    ctx = get_system_context(state)
    assert len(ctx) == 3
    assert ctx[0].role == "system" and ctx[0].content == "You are a helpful assistant."
    assert ctx[1].role == "user" and ctx[1].content == "Hello"
    assert ctx[2].role == "assistant" and ctx[2].content == "Hi there!"

    clear_system_context(state)
    assert get_system_context(state) == []


def test_append_to_context_mutates_state():
    """append_to_context mutates state.system_context in place."""
    state = MultiAgentState()
    append_to_context(state, "user", "msg1")
    assert state.system_context is get_system_context(state)
    assert len(state.system_context) == 1
    append_to_context(state, "assistant", "msg2")
    assert len(state.system_context) == 2


def test_clear_system_context():
    """clear_system_context empties context; idempotent on empty."""
    state = MultiAgentState()
    append_to_context(state, "user", "x")
    clear_system_context(state)
    assert get_system_context(state) == []
    clear_system_context(state)
    assert get_system_context(state) == []


def test_reset_clears_system_context():
    """Plugin reset creates fresh state with empty system_context."""
    register(MinimalPlugin())
    plugin = get_plugin("minimal")
    assert plugin is not None
    plugin.reset(seed=1, config=None)
    assert len(get_system_context(plugin.get_state())) == 0


def test_step_state_includes_system_context():
    """MultiAgentStep.state includes system_context when plugin returns it."""
    register(MinimalPlugin())
    plugin = get_plugin("minimal")
    assert plugin is not None
    plugin.reset(seed=1, config=None)
    step = plugin.generate_step(seed=1, step_index=0)
    assert step.state is not None
    assert hasattr(step.state, "system_context")
    assert isinstance(step.state.system_context, list)


def test_context_accumulates_across_steps():
    """Plugin that uses append_to_context accumulates context across generate_step calls."""
    register(ContextAwarePlugin())
    plugin = get_plugin("context_aware")
    assert plugin is not None
    plugin.reset(seed=1, config=None)
    assert len(get_system_context(plugin.get_state())) == 0

    plugin.generate_step(seed=1, step_index=0)
    ctx = get_system_context(plugin.get_state())
    assert len(ctx) == 2
    assert ctx[0].role == "user" and ctx[0].content == "user_step_0"
    assert ctx[1].role == "assistant" and ctx[1].content == "assistant_step_0"

    plugin.generate_step(seed=1, step_index=1)
    ctx = get_system_context(plugin.get_state())
    assert len(ctx) == 4
    assert ctx[2].content == "user_step_1"
    assert ctx[3].content == "assistant_step_1"


def test_context_cleared_on_reset():
    """Context accumulated across steps is cleared when reset is called."""
    register(ContextAwarePlugin())
    plugin = get_plugin("context_aware")
    assert plugin is not None
    plugin.reset(seed=1, config=None)
    plugin.generate_step(seed=1, step_index=0)
    assert len(get_system_context(plugin.get_state())) == 2

    plugin.reset(seed=99, config=None)
    assert len(get_system_context(plugin.get_state())) == 0


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
