"""Tests for base plugin interface and registry. No API key required."""

from __future__ import annotations

import pytest

from watchdog_env.plugins.base import (
    AgentTurn,
    MultiAgentConfig,
    MultiAgentState,
    MultiAgentStep,
    MultiAgentSystemPlugin,
)
from watchdog_env.plugins.registry import get_plugin, get_registry, list_game_ids, register


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
