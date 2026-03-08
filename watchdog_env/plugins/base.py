"""Base types and interface for multi-agent system plugins.

Primitive is a step; each step can have multi-agent turns. Plugins implement
MultiAgentSystemPlugin and use MultiAgentState (history) when generating each step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTurn:
    """A single agent's action in system text form."""

    agent_id: str
    action_text: str
    step_index: int = 0
    phase: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAgentConfig:
    """Base config for a multi-agent system run. Plugins subclass for game-specific fields."""

    pass


@dataclass
class MultiAgentState:
    """Tracks system behaviour across the run. Used when generating each MultiAgentStep."""

    step_index: int = 0
    turns_so_far: list[AgentTurn] = field(default_factory=list)
    config: MultiAgentConfig | None = None
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAgentStep:
    """One step: multiple agent turns. done=True means scenario is finished."""

    turns: list[AgentTurn]
    done: bool = False
    step_index: int = 0
    game_id: str = ""
    task_id: str = ""
    domain: str = ""
    state: MultiAgentState | None = None


class MultiAgentSystemPlugin(ABC):
    """Base interface for multi-agent system plugins. All methods must be implemented."""

    @abstractmethod
    def get_game_id(self) -> str:
        """Unique identifier for this game (e.g. 'cicero')."""
        ...

    @abstractmethod
    def reset(
        self,
        seed: int | None = None,
        config: MultiAgentConfig | None = None,
    ) -> None:
        """Start or restart the scenario with this seed and config. Clear state if stateful."""
        ...

    @abstractmethod
    def generate_step(self, seed: int | None, step_index: int) -> MultiAgentStep:
        """Produce one step based on the history of state (e.g. turns_so_far). Update state after step."""
        ...

    @abstractmethod
    def get_state(self) -> MultiAgentState:
        """Current system behaviour (step_index, turns_so_far, done, etc.)."""
        ...

    @abstractmethod
    def get_display_name(self) -> str:
        """Human-readable name for UI."""
        ...

    @abstractmethod
    def list_agent_ids(self) -> list[str]:
        """List of agent IDs (e.g. power names) for schema/docs."""
        ...
