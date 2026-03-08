"""Base interface for multi-agent system plugins.

Primitive is a step; each step can have multi-agent turns. Plugins implement
MultiAgentSystemPlugin and use MultiAgentState (history) when generating each step.

Context is a plain conversation_log: list of {speaker_id, speaker_display, message, ...}.
Shared types (AgentTurn, MultiAgentStep, etc.) live in watchdog_env.models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from watchdog_env.models import (
    AgentTurn,
    ConversationLogEntry,
    MultiAgentConfig,
    MultiAgentState,
    MultiAgentStep,
)


def get_conversation_log(state: MultiAgentState) -> list[ConversationLogEntry]:
    """Return the conversation log (plain transcript) for this run."""
    return state.conversation_log


def append_to_conversation_log(
    state: MultiAgentState,
    speaker_id: str,
    speaker_display: str,
    message: str,
    moderator_prompt: str = "",
    **extra: Any,
) -> None:
    """Append an entry to the conversation log."""
    entry: ConversationLogEntry = {
        "speaker_id": speaker_id,
        "speaker_display": speaker_display,
        "message": message,
    }
    if moderator_prompt:
        entry["moderator_prompt"] = moderator_prompt
    entry.update(extra)
    state.conversation_log.append(entry)


def clear_conversation_log(state: MultiAgentState) -> None:
    """Clear the conversation log. Called from reset."""
    state.conversation_log.clear()


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
        """Start or restart the scenario with this seed and config. Clear state and conversation_log."""
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


__all__ = [
    "AgentTurn",
    "ConversationLogEntry",
    "MultiAgentConfig",
    "MultiAgentState",
    "MultiAgentStep",
    "MultiAgentSystemPlugin",
    "append_to_conversation_log",
    "clear_conversation_log",
    "get_conversation_log",
]
