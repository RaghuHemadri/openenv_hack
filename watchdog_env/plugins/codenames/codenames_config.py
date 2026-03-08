"""Codenames plugin config: board size, team words, complexity level, model settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from watchdog_env.plugins.base import MultiAgentConfig


CODENAMES_AGENTS = [
    "red_spymaster",
    "red_operative",
    "blue_spymaster",
    "blue_operative",
]

TeamType = Literal["red", "blue"]

# Default model - can be overridden via environment variable
DEFAULT_MODEL_NAME = os.environ.get("CODENAMES_MODEL", os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"))
DEFAULT_TEMPERATURE = float(os.environ.get("CODENAMES_TEMPERATURE", os.environ.get("GEMINI_TEMPERATURE", "0.7")))


@dataclass
class CodenamesConfig(MultiAgentConfig):
    """Config for a Codenames multi-agent game.
    
    Model configuration can be set via:
    1. Constructor arguments (highest priority)
    2. Environment variables: CODENAMES_MODEL, CODENAMES_TEMPERATURE
    3. Fallback env vars: GEMINI_MODEL, GEMINI_TEMPERATURE
    4. Default: gemini-2.0-flash-preview, temperature=0.7
    """

    board_size: int = 25
    red_words: int = 9
    blue_words: int = 8
    neutral_words: int = 7
    assassin_words: int = 1
    starting_team: TeamType = "red"
    max_turns: int = 20
    model_name: str = field(default_factory=lambda: DEFAULT_MODEL_NAME)
    temperature: float = field(default_factory=lambda: DEFAULT_TEMPERATURE)
    complexity_level: int = 2  # 1=basic, 2=medium, 3=complex word interactions
    
    def get_agents(self) -> list[str]:
        """Return the list of agent IDs."""
        return list(CODENAMES_AGENTS)
    
    def validate(self) -> None:
        """Validate configuration values."""
        total = self.red_words + self.blue_words + self.neutral_words + self.assassin_words
        if total != self.board_size:
            raise ValueError(
                f"Word counts ({total}) must equal board_size ({self.board_size})"
            )
        if self.starting_team not in ("red", "blue"):
            raise ValueError(f"starting_team must be 'red' or 'blue', got {self.starting_team}")
        if self.complexity_level not in (1, 2, 3):
            raise ValueError(f"complexity_level must be 1, 2, or 3, got {self.complexity_level}")
