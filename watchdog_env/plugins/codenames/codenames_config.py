"""Codenames plugin config: board size, team words, complexity level.

LLM backend is configured via WATCHDOG_LLM_BACKEND env var:
  - "local"  (default): shared Qwen3 8B game-play model
  - "gemini": Google Gemini (requires GEMINI_API_KEY or GOOGLE_API_KEY)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from watchdog_env.plugins.base import MultiAgentConfig


CODENAMES_AGENTS = [
    "red_spymaster",
    "red_operative",
    "blue_spymaster",
    "blue_operative",
]

TeamType = Literal["red", "blue"]


@dataclass
class CodenamesConfig(MultiAgentConfig):
    """Config for a Codenames multi-agent game.
    
    LLM backend is configured via WATCHDOG_LLM_BACKEND env var:
      - "local"  (default): Qwen3 8B game-play model
      - "gemini": Google Gemini (requires API key)
    """

    board_size: int = 25
    red_words: int = 9
    blue_words: int = 8
    neutral_words: int = 7
    assassin_words: int = 1
    starting_team: TeamType = "red"
    max_turns: int = 20
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
