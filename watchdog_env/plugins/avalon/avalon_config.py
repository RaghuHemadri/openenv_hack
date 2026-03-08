"""Avalon plugin config: level, num_rounds, LEVEL_CONFIG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from watchdog_env.models import MultiAgentConfig


# ─── Level Config ───────────────────────────────────────────────────

LEVEL_CONFIG: dict[int, dict[str, Any]] = {
    1: {"num_rounds": 2, "max_difficulty": 1, "clean_ratio": 0.50},
    2: {"num_rounds": 2, "max_difficulty": 2, "clean_ratio": 0.40},
    3: {"num_rounds": 3, "max_difficulty": 3, "clean_ratio": 0.30},
    4: {"num_rounds": 3, "max_difficulty": 3, "clean_ratio": 0.35},
}


@dataclass
class AvalonConfig(MultiAgentConfig):
    """Config for Avalon (Werewolf) multi-agent run."""

    level: int = 2

    def get_num_rounds(self) -> int:
        """Return num_rounds for this level."""
        cfg = LEVEL_CONFIG.get(self.level, LEVEL_CONFIG[2])
        return cfg.get("num_rounds", 2)
