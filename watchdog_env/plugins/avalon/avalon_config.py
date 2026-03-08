"""Avalon plugin config: level, num_rounds from LEVEL_CONFIG."""

from __future__ import annotations

from dataclasses import dataclass

from watchdog_env.models import MultiAgentConfig

from watchdog_env.envs.avalon import LEVEL_CONFIG


@dataclass
class AvalonConfig(MultiAgentConfig):
    """Config for Avalon (Werewolf) multi-agent run."""

    level: int = 2

    def get_num_rounds(self) -> int:
        """Return num_rounds for this level."""
        cfg = LEVEL_CONFIG.get(self.level, LEVEL_CONFIG[2])
        return cfg.get("num_rounds", 2)
