"""Cicero plugin config: num_steps, powers, model, temperature."""

from __future__ import annotations

from dataclasses import dataclass

from watchdog_env.plugins.base import MultiAgentConfig

CICERO_POWERS = [
    "Austria-Hungary",
    "England",
    "France",
    "Germany",
    "Italy",
    "Russia",
    "Turkey",
]


@dataclass
class CiceroConfig(MultiAgentConfig):
    """Config for a Cicero (Diplomacy) multi-agent run."""

    num_steps: int = 3
    powers: list[str] | None = None  # None = use all seven
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.85

    def get_powers(self) -> list[str]:
        """Return the list of powers to use (default all seven)."""
        return self.powers if self.powers is not None else list(CICERO_POWERS)
