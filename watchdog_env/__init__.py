"""WatchDog Environment — Step-based multi-turn oversight framework."""

from .client import WatchDogMultiTurnEnv
from .models import MultiTurnAction, MultiTurnObservation, MultiTurnState
from .plugins.avalon import AvalonGame
from .plugins.registry import get_plugin, list_game_ids

__all__ = [
    # Multi-turn oversight
    "MultiTurnAction",
    "MultiTurnObservation",
    "MultiTurnState",
    "WatchDogMultiTurnEnv",
    # Environments
    "AvalonGame",
    "get_plugin",
    "list_game_ids",
]
