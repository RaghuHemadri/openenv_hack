"""WatchDog Environment — Train the AI that watches the AI."""

from .client import WatchDogEnv, WatchDogMultiTurnEnv
from .models import WatchDogAction, WatchDogObservation, WatchDogState
from .models import MultiTurnAction, MultiTurnObservation, MultiTurnState

__all__ = [
    # Single-turn (legacy)
    "WatchDogAction",
    "WatchDogObservation",
    "WatchDogState",
    "WatchDogEnv",
    # Multi-turn (primary)
    "MultiTurnAction",
    "MultiTurnObservation",
    "MultiTurnState",
    "WatchDogMultiTurnEnv",
]
