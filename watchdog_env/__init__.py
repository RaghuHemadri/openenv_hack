"""WatchDog Environment — Step-based multi-turn oversight framework."""

from .client import WatchDogMultiTurnEnv
from .models import MultiTurnAction, MultiTurnObservation, MultiTurnState
from .envs import load_env, AVAILABLE_ENVS
from .envs.avalon import AvalonGame

__all__ = [
    # Multi-turn oversight
    "MultiTurnAction",
    "MultiTurnObservation",
    "MultiTurnState",
    "WatchDogMultiTurnEnv",
    # Environments
    "AvalonGame",
    "load_env",
    "AVAILABLE_ENVS",
]
