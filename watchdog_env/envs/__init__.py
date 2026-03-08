"""Plug-and-play environment games for WatchDog.

Each environment is a step-based game engine that generates player turns
via LangChain LLM calls. The WatchDog layer wraps it with mutations.

To add a new environment:
    1. Create a new file in this directory (e.g. `my_env.py`)
    2. Implement a game class with reset() and step() methods
    3. Register it in AVAILABLE_ENVS below
"""

from __future__ import annotations

from .avalon import AvalonGame

AVAILABLE_ENVS: dict[str, type] = {
    "avalon": AvalonGame,
}


def load_env(env_name: str, **kwargs):
    """Load an environment game by name.

    Returns an instantiated game object.
    Raises ValueError if the environment is not registered.
    """
    if env_name not in AVAILABLE_ENVS:
        available = ", ".join(sorted(AVAILABLE_ENVS.keys()))
        raise ValueError(
            f"Unknown environment '{env_name}'. Available: {available}"
        )
    return AVAILABLE_ENVS[env_name](**kwargs)


__all__ = ["AvalonGame", "AVAILABLE_ENVS", "load_env"]
