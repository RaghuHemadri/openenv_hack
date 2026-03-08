"""Codenames multi-agent plugin: 4-player word guessing game.

Uses shared local Qwen3 8B game-play model from avalon/llm.py.
"""

from watchdog_env.plugins.codenames.codenames_config import CodenamesConfig
from watchdog_env.plugins.codenames.codenames_plugin import CodenamesPlugin

__all__ = ["CodenamesPlugin", "CodenamesConfig"]
