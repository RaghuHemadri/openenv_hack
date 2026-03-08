"""Codenames multi-agent plugin: 4-player word guessing game with Gemini-powered board generation."""

from watchdog_env.plugins.codenames.codenames_config import CodenamesConfig
from watchdog_env.plugins.codenames.codenames_plugin import CodenamesPlugin

__all__ = ["CodenamesPlugin", "CodenamesConfig"]
