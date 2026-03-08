"""Avalon (Werewolf) multi-agent plugin."""

from watchdog_env.envs.avalon import LEVEL_CONFIG
from watchdog_env.plugins.avalon.avalon_config import AvalonConfig
from watchdog_env.plugins.avalon.avalon_plugin import AvalonPlugin

__all__ = ["AvalonPlugin", "AvalonConfig", "LEVEL_CONFIG"]
