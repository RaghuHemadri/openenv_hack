"""Cicero multi-agent plugin: Diplomacy-style negotiation (LangChain + Gemini)."""

from watchdog_env.plugins.cicero.cicero_config import CICERO_POWERS, CiceroConfig
from watchdog_env.plugins.cicero.cicero_plugin import CiceroPlugin

__all__ = ["CiceroPlugin", "CiceroConfig", "CICERO_POWERS"]
