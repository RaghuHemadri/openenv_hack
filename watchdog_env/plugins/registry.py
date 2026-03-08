"""Registry for multi-agent system plugins. Look up by game_id."""

from __future__ import annotations

from .base import MultiAgentSystemPlugin

_registry: dict[str, MultiAgentSystemPlugin] = {}


def register(plugin: MultiAgentSystemPlugin) -> None:
    """Register a plugin by its game_id."""
    _registry[plugin.get_game_id()] = plugin


def get_plugin(game_id: str) -> MultiAgentSystemPlugin | None:
    """Return the plugin for this game_id, or None."""
    return _registry.get(game_id)


def list_game_ids() -> list[str]:
    """Return all registered game_ids."""
    return list(_registry.keys())


def get_registry() -> dict[str, MultiAgentSystemPlugin]:
    """Return the underlying registry dict (for tests or advanced use)."""
    return _registry
