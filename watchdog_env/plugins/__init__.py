"""Multi-agent system plugins: base interface, registry, and built-in plugins."""

from watchdog_env.plugins.base import (
    AgentTurn,
    ContextMessage,
    MultiAgentConfig,
    MultiAgentState,
    MultiAgentStep,
    MultiAgentSystemPlugin,
    append_to_context,
    clear_system_context,
    get_system_context,
)
from watchdog_env.plugins.registry import get_plugin, get_registry, list_game_ids, register

# Auto-register Cicero so game_id="cicero" is available
try:
    from watchdog_env.plugins.cicero import CiceroPlugin
    register(CiceroPlugin())
except Exception:  # optional: Cicero may depend on langchain-google-genai
    CiceroPlugin = None  # type: ignore[misc, assignment]

__all__ = [
    "AgentTurn",
    "ContextMessage",
    "MultiAgentConfig",
    "MultiAgentState",
    "MultiAgentStep",
    "MultiAgentSystemPlugin",
    "append_to_context",
    "clear_system_context",
    "get_plugin",
    "get_system_context",
    "get_registry",
    "list_game_ids",
    "register",
    "CiceroPlugin",
]
