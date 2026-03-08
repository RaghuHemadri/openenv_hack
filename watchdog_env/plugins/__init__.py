"""Multi-agent system plugins: base interface, registry, and built-in plugins."""

from watchdog_env.plugins.base import (
    AgentTurn,
    ConversationLogEntry,
    MultiAgentConfig,
    MultiAgentState,
    MultiAgentStep,
    MultiAgentSystemPlugin,
    append_to_conversation_log,
    clear_conversation_log,
    get_conversation_log,
)
from watchdog_env.plugins.registry import get_plugin, get_registry, list_game_ids, register

# Auto-register Cicero so game_id="cicero" is available
try:
    from watchdog_env.plugins.cicero import CiceroPlugin
    register(CiceroPlugin())
except Exception:  # optional: Cicero may depend on langchain-google-genai
    CiceroPlugin = None  # type: ignore[misc, assignment]

# Auto-register Codenames so game_id="codenames" is available
try:
    from watchdog_env.plugins.codenames import CodenamesPlugin
    register(CodenamesPlugin())
except Exception:  # optional: Codenames may depend on langchain-google-genai
    CodenamesPlugin = None  # type: ignore[misc, assignment]

__all__ = [
    "AgentTurn",
    "ConversationLogEntry",
    "MultiAgentConfig",
    "MultiAgentState",
    "MultiAgentStep",
    "MultiAgentSystemPlugin",
    "append_to_conversation_log",
    "clear_conversation_log",
    "get_plugin",
    "get_conversation_log",
    "get_registry",
    "list_game_ids",
    "register",
    "CiceroPlugin",
    "CodenamesPlugin",
]
