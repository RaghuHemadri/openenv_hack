"""Avalon (Werewolf) multi-agent plugin."""

from watchdog_env.plugins.avalon.avalon_config import AvalonConfig, LEVEL_CONFIG
from watchdog_env.plugins.avalon.avalon_game import AvalonGame
from watchdog_env.plugins.avalon.avalon_models import (
    GameState,
    Player,
    create_game,
    _DAY_EVENTS,
    _DAY_EVENTS_NO_DEATH,
    _DAY_OPENERS,
)
from watchdog_env.plugins.avalon.avalon_plugin import AvalonPlugin
from watchdog_env.plugins.avalon.llm import (
    _HFChatModel,
    _HFChatResponse,
    _generate_player_response_llm,
    _get_llm,
    _get_local_hf_llm,
    _llm,
)

__all__ = [
    "AvalonPlugin",
    "AvalonConfig",
    "AvalonGame",
    "LEVEL_CONFIG",
    "GameState",
    "Player",
    "create_game",
    "_generate_player_response_llm",
    "_get_local_hf_llm",
    "_get_llm",
    "_llm",
    "_HFChatModel",
    "_HFChatResponse",
]
