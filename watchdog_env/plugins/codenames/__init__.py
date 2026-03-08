"""Codenames multi-agent plugin: 4-player word guessing game.

Supports two backends (configured via WATCHDOG_LLM_BACKEND env var):
  - "local"  (default): shared Qwen3 8B game-play model from avalon/llm.py
  - "gemini": Google Gemini via langchain-google-genai (requires API key)
"""

from watchdog_env.plugins.codenames.codenames_config import CodenamesConfig
from watchdog_env.plugins.codenames.codenames_plugin import CodenamesPlugin

__all__ = ["CodenamesPlugin", "CodenamesConfig"]
