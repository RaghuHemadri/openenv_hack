"""Mutation Registry — Plug-and-play error injection for any multi-agent environment."""

from .registry import (
    MutationScenario,
    MutationCategory,
    EnvironmentPlugin,
    MutationRegistry,
)
from .llm_backend import LLMMutator

__all__ = [
    "MutationScenario",
    "MutationCategory",
    "EnvironmentPlugin",
    "MutationRegistry",
    "LLMMutator",
]
