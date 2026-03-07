"""Mutation Registry — Core abstractions for plug-and-play error injection.

Architecture:
    MutationScenario:  Declarative definition of a single error type
    EnvironmentPlugin: Interface for envs to register domain-specific mutations
    MutationRegistry:  Central store. Holds generic + env-specific mutations.
                       Any environment plugs in by registering a plugin.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MutationCategory(str, Enum):
    """Broad mutation categories that generalize across environments."""

    FACTUAL_ERROR = "factual_error"
    LOGIC_ERROR = "logic_error"
    CODE_BUG = "code_bug"
    SAFETY_VIOLATION = "safety_violation"
    SYCOPHANCY = "sycophancy"
    HALLUCINATION = "hallucination"
    INCONSISTENCY = "inconsistency"
    OVERCONFIDENCE = "overconfidence"
    OMISSION = "omission"
    TONE_SHIFT = "tone_shift"
    TEMPORAL_ERROR = "temporal_error"
    NUMERIC_ERROR = "numeric_error"
    ATTRIBUTION_ERROR = "attribution_error"
    CONTEXT_DRIFT = "context_drift"
    INSTRUCTION_VIOLATION = "instruction_violation"


@dataclass
class MutationScenario:
    """A single mutation type that the LLM (or fallback) can apply.

    Attributes:
        name:              Short unique identifier (e.g. "swap_capital_city")
        category:          One of MutationCategory values
        description:       Human-readable description of what this mutation does
        llm_instruction:   Prompt snippet telling the LLM how to apply this mutation.
                           The LLM receives: clean_response + context + this instruction.
        difficulty:        1 (easy to spot) → 3 (subtle)
        env_specific:      If True, only applies to the registered environment
        env_name:          Which environment this belongs to (None = generic)
        fallback_examples: Static examples for when LLM is unavailable.
                           Each dict has at minimum {"clean": ..., "mutated": ..., "explanation": ...}
        metadata:          Arbitrary extra info for reward computation, etc.
    """

    name: str
    category: MutationCategory
    description: str
    llm_instruction: str
    difficulty: int = 1
    env_specific: bool = False
    env_name: str | None = None
    fallback_examples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EnvironmentPlugin(ABC):
    """Interface for any multi-agent environment to plug into the mutation system.

    To add a new environment:
        1. Subclass EnvironmentPlugin
        2. Implement the 4 abstract methods
        3. Call registry.register_plugin(YourPlugin())
    """

    @abstractmethod
    def get_env_name(self) -> str:
        """Unique name for this environment (e.g. 'watchdog', 'diplomacy')."""
        ...

    @abstractmethod
    def get_mutations(self) -> list[MutationScenario]:
        """Return environment-specific mutation scenarios."""
        ...

    @abstractmethod
    def get_conversation_frames(self) -> list[dict[str, Any]]:
        """Return conversation templates / frames for episode generation.

        Each dict should have at minimum:
            {"domain": str, "user_prompt": str, ...extra context fields}
        """
        ...

    @abstractmethod
    def get_clean_responses(self) -> dict[str, list[str]]:
        """Return clean worker responses keyed by domain.

        Used when generating non-error turns.
        """
        ...

    def get_followup_messages(self) -> dict[str, list[str]]:
        """Return follow-up user messages keyed by domain. Optional."""
        return {}

    def get_level_config(self) -> dict[int, dict[str, Any]]:
        """Return curriculum level config. Optional — defaults provided.

        Expected format per level:
            {
                "clean_ratio": float,          # probability of clean episode
                "categories": list[str],       # allowed mutation categories
                "max_difficulty": int,          # 1-3
                "num_rounds": int,              # turns per episode
            }
        """
        return {
            1: {
                "clean_ratio": 0.50,
                "categories": [MutationCategory.FACTUAL_ERROR],
                "max_difficulty": 1,
                "num_rounds": 3,
            },
            2: {
                "clean_ratio": 0.40,
                "categories": [
                    MutationCategory.FACTUAL_ERROR,
                    MutationCategory.LOGIC_ERROR,
                    MutationCategory.CODE_BUG,
                ],
                "max_difficulty": 2,
                "num_rounds": 3,
            },
            3: {
                "clean_ratio": 0.30,
                "categories": [
                    MutationCategory.FACTUAL_ERROR,
                    MutationCategory.LOGIC_ERROR,
                    MutationCategory.CODE_BUG,
                    MutationCategory.SAFETY_VIOLATION,
                    MutationCategory.SYCOPHANCY,
                ],
                "max_difficulty": 3,
                "num_rounds": 4,
            },
            4: {
                "clean_ratio": 0.35,
                "categories": [c for c in MutationCategory],
                "max_difficulty": 3,
                "num_rounds": 4,
            },
        }


class MutationRegistry:
    """Central registry holding generic + environment-specific mutations.

    Usage:
        registry = MutationRegistry()
        register_generic_mutations(registry)   # load built-in generics
        registry.register_plugin(WatchDogPlugin())  # plug in your env

        scenario = registry.sample(difficulty=2, env_name="watchdog")
    """

    def __init__(self) -> None:
        # Generic mutations available to ALL environments
        self._generic: list[MutationScenario] = []
        # Environment-specific mutations keyed by env_name
        self._env_mutations: dict[str, list[MutationScenario]] = {}
        # Registered plugins
        self._plugins: dict[str, EnvironmentPlugin] = {}

    # ── Registration ────────────────────────────────────────────

    def register_generic(self, scenario: MutationScenario) -> None:
        """Register a generic mutation usable by any environment."""
        scenario.env_specific = False
        scenario.env_name = None
        self._generic.append(scenario)

    def register_many_generic(self, scenarios: list[MutationScenario]) -> None:
        for s in scenarios:
            self.register_generic(s)

    def register_plugin(self, plugin: EnvironmentPlugin) -> None:
        """Register an environment plugin and its mutations."""
        name = plugin.get_env_name()
        self._plugins[name] = plugin
        mutations = plugin.get_mutations()
        for m in mutations:
            m.env_specific = True
            m.env_name = name
        self._env_mutations[name] = mutations

    # ── Querying ────────────────────────────────────────────────

    def get_plugin(self, env_name: str) -> EnvironmentPlugin | None:
        return self._plugins.get(env_name)

    def list_categories(self) -> list[MutationCategory]:
        """All unique categories across all registered mutations."""
        cats = {s.category for s in self._generic}
        for mutations in self._env_mutations.values():
            cats |= {s.category for s in mutations}
        return sorted(cats, key=lambda c: c.value)

    def list_env_names(self) -> list[str]:
        return list(self._plugins.keys())

    def count(self, env_name: str | None = None) -> int:
        """Count registered mutations. None = generic only."""
        if env_name is None:
            return len(self._generic)
        return len(self._generic) + len(self._env_mutations.get(env_name, []))

    # ── Sampling ────────────────────────────────────────────────

    def sample(
        self,
        difficulty: int | None = None,
        category: MutationCategory | str | None = None,
        env_name: str | None = None,
        include_generic: bool = True,
    ) -> MutationScenario:
        """Sample a random mutation matching the filters.

        Args:
            difficulty: Filter by difficulty level (1-3). None = any.
            category:   Filter by category. None = any.
            env_name:   Include env-specific mutations for this environment.
            include_generic: Whether to also consider generic mutations.

        Returns:
            A randomly chosen MutationScenario matching the filters.

        Raises:
            ValueError if no mutations match the filters.
        """
        pool = self._build_pool(difficulty, category, env_name, include_generic)
        if not pool:
            raise ValueError(
                f"No mutations match filters: difficulty={difficulty}, "
                f"category={category}, env_name={env_name}"
            )
        return random.choice(pool)

    def sample_n(
        self,
        n: int,
        difficulty: int | None = None,
        category: MutationCategory | str | None = None,
        env_name: str | None = None,
        include_generic: bool = True,
        allow_duplicates: bool = False,
    ) -> list[MutationScenario]:
        """Sample n mutations. Without replacement by default."""
        pool = self._build_pool(difficulty, category, env_name, include_generic)
        if not pool:
            raise ValueError("No mutations match the given filters")
        if allow_duplicates:
            return [random.choice(pool) for _ in range(n)]
        return random.sample(pool, min(n, len(pool)))

    def get_all(
        self,
        difficulty: int | None = None,
        category: MutationCategory | str | None = None,
        env_name: str | None = None,
        include_generic: bool = True,
    ) -> list[MutationScenario]:
        """Return all mutations matching filters."""
        return self._build_pool(difficulty, category, env_name, include_generic)

    def _build_pool(
        self,
        difficulty: int | None,
        category: MutationCategory | str | None,
        env_name: str | None,
        include_generic: bool,
    ) -> list[MutationScenario]:
        pool: list[MutationScenario] = []

        if include_generic:
            pool.extend(self._generic)

        if env_name and env_name in self._env_mutations:
            pool.extend(self._env_mutations[env_name])

        # Filter by difficulty
        if difficulty is not None:
            pool = [s for s in pool if s.difficulty <= difficulty]

        # Filter by category
        if category is not None:
            if isinstance(category, str):
                pool = [s for s in pool if s.category.value == category]
            else:
                pool = [s for s in pool if s.category == category]

        return pool
