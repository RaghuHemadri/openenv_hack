"""Mutation Registry — Core abstractions for plug-and-play error injection.

Architecture:
    MutationScenario:  Declarative definition of a single error type
    MutationRegistry:  Central store. Holds env-specific mutations.
                       Mutations are registered directly via register_env().
"""

from __future__ import annotations

import random
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


class MutationRegistry:
    """Central registry holding environment-specific mutations.

    Usage:
        registry = MutationRegistry()
        registry.register_env("avalon", avalon_mutations)
        scenario = registry.sample(difficulty=2, env_name="avalon")
    """

    def __init__(self) -> None:
        self._generic: list[MutationScenario] = []
        self._env_mutations: dict[str, list[MutationScenario]] = {}

    # ── Registration ────────────────────────────────────────────

    def register_generic(self, scenario: MutationScenario) -> None:
        """Register a generic mutation usable by any environment."""
        scenario.env_specific = False
        scenario.env_name = None
        self._generic.append(scenario)

    def register_many_generic(self, scenarios: list[MutationScenario]) -> None:
        for s in scenarios:
            self.register_generic(s)

    def register_env(self, env_name: str, mutations: list[MutationScenario]) -> None:
        """Register environment-specific mutations directly."""
        for m in mutations:
            m.env_specific = True
            m.env_name = env_name
        self._env_mutations[env_name] = mutations

    # ── Querying ────────────────────────────────────────────────

    def list_categories(self) -> list[MutationCategory]:
        """All unique categories across all registered mutations."""
        cats = {s.category for s in self._generic}
        for mutations in self._env_mutations.values():
            cats |= {s.category for s in mutations}
        return sorted(cats, key=lambda c: c.value)

    def list_env_names(self) -> list[str]:
        return list(self._env_mutations.keys())

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
