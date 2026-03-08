#!/usr/bin/env python3
"""
Test Codenames Mutations
========================
Tests for Codenames-specific mutation scenarios in the error engine.

Run:
    cd watchdog_env && python -m pytest plugins/tests/test_codenames_mutations.py -v
    # or from repo root:
    PYTHONPATH=watchdog_env python -m pytest watchdog_env/plugins/tests/test_codenames_mutations.py -v
"""

import os
import pytest
import random

# Force template mode for testing
os.environ["WATCHDOG_USE_LLM"] = "0"

from watchdog_env.error_engine import (
    CODENAMES_MUTATIONS,
    CODENAMES_LEVEL_CONFIG,
    start_episode,
    maybe_mutate,
    _ensure_init,
)
from watchdog_env.mutations.registry import MutationCategory


class TestCodenamesMutationsConfig:
    """Test Codenames mutation configuration."""

    def test_codenames_mutations_exist(self):
        """Verify Codenames mutations are defined."""
        assert len(CODENAMES_MUTATIONS) > 0
        assert len(CODENAMES_MUTATIONS) == 12  # 3 easy + 4 moderate + 5 hard

    def test_level_config_exists(self):
        """Verify level config for Codenames."""
        assert len(CODENAMES_LEVEL_CONFIG) == 4
        for level in [1, 2, 3, 4]:
            assert level in CODENAMES_LEVEL_CONFIG
            config = CODENAMES_LEVEL_CONFIG[level]
            assert "max_difficulty" in config
            assert "clean_ratio" in config

    def test_difficulty_distribution(self):
        """Verify mutations are distributed across difficulty levels."""
        by_difficulty = {}
        for m in CODENAMES_MUTATIONS:
            d = m.difficulty
            by_difficulty[d] = by_difficulty.get(d, 0) + 1
        
        assert 1 in by_difficulty, "Should have difficulty 1 mutations"
        assert 2 in by_difficulty, "Should have difficulty 2 mutations"
        assert 3 in by_difficulty, "Should have difficulty 3 mutations"

    def test_mutation_categories(self):
        """Verify mutations cover different categories."""
        categories = set(m.category for m in CODENAMES_MUTATIONS)
        
        assert MutationCategory.FACTUAL_ERROR in categories
        assert MutationCategory.LOGIC_ERROR in categories
        assert MutationCategory.SAFETY_VIOLATION in categories
        assert MutationCategory.OVERCONFIDENCE in categories


class TestCodenamesMutationScenarios:
    """Test individual Codenames mutation scenarios."""

    def test_wrong_clue_number_mutation(self):
        """Test codenames_wrong_clue_number mutation."""
        mutations = [m for m in CODENAMES_MUTATIONS if m.name == "codenames_wrong_clue_number"]
        assert len(mutations) == 1
        m = mutations[0]
        assert m.difficulty == 1
        assert m.category == MutationCategory.FACTUAL_ERROR
        assert "clue number" in m.description.lower()

    def test_mention_assassin_word_mutation(self):
        """Test codenames_mention_assassin_word mutation."""
        mutations = [m for m in CODENAMES_MUTATIONS if m.name == "codenames_mention_assassin_word"]
        assert len(mutations) == 1
        m = mutations[0]
        assert m.difficulty == 1
        assert m.category == MutationCategory.SAFETY_VIOLATION

    def test_guess_already_revealed_mutation(self):
        """Test codenames_guess_already_revealed mutation."""
        mutations = [m for m in CODENAMES_MUTATIONS if m.name == "codenames_guess_already_revealed"]
        assert len(mutations) == 1
        m = mutations[0]
        assert m.difficulty == 1
        assert m.category == MutationCategory.FACTUAL_ERROR

    def test_misleading_clue_connection_mutation(self):
        """Test codenames_misleading_clue_connection mutation."""
        mutations = [m for m in CODENAMES_MUTATIONS if m.name == "codenames_misleading_clue_connection"]
        assert len(mutations) == 1
        m = mutations[0]
        assert m.difficulty == 2
        assert m.category == MutationCategory.LOGIC_ERROR

    def test_overconfident_guess_mutation(self):
        """Test codenames_overconfident_guess mutation."""
        mutations = [m for m in CODENAMES_MUTATIONS if m.name == "codenames_overconfident_guess"]
        assert len(mutations) == 1
        m = mutations[0]
        assert m.difficulty == 2
        assert m.category == MutationCategory.OVERCONFIDENCE

    def test_subtle_misdirection_mutation(self):
        """Test codenames_subtle_misdirection mutation."""
        mutations = [m for m in CODENAMES_MUTATIONS if m.name == "codenames_subtle_misdirection"]
        assert len(mutations) == 1
        m = mutations[0]
        assert m.difficulty == 3
        assert m.category == MutationCategory.LOGIC_ERROR

    def test_attribution_error_mutation(self):
        """Test codenames_attribution_error mutation."""
        mutations = [m for m in CODENAMES_MUTATIONS if m.name == "codenames_attribution_error"]
        assert len(mutations) == 1
        m = mutations[0]
        assert m.difficulty == 3
        assert m.category == MutationCategory.ATTRIBUTION_ERROR


class TestCodenamesMutationTracking:
    """Test mutation tracking for Codenames episodes."""

    def setup_method(self):
        """Reset mutation engine state before each test."""
        from watchdog_env import error_engine
        error_engine._registry = None
        error_engine._mutator = None
        _ensure_init()

    def test_start_episode_initializes_tracking(self):
        """Test that start_episode initializes Codenames tracking."""
        from watchdog_env import error_engine
        
        start_episode(game_id="codenames", num_turns=15)
        
        assert error_engine._game_id == "codenames"
        assert error_engine._episode_codenames_turns_remaining == 15
        assert error_engine._episode_has_mutation is False

    def test_mutation_decrements_turns(self):
        """Test that mutations decrement remaining turns."""
        from watchdog_env import error_engine
        
        start_episode(game_id="codenames", num_turns=10)
        initial_turns = error_engine._episode_codenames_turns_remaining
        
        maybe_mutate(
            clean_response="CLUE: ANIMAL 3",
            speaker_role="",
            level=2,
            context={"phase": "clue", "team": "Red"},
            game_id="codenames",
        )
        
        assert error_engine._episode_codenames_turns_remaining == initial_turns - 1

    def test_at_least_one_mutation_per_episode(self):
        """Test that at least one mutation occurs per episode."""
        random.seed(42)
        start_episode(game_id="codenames", num_turns=5)
        
        mutations_found = 0
        for i in range(5):
            _, has_error, _ = maybe_mutate(
                clean_response=f"Turn {i+1} response",
                speaker_role="",
                level=2,
                context={"phase": "guess", "team": "Blue", "step_index": i},
                game_id="codenames",
            )
            if has_error:
                mutations_found += 1
        
        assert mutations_found >= 1, "At least one mutation should occur per episode"


class TestCodenamesMutationContent:
    """Test that mutations produce valid content."""

    def setup_method(self):
        """Reset mutation engine state before each test."""
        from watchdog_env import error_engine
        error_engine._registry = None
        error_engine._mutator = None
        _ensure_init()

    def test_mutation_returns_string(self):
        """Test that mutation returns a string response."""
        start_episode(game_id="codenames", num_turns=3)
        
        # Force a mutation by running until we get one
        for _ in range(10):
            result, has_error, detail = maybe_mutate(
                clean_response="CLUE: SCIENCE 2 - This connects physics and chemistry",
                speaker_role="",
                level=1,
                context={"phase": "clue", "team": "Red"},
                game_id="codenames",
            )
            if has_error:
                assert isinstance(result, str)
                assert len(result) > 0
                break
            start_episode(game_id="codenames", num_turns=3)

    def test_mutation_detail_structure(self):
        """Test that mutation detail has expected structure."""
        start_episode(game_id="codenames", num_turns=3)
        
        for _ in range(10):
            result, has_error, detail = maybe_mutate(
                clean_response="GUESS: APPLE - I think this relates to the clue FRUIT",
                speaker_role="",
                level=2,
                context={"phase": "guess", "team": "Blue"},
                game_id="codenames",
            )
            if has_error:
                assert detail is not None
                assert "type" in detail
                assert "mutation_name" in detail
                assert "difficulty" in detail
                assert detail["mutation_name"].startswith("codenames_")
                break
            start_episode(game_id="codenames", num_turns=3)

    def test_mutation_respects_difficulty(self):
        """Test that mutations respect difficulty limits."""
        random.seed(123)
        
        # At level 1, only difficulty 1 mutations should be allowed
        start_episode(game_id="codenames", num_turns=20)
        
        level_1_mutations = []
        for _ in range(20):
            _, has_error, detail = maybe_mutate(
                clean_response=f"CLUE: TEST {random.randint(1,5)}",
                speaker_role="",
                level=1,
                context={"phase": "clue", "team": "Red"},
                game_id="codenames",
            )
            if has_error and detail:
                level_1_mutations.append(detail.get("difficulty", 0))
        
        if level_1_mutations:
            max_diff = CODENAMES_LEVEL_CONFIG[1]["max_difficulty"]
            for diff in level_1_mutations:
                assert diff <= max_diff, f"Level 1 should only have difficulty <= {max_diff}"


class TestCodenamesMutationRegistry:
    """Test that Codenames mutations are properly registered."""

    def test_codenames_registered_in_registry(self):
        """Test that Codenames mutations are in the registry."""
        from watchdog_env import error_engine
        error_engine._registry = None
        error_engine._mutator = None
        
        registry, _ = _ensure_init()
        
        # Access the internal _env_mutations dict to check registration
        codenames_mutations = registry._env_mutations.get("codenames", [])
        assert len(codenames_mutations) == len(CODENAMES_MUTATIONS)

    def test_all_mutation_names_unique(self):
        """Test that all Codenames mutation names are unique."""
        names = [m.name for m in CODENAMES_MUTATIONS]
        assert len(names) == len(set(names)), "Mutation names should be unique"

    def test_all_mutations_have_llm_instruction(self):
        """Test that all mutations have LLM instructions."""
        for m in CODENAMES_MUTATIONS:
            assert m.llm_instruction, f"Mutation {m.name} should have llm_instruction"
            assert len(m.llm_instruction) > 10, f"Mutation {m.name} instruction too short"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
