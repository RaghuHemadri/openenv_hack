"""Tests for Cicero mutations in the error engine."""

from __future__ import annotations

import pytest

try:
    from watchdog_env.error_engine import (
        CICERO_MUTATIONS,
        CICERO_LEVEL_CONFIG,
        maybe_mutate,
        start_episode,
        _ensure_init,
    )
except ImportError as e:
    if "openenv" in str(e):
        pytest.skip("openenv not installed", allow_module_level=True)
    raise


def test_cicero_mutations_registered():
    """Cicero mutations are registered in the error engine."""
    registry, _ = _ensure_init()
    assert "cicero" in registry.list_env_names()
    cicero_mutations = registry.get_all(env_name="cicero", include_generic=False)
    assert len(cicero_mutations) == len(CICERO_MUTATIONS)
    assert len(CICERO_MUTATIONS) >= 8


def test_cicero_level_config():
    """CICERO_LEVEL_CONFIG has expected structure."""
    for level in (1, 2, 3, 4):
        assert level in CICERO_LEVEL_CONFIG
        cfg = CICERO_LEVEL_CONFIG[level]
        assert "max_difficulty" in cfg
        assert "clean_ratio" in cfg
        assert 1 <= cfg["max_difficulty"] <= 3
        assert 0 < cfg["clean_ratio"] < 1


def test_maybe_mutate_cicero():
    """maybe_mutate with game_id=cicero can mutate when appropriate."""
    start_episode(game_id="cicero", num_steps=2)
    out, has_err, detail = maybe_mutate(
        clean_response="I propose we support each other in Vienna this season.",
        speaker_role="",
        level=1,
        context={
            "speaker_id": "Austria-Hungary",
            "season": "Spring 1901",
            "region": "Vienna",
        },
        game_id="cicero",
    )
    # With force-mutate (last turn) or random, we may get mutation
    assert isinstance(out, str)
    assert isinstance(has_err, bool)
    if has_err:
        assert detail is not None
        assert "mutation_name" in detail
        assert "description" in detail
