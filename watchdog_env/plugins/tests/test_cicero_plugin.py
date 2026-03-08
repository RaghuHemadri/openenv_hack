"""Tests for Cicero plugin. Require GEMINI_API_KEY or GOOGLE_API_KEY to run; skipped if unset."""

from __future__ import annotations

import os

import pytest

from watchdog_env.plugins.cicero import CiceroConfig, CiceroPlugin
from watchdog_env.plugins.registry import get_plugin

# Skip all tests in this module if no Gemini API key
pytestmark = pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY or GOOGLE_API_KEY required for Cicero integration tests",
)


@pytest.fixture
def plugin():
    return CiceroPlugin()


def test_get_game_id(plugin):
    out = plugin.get_game_id()
    print(f"\n[get_game_id] {out}")
    assert out == "cicero"


def test_get_display_name(plugin):
    name = plugin.get_display_name()
    print(f"\n[get_display_name] {name}")
    assert isinstance(name, str) and len(name) > 0


def test_list_agent_ids(plugin):
    ids = plugin.list_agent_ids()
    print(f"\n[list_agent_ids] {ids}")
    assert isinstance(ids, list)
    seven = ["Austria-Hungary", "England", "France", "Germany", "Italy", "Russia", "Turkey"]
    for p in seven:
        assert p in ids


def test_reset_and_get_state(plugin):
    plugin.reset(seed=42, config=CiceroConfig(num_steps=2))
    state = plugin.get_state()
    print(f"\n[get_state after reset] step_index={state.step_index}, turns_so_far={len(state.turns_so_far)}, done={state.done}")
    assert state.step_index == 0
    assert len(state.turns_so_far) == 0
    assert state.done is False
    assert state.config is not None


def test_generate_step_based_on_state_history(plugin):
    """generate_step uses state history; reset then step 0, 1 and assert turns and done."""
    plugin.reset(seed=123, config=CiceroConfig(num_steps=2))
    step0 = plugin.generate_step(seed=123, step_index=0)
    print(f"\n[generate_step 0] step_index={step0.step_index}, done={step0.done}, turns={[(t.agent_id, t.action_text[:50]+'...' if len(t.action_text)>50 else t.action_text) for t in step0.turns]}")
    assert step0.step_index == 0
    assert len(step0.turns) >= 1
    for t in step0.turns:
        assert t.agent_id and t.action_text
        assert t.agent_id in plugin.list_agent_ids()
    assert hasattr(step0, "done")
    state_after_0 = plugin.get_state()
    assert state_after_0.step_index == 1
    assert len(state_after_0.turns_so_far) == len(step0.turns)

    step1 = plugin.generate_step(seed=123, step_index=1)
    print(f"\n[generate_step 1] step_index={step1.step_index}, done={step1.done}, turns={[(t.agent_id, t.action_text[:50]+'...' if len(t.action_text)>50 else t.action_text) for t in step1.turns]}")
    assert step1.step_index == 1
    assert len(step1.turns) >= 1
    assert step1.done is True
    state_after_1 = plugin.get_state()
    print(f"\n[get_state after step 1] step_index={state_after_1.step_index}, turns_so_far count={len(state_after_1.turns_so_far)}, done={state_after_1.done}")
    assert state_after_1.done is True


def test_cicero_registered():
    """Cicero plugin is registered so get_plugin('cicero') returns it."""
    p = get_plugin("cicero")
    print(f"\n[get_plugin('cicero')] {type(p).__name__} (game_id={p.get_game_id() if p else None})")
    assert p is not None
    assert p.get_game_id() == "cicero"
