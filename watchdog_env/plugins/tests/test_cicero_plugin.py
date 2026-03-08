"""Tests for Cicero plugin. Require GEMINI_API_KEY or GOOGLE_API_KEY; use LLM (no fallback)."""

from __future__ import annotations

import os

import pytest

from watchdog_env.plugins.base import get_conversation_log
from watchdog_env.plugins.cicero import CiceroConfig, CiceroPlugin
from watchdog_env.plugins.registry import get_plugin

pytestmark = pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY or GOOGLE_API_KEY required",
)


@pytest.fixture(autouse=True)
def use_llm(monkeypatch):
    """Use LLM in tests (default when API key present). Ensure template not forced."""
    monkeypatch.delenv("WATCHDOG_CICERO_USE_TEMPLATE", raising=False)


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
    plugin.reset(seed=42, config=CiceroConfig(num_steps=5))
    state = plugin.get_state()
    print(f"\n[get_state after reset] step_index={state.step_index}, turns_so_far={len(state.turns_so_far)}, done={state.done}")
    assert state.step_index == 0
    assert len(state.turns_so_far) == 0
    assert state.done is False
    assert state.config is not None
    assert len(state.conversation_log) == 0  # conversation_log cleared on reset


def test_generate_step_based_on_state_history(plugin):
    """generate_step uses state history; reset then run steps until done."""
    plugin.reset(seed=123, config=CiceroConfig(num_steps=5))
    step_index = 0
    while not plugin.get_state().done and step_index < 10:
        step = plugin.generate_step(seed=123, step_index=step_index)
        for t in step.turns:
            msg = t.action_text[:50] + "..." if len(t.action_text) > 50 else t.action_text
            print(f"  [step {step_index}] {t.agent_id}: {msg}")
        assert step.step_index == step_index
        assert len(step.turns) >= 1
        for t in step.turns:
            assert t.agent_id and t.action_text
            assert t.agent_id in plugin.list_agent_ids()
        assert step.state is not None and hasattr(step.state, "conversation_log")
        step_index += 1

    print(f"\n[done] steps={step_index}, done={plugin.get_state().done}")
    assert plugin.get_state().done or step_index >= 10


def test_cicero_registered():
    """Cicero plugin is registered so get_plugin('cicero') returns it."""
    p = get_plugin("cicero")
    print(f"\n[get_plugin('cicero')] {type(p).__name__} (game_id={p.get_game_id() if p else None})")
    assert p is not None
    assert p.get_game_id() == "cicero"


def test_cicero_context_in_step_state(plugin):
    """Each step returns state with conversation_log; entries have speaker_id, message."""
    plugin.reset(seed=1, config=CiceroConfig(num_steps=5))
    step = plugin.generate_step(seed=1, step_index=0)
    assert step.state is not None
    assert isinstance(step.state.conversation_log, list)
    log = step.state.conversation_log
    for entry in log:
        assert "speaker_id" in entry and "message" in entry
        assert isinstance(entry["message"], str) and len(entry["message"]) > 0


def test_cicero_turns_have_rich_metadata(plugin):
    """Each AgentTurn has rich metadata (season, region, domain_name, domain_desc, counterpart)."""
    plugin.reset(seed=1, config=CiceroConfig(num_steps=5))
    step = plugin.generate_step(seed=1, step_index=0)
    for t in step.turns:
        assert "season" in t.metadata
        assert "region" in t.metadata
        assert "domain_name" in t.metadata
        assert "domain_desc" in t.metadata
        assert "counterpart" in t.metadata
        assert t.metadata["counterpart"] in plugin.list_agent_ids()


def test_cicero_context_cleared_on_reset(plugin):
    """Reset clears conversation_log (empty after reset regardless of LLM vs fallback)."""
    plugin.reset(seed=1, config=CiceroConfig(num_steps=5))
    plugin.generate_step(seed=1, step_index=0)
    plugin.reset(seed=99, config=CiceroConfig(num_steps=2))
    assert len(get_conversation_log(plugin.get_state())) == 0
