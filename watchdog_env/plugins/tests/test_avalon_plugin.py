"""Tests for Avalon plugin. Require GEMINI_API_KEY or GOOGLE_API_KEY; use LLM (no fallback)."""

from __future__ import annotations

import os

import pytest

from watchdog_env.plugins.avalon import AvalonConfig, AvalonPlugin
from watchdog_env.plugins.registry import get_plugin

pytestmark = pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY or GOOGLE_API_KEY required",
)


@pytest.fixture(autouse=True)
def use_llm(monkeypatch):
    """Use LLM in tests (default when API key present). Ensure template not forced."""
    monkeypatch.delenv("WATCHDOG_AVALON_USE_TEMPLATE", raising=False)


@pytest.fixture
def plugin():
    return AvalonPlugin()


def test_get_game_id(plugin):
    out = plugin.get_game_id()
    print(f"\n[get_game_id] {out}")
    assert out == "avalon"


def test_get_display_name(plugin):
    name = plugin.get_display_name()
    print(f"\n[get_display_name] {name}")
    assert isinstance(name, str) and "Avalon" in name


def test_reset_and_generate_step(plugin):
    """Reset then generate steps until done."""
    plugin.reset(seed=42, config=AvalonConfig(level=1))
    state = plugin.get_state()
    print(f"\n[reset] step_index={state.step_index}, turns_so_far={len(state.turns_so_far)}, done={state.done}")
    assert state.step_index == 0
    assert len(state.turns_so_far) == 0
    assert state.done is False

    step_index = 0
    while not plugin.get_state().done and step_index < 30:
        step = plugin.generate_step(seed=42, step_index=step_index)
        for t in step.turns:
            msg = t.action_text[:60] + "..." if len(t.action_text) > 60 else t.action_text
            print(f"  [step {step_index}] {t.display_name or t.agent_id}: {msg}")
        assert step.step_index == step_index
        assert len(step.turns) >= 1
        for t in step.turns:
            assert t.agent_id and t.action_text
            assert t.display_name or t.agent_id
        step_index += 1

    print(f"\n[done] steps={step_index}, done={plugin.get_state().done}")
    assert plugin.get_state().done or step_index >= 30


def test_avalon_registered():
    """Avalon plugin is registered."""
    p = get_plugin("avalon")
    print(f"\n[get_plugin('avalon')] {type(p).__name__} (game_id={p.get_game_id() if p else None})")
    assert p is not None
    assert p.get_game_id() == "avalon"


def test_list_agent_ids_after_reset(plugin):
    """list_agent_ids returns player IDs after reset."""
    plugin.reset(seed=1, config=AvalonConfig(level=1))
    ids = plugin.list_agent_ids()
    print(f"\n[list_agent_ids] {ids}")
    assert isinstance(ids, list)
    assert len(ids) >= 5
    assert all(id.startswith("P") for id in ids)
