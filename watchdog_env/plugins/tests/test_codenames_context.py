"""Codenames plugin context tests. Require GEMINI_API_KEY or GOOGLE_API_KEY; use LLM (no fallback)."""

from __future__ import annotations

import os

import pytest

from watchdog_env.plugins.base import get_conversation_log
from watchdog_env.plugins.codenames import CodenamesConfig, CodenamesPlugin

pytestmark = pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY or GOOGLE_API_KEY required",
)


@pytest.fixture(autouse=True)
def use_llm(monkeypatch):
    monkeypatch.delenv("WATCHDOG_CODENAMES_USE_TEMPLATE", raising=False)


def test_codenames_reset_clears_context():
    """Reset clears conversation_log."""
    plugin = CodenamesPlugin()
    plugin.reset(seed=42, config=CodenamesConfig(complexity_level=1))
    log = get_conversation_log(plugin.get_state())
    print(f"\n[reset] conversation_log len={len(log)}")
    assert len(log) == 0


def test_codenames_step_state_has_conversation_log():
    """generate_step returns state with conversation_log field populated."""
    plugin = CodenamesPlugin()
    plugin.reset(seed=42, config=CodenamesConfig(complexity_level=1))
    step = plugin.generate_step(seed=42, step_index=0)
    assert step.state is not None
    assert hasattr(step.state, "conversation_log")
    assert isinstance(step.state.conversation_log, list)
    print(f"\n[step 0] conversation_log len={len(step.state.conversation_log)}, turns={len(step.turns)}")
    assert len(step.state.conversation_log) >= 1


def test_codenames_context_structure_after_steps():
    """After steps, plugin state has conversation_log populated."""
    plugin = CodenamesPlugin()
    plugin.reset(seed=1, config=CodenamesConfig(complexity_level=1))
    plugin.generate_step(seed=1, step_index=0)
    plugin.generate_step(seed=1, step_index=1)
    state = plugin.get_state()
    assert hasattr(state, "conversation_log")
    assert isinstance(state.conversation_log, list)
    print(f"\n[after 2 steps] conversation_log len={len(state.conversation_log)}")
    assert len(state.conversation_log) >= 2


def test_codenames_conversation_log_entry_structure():
    """Conversation log entries have speaker_id, speaker_display, message."""
    plugin = CodenamesPlugin()
    plugin.reset(seed=42, config=CodenamesConfig(complexity_level=1))
    plugin.generate_step(seed=42, step_index=0)
    log = get_conversation_log(plugin.get_state())
    
    assert len(log) >= 1
    entry = log[0]
    
    # Required fields
    assert "speaker_id" in entry
    assert "speaker_display" in entry
    assert "message" in entry
    
    # First entry should be from red_spymaster
    assert entry["speaker_id"] == "red_spymaster"
    assert "Red Spymaster" in entry["speaker_display"]
    assert "CLUE:" in entry["message"]
    
    print(f"\n[entry] speaker_id={entry['speaker_id']}, speaker_display={entry['speaker_display']}")
    print(f"        message={entry['message'][:60]}...")


def test_codenames_conversation_log_includes_game_metadata():
    """Conversation log entries include game-specific metadata like phase and team."""
    plugin = CodenamesPlugin()
    plugin.reset(seed=42, config=CodenamesConfig(complexity_level=1))
    plugin.generate_step(seed=42, step_index=0)
    log = get_conversation_log(plugin.get_state())
    
    assert len(log) >= 1
    entry = log[0]
    
    # Codenames-specific fields
    assert "phase" in entry
    assert entry["phase"] == "clue"
    assert "team" in entry
    assert entry["team"] == "red"
    
    print(f"\n[entry metadata] phase={entry['phase']}, team={entry['team']}")
