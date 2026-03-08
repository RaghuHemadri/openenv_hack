"""Cicero plugin context tests. Require GEMINI_API_KEY or GOOGLE_API_KEY; use LLM (no fallback)."""

from __future__ import annotations

import os

import pytest

from watchdog_env.plugins.base import get_conversation_log
from watchdog_env.plugins.cicero import CiceroConfig, CiceroPlugin

pytestmark = pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY or GOOGLE_API_KEY required",
)


@pytest.fixture(autouse=True)
def use_llm(monkeypatch):
    monkeypatch.delenv("WATCHDOG_CICERO_USE_TEMPLATE", raising=False)


def test_cicero_reset_clears_context():
    """Reset clears conversation_log."""
    plugin = CiceroPlugin()
    plugin.reset(seed=42, config=CiceroConfig(num_steps=5))
    log = get_conversation_log(plugin.get_state())
    print(f"\n[reset] conversation_log len={len(log)}")
    assert len(log) == 0


def test_cicero_step_state_has_conversation_log():
    """generate_step returns state with conversation_log field populated."""
    plugin = CiceroPlugin()
    plugin.reset(seed=42, config=CiceroConfig(num_steps=5))
    step = plugin.generate_step(seed=42, step_index=0)
    assert step.state is not None
    assert hasattr(step.state, "conversation_log")
    assert isinstance(step.state.conversation_log, list)
    print(f"\n[step 0] conversation_log len={len(step.state.conversation_log)}, turns={len(step.turns)}")
    assert len(step.state.conversation_log) >= 1


def test_cicero_context_structure_after_fallback_steps():
    """After steps, plugin state has conversation_log populated."""
    plugin = CiceroPlugin()
    plugin.reset(seed=1, config=CiceroConfig(num_steps=5))
    plugin.generate_step(seed=1, step_index=0)
    plugin.generate_step(seed=1, step_index=1)
    state = plugin.get_state()
    assert hasattr(state, "conversation_log")
    assert isinstance(state.conversation_log, list)
    print(f"\n[after 2 steps] conversation_log len={len(state.conversation_log)}")
    assert len(state.conversation_log) >= 2
