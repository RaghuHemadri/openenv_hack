"""Cicero plugin context tests. Run without API key (fallback mode)."""

from __future__ import annotations

from watchdog_env.plugins.base import get_system_context
from watchdog_env.plugins.cicero import CiceroConfig, CiceroPlugin


def test_cicero_reset_clears_context():
    """Reset clears system_context."""
    plugin = CiceroPlugin()
    plugin.reset(seed=42, config=CiceroConfig(num_steps=2))
    assert len(get_system_context(plugin.get_state())) == 0


def test_cicero_step_state_has_system_context():
    """generate_step returns state with system_context field (empty in fallback mode)."""
    plugin = CiceroPlugin()
    plugin.reset(seed=42, config=CiceroConfig(num_steps=2))
    step = plugin.generate_step(seed=42, step_index=0)
    assert step.state is not None
    assert hasattr(step.state, "system_context")
    assert isinstance(step.state.system_context, list)
    # Fallback path does not append to context
    assert len(step.state.system_context) == 0


def test_cicero_context_structure_after_fallback_steps():
    """After fallback steps, plugin state has system_context (empty)."""
    plugin = CiceroPlugin()
    plugin.reset(seed=1, config=CiceroConfig(num_steps=2))
    plugin.generate_step(seed=1, step_index=0)
    plugin.generate_step(seed=1, step_index=1)
    state = plugin.get_state()
    assert hasattr(state, "system_context")
    assert isinstance(state.system_context, list)
    assert len(state.system_context) == 0  # fallback does not append
