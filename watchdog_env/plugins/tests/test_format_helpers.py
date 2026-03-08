"""Tests for format helpers from shared models."""

from __future__ import annotations

import pytest

from watchdog_env.models import AgentTurn, format_conversation, format_current_turn


def test_format_conversation_empty():
    """Empty turns returns conversation start."""
    result = format_conversation([])
    assert "[Conversation start]" in result


def test_format_conversation_with_display_name():
    """Turns with display_name are formatted correctly."""
    turns = [
        AgentTurn(agent_id="P1", action_text="Hello", display_name="[P1] Alice"),
        AgentTurn(agent_id="P2", action_text="Hi there", display_name="[P2] Bob"),
    ]
    result = format_conversation(turns)
    assert "[Turn 1]" in result
    assert "[P1] Alice" in result
    assert "Hello" in result
    assert "[Turn 2]" in result
    assert "[P2] Bob" in result
    assert "Hi there" in result


def test_format_conversation_fallback_to_agent_id():
    """When display_name is empty, agent_id is used."""
    turns = [AgentTurn(agent_id="P1", action_text="Hello", display_name="")]
    result = format_conversation(turns)
    assert "P1" in result
    assert "Hello" in result


def test_format_current_turn_with_moderator():
    """Moderator prompt is included when present."""
    turn = AgentTurn(
        agent_id="P1",
        action_text="I think P2 is suspicious.",
        display_name="[P1] Alice",
        moderator_prompt="Day 1 begins. What do you say?",
    )
    result = format_current_turn(turn, moderator_prompt=turn.moderator_prompt)
    assert "[Moderator]" in result
    assert "Day 1 begins" in result
    assert "[P1] Alice" in result or "Alice" in result
    assert "I think P2 is suspicious" in result


def test_format_current_turn_without_moderator():
    """Without moderator, only agent and action_text."""
    turn = AgentTurn(
        agent_id="England",
        action_text="We should coordinate.",
        display_name="England",
    )
    result = format_current_turn(turn, moderator_prompt="")
    assert "[England]" in result
    assert "We should coordinate" in result
    assert "[Moderator]" not in result
