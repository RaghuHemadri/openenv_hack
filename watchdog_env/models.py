"""WatchDog Environment — Shared models for multi-turn oversight and plugins.

Shared types (used by plugins and env): AgentTurn, MultiAgentStep, MultiAgentState,
MultiAgentConfig, ContextMessage. Env-specific types extend OpenEnv Action/Observation/State.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from pydantic import Field

if TYPE_CHECKING:
    pass

# ─── Shared Types (plugins + env) ────────────────────────────────────

ContextRole = Literal["system", "user", "assistant"]


@dataclass
class ContextMessage:
    """A single message in the system context (LLM conversation history)."""

    role: ContextRole
    content: str


@dataclass
class AgentTurn:
    """Canonical turn representation. Plugins and env both use this."""

    agent_id: str
    action_text: str
    step_index: int = 0
    phase: str = ""
    display_name: str = ""
    moderator_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAgentConfig:
    """Base config for a multi-agent system run. Plugins subclass for game-specific fields."""

    pass


ConversationLogEntry = dict[str, Any]
"""Plain conversation log entry: speaker_id, speaker_display, message, optionally moderator_prompt."""


@dataclass
class MultiAgentState:
    """Tracks system behaviour across the run. Used when generating each MultiAgentStep."""

    step_index: int = 0
    turns_so_far: list[AgentTurn] = field(default_factory=list)
    config: MultiAgentConfig | None = None
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    conversation_log: list[ConversationLogEntry] = field(default_factory=list)


@dataclass
class MultiAgentStep:
    """One step: multiple agent turns. done=True means scenario is finished."""

    turns: list[AgentTurn]
    done: bool = False
    step_index: int = 0
    game_id: str = ""
    task_id: str = ""
    domain: str = ""
    state: MultiAgentState | None = None


# ─── Format Helpers ──────────────────────────────────────────────────

def format_conversation(turns: list[AgentTurn]) -> str:
    """Format turns for conversation_so_far. Uses display_name or agent_id."""
    if not turns:
        return "[Conversation start]"
    lines = []
    for i, t in enumerate(turns):
        label = t.display_name or t.agent_id
        lines.append(f"[Turn {i + 1}] {label}: {t.action_text}")
    return "\n".join(lines)


def format_current_turn(turn: AgentTurn, moderator_prompt: str = "") -> str:
    """Build current_turn string. Includes moderator prompt if present."""
    label = turn.display_name or turn.agent_id
    prompt = moderator_prompt or turn.moderator_prompt
    if prompt:
        return f"[Moderator]: {prompt}\n\n[{label}]: {turn.action_text}"
    return f"[{label}]: {turn.action_text}"


def agent_turn_to_dict(
    turn: AgentTurn,
    has_error: bool = False,
    displayed_response: str | None = None,
    error_detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert AgentTurn to dict for _turns_seen compatibility."""
    out: dict[str, Any] = {
        "speaker_id": turn.agent_id,
        "speaker_display": turn.display_name or turn.agent_id,
        "message": turn.action_text,
        "displayed_response": displayed_response if displayed_response is not None else turn.action_text,
        "has_error": has_error,
        "moderator_prompt": turn.moderator_prompt or "",
        "phase": turn.phase,
        **turn.metadata,
    }
    if error_detail is not None:
        out["error_detail"] = error_detail
    return out


# ─── Env-Specific (extend OpenEnv types) ──────────────────────────────

from openenv.core.env_server.types import Action, Observation, State
from typing import Optional


class MultiTurnAction(Action):
    """Overseer action in a multi-turn oversight episode. Supports any multi-agent system."""

    action_type: str = Field(
        ...,
        description='One of: "pass", "flag", "question", "intervene"',
    )
    error_type: Optional[str] = Field(
        default=None,
        description='For flag: "factual_error", "logic_error", "code_bug", "safety_violation", "sycophancy"',
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Explanation for flag or intervene actions",
    )
    question_text: Optional[str] = Field(
        default=None,
        description="Question to ask the Worker AI (for question action)",
    )


class MultiTurnObservation(Observation):
    """Observation at each step of a multi-agent oversight episode. Plugin-agnostic."""

    conversation_so_far: str = Field(
        default="", description="All turns revealed so far"
    )
    current_turn: str = Field(
        default="", description="The latest turn to evaluate"
    )
    current_turn_number: int = Field(
        default=0, description="Current worker turn number (1-indexed)"
    )
    total_turns: int = Field(
        default=0, description="Total worker turns in this episode"
    )
    task_domain: str = Field(default="general", description="Conversation domain / game_id")
    task_id: str = Field(default="", description="Episode ID")
    difficulty: int = Field(default=1, description="Curriculum difficulty 1-4")
    remaining_questions: int = Field(
        default=2, description="QUESTION actions remaining (investigation budget)"
    )
    flags_so_far: int = Field(
        default=0, description="Number of FLAGS issued this episode"
    )
    phase: str = Field(
        default="observe",
        description='"observe" (new worker turn), "question_response" (after QUESTION), "done" (episode over)',
    )
    step_reward: Optional[float] = Field(
        default=None, description="Reward from last action"
    )
    cumulative_reward: Optional[float] = Field(
        default=None, description="Total reward this episode"
    )
    feedback: Optional[str] = Field(
        default=None, description="Feedback from last action"
    )


class MultiTurnState(State):
    """Episode state for multi-agent oversight. Works with any plugin."""

    episode_id: str = Field(default="", description="Episode identifier")
    step_count: int = Field(default=0, description="Steps in current episode")
    current_level: int = Field(default=1, description="Current curriculum level")
    total_episodes: int = Field(default=0, description="Total episodes completed")
    errors_detected: int = 0
    errors_missed: int = 0
    false_flags: int = 0
    correct_passes: int = 0
    questions_used: int = 0
    interventions_correct: int = 0
    interventions_wrong: int = 0
    cumulative_reward: float = 0.0
