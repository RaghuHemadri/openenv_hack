"""WatchDog Environment — Pydantic Models."""

from typing import Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class WatchDogAction(Action):
    """The oversight agent's verdict on a conversation."""

    verdict: str = Field(
        ...,
        description='One of: "clean", "factual_error", "logic_error", "code_bug", "safety_violation", "sycophancy"',
    )
    location: Optional[str] = Field(
        default=None, description='e.g. "assistant_turn_2" or "line_15"'
    )
    explanation: Optional[str] = Field(
        default=None, description="Free-text explanation of why it's an error"
    )


class WatchDogObservation(Observation):
    """What the oversight agent sees."""

    conversation: str = Field(default="", description="The User-Worker conversation to review")
    task_domain: str = Field(default="general", description="Domain of the conversation")
    task_id: str = Field(default="", description="Unique ID for this conversation")
    difficulty: int = Field(default=1, description="Curriculum level 1-4")
    feedback: Optional[str] = Field(
        default=None, description="After step: was the verdict correct?"
    )


class WatchDogState(State):
    """Episode state tracking."""

    current_level: int = Field(default=1, description="Current curriculum level")
    total_episodes: int = Field(default=0, description="Total episodes played")
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    cumulative_reward: float = 0.0


# ─── Multi-Turn Models (Avalon/Werewolf-style oversight) ───────────


class MultiTurnAction(Action):
    """Overseer's action in a multi-turn oversight episode.

    Like Avalon/Werewolf: the overseer investigates worker turns (villagers)
    to find hidden errors (werewolves) using limited investigation budget.
    """

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
    """What the overseer sees at each step of a multi-turn episode.

    Turns are revealed incrementally (like day/night phases in Werewolf).
    The overseer must deduce which worker turns contain hidden errors.
    """

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
    task_domain: str = Field(default="general", description="Conversation domain")
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
    """Multi-turn episode state tracking."""

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
