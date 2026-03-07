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
