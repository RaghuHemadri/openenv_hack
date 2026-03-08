"""WatchDog Environment — Client implementation for multi-turn oversight."""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import MultiTurnAction, MultiTurnObservation, MultiTurnState


class WatchDogMultiTurnEnv(
    EnvClient[MultiTurnAction, MultiTurnObservation, MultiTurnState]
):
    """Client for the WatchDog multi-turn oversight environment.

    Example:
        >>> with WatchDogMultiTurnEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.current_turn)
        ...     result = client.step(MultiTurnAction(action_type="pass"))
        ...     print(result.observation.feedback)
    """

    def _step_payload(self, action: MultiTurnAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[MultiTurnObservation]:
        obs_data = payload.get("observation", {})
        observation = MultiTurnObservation(
            conversation_so_far=obs_data.get("conversation_so_far", ""),
            current_turn=obs_data.get("current_turn", ""),
            current_turn_number=obs_data.get("current_turn_number", 0),
            total_turns=obs_data.get("total_turns", 0),
            task_domain=obs_data.get("task_domain", "general"),
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", 1),
            remaining_questions=obs_data.get("remaining_questions", 0),
            flags_so_far=obs_data.get("flags_so_far", 0),
            phase=obs_data.get("phase", "observe"),
            step_reward=obs_data.get("step_reward"),
            cumulative_reward=obs_data.get("cumulative_reward"),
            feedback=obs_data.get("feedback"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> MultiTurnState:
        return MultiTurnState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_level=payload.get("current_level", 1),
            total_episodes=payload.get("total_episodes", 0),
            errors_detected=payload.get("errors_detected", 0),
            errors_missed=payload.get("errors_missed", 0),
            false_flags=payload.get("false_flags", 0),
            correct_passes=payload.get("correct_passes", 0),
            questions_used=payload.get("questions_used", 0),
            interventions_correct=payload.get("interventions_correct", 0),
            interventions_wrong=payload.get("interventions_wrong", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
