"""WatchDog Environment — Client implementation."""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import WatchDogAction, WatchDogObservation, WatchDogState


class WatchDogEnv(EnvClient[WatchDogAction, WatchDogObservation, WatchDogState]):
    """
    Client for the WatchDog oversight environment.

    Example:
        >>> with WatchDogEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.conversation)
        ...
        ...     result = client.step(WatchDogAction(verdict="clean"))
        ...     print(result.observation.feedback)
    """

    def _step_payload(self, action: WatchDogAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[WatchDogObservation]:
        obs_data = payload.get("observation", {})
        observation = WatchDogObservation(
            conversation=obs_data.get("conversation", ""),
            task_domain=obs_data.get("task_domain", "general"),
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", 1),
            feedback=obs_data.get("feedback"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> WatchDogState:
        return WatchDogState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_level=payload.get("current_level", 1),
            total_episodes=payload.get("total_episodes", 0),
            true_positives=payload.get("true_positives", 0),
            false_positives=payload.get("false_positives", 0),
            true_negatives=payload.get("true_negatives", 0),
            false_negatives=payload.get("false_negatives", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
