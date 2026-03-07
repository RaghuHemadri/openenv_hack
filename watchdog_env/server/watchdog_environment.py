"""WatchDog Environment — Server-side implementation."""

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import WatchDogAction, WatchDogObservation, WatchDogState
from error_engine import sample_episode
from rewards import compute_reward


class WatchDogEnvironment(Environment[WatchDogAction, WatchDogObservation, WatchDogState]):
    """RL environment for training AI oversight agents.

    The agent reviews conversations between a User and Worker AI,
    and must detect errors (factual, logic, code, safety, sycophancy).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = WatchDogState(episode_id=str(uuid.uuid4()), step_count=0)
        self._current_manifest: Optional[dict] = None
        self._current_conversation: Optional[str] = None
        self._current_domain: Optional[str] = None
        self._episode_done: bool = False
        # Curriculum tracking
        self._rolling_window = 50
        self._recent_results: list[str] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> WatchDogObservation:
        """Start a new episode: generate a conversation to review."""
        self._state.episode_id = episode_id or str(uuid.uuid4())
        self._state.step_count = 0
        self._state.total_episodes += 1
        self._episode_done = False

        self._maybe_advance_level()

        conversation, manifest = sample_episode(self._state.current_level)
        self._current_manifest = manifest
        self._current_conversation = conversation
        self._current_domain = manifest.get("domain", "unknown")

        return WatchDogObservation(
            conversation=conversation,
            task_domain=self._current_domain,
            task_id=self._state.episode_id,
            difficulty=self._state.current_level,
            feedback=None,
            done=False,
            reward=None,
        )

    def step(
        self,
        action: WatchDogAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> WatchDogObservation:
        """Process the oversight agent's verdict and return reward."""
        self._state.step_count += 1

        reward, feedback, result_type = compute_reward(action, self._current_manifest or {})
        self._state.cumulative_reward += reward
        self._track_result(result_type)
        self._episode_done = True

        return WatchDogObservation(
            conversation=self._current_conversation or "",
            task_domain=self._current_domain or "unknown",
            task_id=self._state.episode_id or "",
            difficulty=self._state.current_level,
            feedback=feedback,
            done=True,
            reward=reward,
        )

    @property
    def state(self) -> WatchDogState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="WatchDog",
            description="RL environment for training AI oversight agents to detect errors in AI conversations",
            version="0.1.0",
            author="WatchDog Team",
        )

    def _track_result(self, result_type: str) -> None:
        """Track TP/FP/TN/FN for curriculum and metrics."""
        if result_type == "tp":
            self._state.true_positives += 1
        elif result_type == "fp":
            self._state.false_positives += 1
        elif result_type == "tn":
            self._state.true_negatives += 1
        elif result_type == "fn":
            self._state.false_negatives += 1

        self._recent_results.append(result_type)
        if len(self._recent_results) > self._rolling_window:
            self._recent_results.pop(0)

    def _compute_rolling_f1(self) -> float:
        """Compute F1 over the recent window."""
        if len(self._recent_results) < 20:
            return 0.0
        tp = self._recent_results.count("tp")
        fp = self._recent_results.count("fp")
        fn = self._recent_results.count("fn")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _maybe_advance_level(self) -> None:
        """Auto-advance curriculum level based on F1."""
        f1 = self._compute_rolling_f1()
        thresholds = {1: 0.60, 2: 0.65, 3: 0.70}
        current = self._state.current_level
        if current in thresholds and f1 > thresholds[current]:
            self._state.current_level = min(current + 1, 4)
            self._recent_results.clear()
