"""WatchDog Environment — Server-side implementation."""

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import WatchDogAction, WatchDogObservation, WatchDogState
from models import MultiTurnAction, MultiTurnObservation, MultiTurnState
from error_engine import sample_episode, generate_multiturn_episode
from rewards import (
    compute_reward,
    compute_flag_reward,
    compute_pass_reward,
    compute_intervene_reward,
    compute_question_cost,
    compute_episode_end_bonus,
)


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


# ═══════════════════════════════════════════════════════════════════
# MULTI-TURN ENVIRONMENT (Avalon/Werewolf-style dynamic oversight)
# ═══════════════════════════════════════════════════════════════════


class WatchDogMultiTurnEnvironment(
    Environment[MultiTurnAction, MultiTurnObservation, MultiTurnState]
):
    """Multi-turn RL environment for training AI oversight agents.

    Inspired by social deduction games (Avalon / Werewolf):
    - Worker turns are "villagers" — some secretly contain errors ("werewolves")
    - The Overseer is the "Seer" who investigates turns (QUESTION) and accuses (FLAG)
    - Limited investigation budget forces strategic decision-making
    - At high difficulty, workers respond deceptively to questions (werewolves lie)

    Layer 1 (Incremental Revelation):
        Conversation unfolds turn-by-turn. Overseer decides after each worker turn.

    Layer 2 (Reactive Worker):
        QUESTION action triggers branching responses — honest backtrack (easy)
        or deceptive doubledown (hard), creating theory-of-mind dynamics.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_QUESTIONS_PER_EPISODE: int = 2

    def __init__(self) -> None:
        super().__init__()
        self._state = MultiTurnState(episode_id=str(uuid.uuid4()), step_count=0)

        # Episode state
        self._rounds: list[dict] = []
        self._domain: str = ""
        self._current_round_idx: int = 0  # which round we're showing
        self._phase: str = "observe"  # "observe" | "question_response" | "done"
        self._episode_done: bool = False

        # Per-episode tracking
        self._episode_reward: float = 0.0
        self._questions_remaining: int = self.MAX_QUESTIONS_PER_EPISODE
        self._flags_issued: int = 0
        self._flagged_error_rounds: set[int] = set()  # round indices correctly flagged
        self._all_flag_rounds: set[int] = set()  # all rounds flagged (for FP tracking)

        # Curriculum tracking
        self._rolling_window = 50
        self._recent_results: list[str] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MultiTurnObservation:
        """Start a new multi-turn oversight episode.

        Generates a conversation with N worker turns (some containing errors)
        and returns the first round for the Overseer to evaluate.
        """
        self._state.episode_id = episode_id or str(uuid.uuid4())
        self._state.step_count = 0
        self._state.total_episodes += 1
        self._episode_done = False

        self._maybe_advance_level()

        # Generate multi-turn conversation
        rounds, domain, task_id = generate_multiturn_episode(
            self._state.current_level, seed=seed
        )
        self._rounds = rounds
        self._domain = domain
        self._current_round_idx = 0
        self._phase = "observe"

        # Reset per-episode counters
        self._episode_reward = 0.0
        self._questions_remaining = self.MAX_QUESTIONS_PER_EPISODE
        self._flags_issued = 0
        self._flagged_error_rounds = set()
        self._all_flag_rounds = set()

        return self._build_observation(
            step_reward=None,
            feedback=None,
        )

    def step(
        self,
        action: MultiTurnAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MultiTurnObservation:
        """Process the Overseer's action and advance the episode.

        Actions:
            pass:      Let this turn through, advance to next round
            flag:      Accuse this turn of containing an error
            question:  Investigate — ask the Worker to clarify (limited budget)
            intervene: Emergency stop — ends episode immediately
        """
        self._state.step_count += 1
        current_round = self._rounds[self._current_round_idx]
        action_type = action.action_type.lower().strip()

        # ── PASS ────────────────────────────────────────────────
        if action_type == "pass":
            reward, feedback = compute_pass_reward(current_round)
            self._episode_reward += reward

            if not current_round.get("has_error", False):
                self._state.correct_passes += 1

            # Advance to next round
            self._current_round_idx += 1
            self._phase = "observe"

            # Check if episode is over
            if self._current_round_idx >= len(self._rounds):
                return self._end_episode(reward, feedback)

            return self._build_observation(
                step_reward=reward,
                feedback=feedback if feedback else None,
            )

        # ── FLAG ────────────────────────────────────────────────
        elif action_type == "flag":
            reward, feedback, result_type = compute_flag_reward(action, current_round)
            self._episode_reward += reward
            self._flags_issued += 1
            self._all_flag_rounds.add(self._current_round_idx)

            if result_type == "tp":
                self._flagged_error_rounds.add(self._current_round_idx)
                self._state.errors_detected += 1
                self._track_result("tp")
            else:
                self._state.false_flags += 1
                self._track_result("fp")

            # Advance to next round after flagging
            self._current_round_idx += 1
            self._phase = "observe"

            if self._current_round_idx >= len(self._rounds):
                return self._end_episode(reward, feedback)

            return self._build_observation(
                step_reward=reward,
                feedback=feedback,
            )

        # ── QUESTION ────────────────────────────────────────────
        elif action_type == "question":
            if self._questions_remaining <= 0:
                # No budget left — treat as PASS
                reward, feedback = compute_pass_reward(current_round)
                feedback = "No questions remaining! Treated as PASS. " + feedback
                self._episode_reward += reward
                self._current_round_idx += 1
                self._phase = "observe"

                if self._current_round_idx >= len(self._rounds):
                    return self._end_episode(reward, feedback)

                return self._build_observation(
                    step_reward=reward,
                    feedback=feedback,
                )

            # Deduct question budget
            self._questions_remaining -= 1
            self._state.questions_used += 1
            q_cost, q_feedback = compute_question_cost()
            self._episode_reward += q_cost

            # Get reactive worker response
            q_responses = current_round.get("question_responses", {})
            worker_q_response = q_responses.get(
                "response",
                "I'm happy to elaborate on that point.",
            )
            response_type = q_responses.get("type", "elaborate")

            # Stay on same round — Overseer must now PASS or FLAG
            self._phase = "question_response"

            return self._build_observation(
                step_reward=q_cost,
                feedback=f"{q_feedback} Worker responded ({response_type}). Now decide: PASS or FLAG.",
                question_response_text=worker_q_response,
            )

        # ── INTERVENE ───────────────────────────────────────────
        elif action_type == "intervene":
            reward, feedback, result_type = compute_intervene_reward(current_round)
            self._episode_reward += reward

            if result_type == "intervene_correct":
                self._state.interventions_correct += 1
                self._flagged_error_rounds.add(self._current_round_idx)
            else:
                self._state.interventions_wrong += 1

            return self._end_episode(reward, feedback)

        else:
            # Unknown action — treat as PASS
            reward, feedback = compute_pass_reward(current_round)
            feedback = f"Unknown action '{action_type}', treated as PASS. " + feedback
            self._episode_reward += reward
            self._current_round_idx += 1
            self._phase = "observe"

            if self._current_round_idx >= len(self._rounds):
                return self._end_episode(reward, feedback)

            return self._build_observation(
                step_reward=reward,
                feedback=feedback,
            )

    @property
    def state(self) -> MultiTurnState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="WatchDog Multi-Turn",
            description=(
                "Avalon/Werewolf-inspired multi-turn RL environment for training "
                "AI oversight agents. Workers hide errors in conversation turns; "
                "the Overseer must investigate and detect them."
            ),
            version="0.2.0",
            author="WatchDog Team",
        )

    # ── Private helpers ─────────────────────────────────────────

    def _build_observation(
        self,
        step_reward: float | None,
        feedback: str | None,
        question_response_text: str | None = None,
    ) -> MultiTurnObservation:
        """Build an observation for the current state of the episode."""
        if self._current_round_idx >= len(self._rounds):
            # Episode over
            return MultiTurnObservation(
                conversation_so_far=self._format_conversation(len(self._rounds)),
                current_turn="[Episode complete]",
                current_turn_number=len(self._rounds),
                total_turns=len(self._rounds),
                task_domain=self._domain,
                task_id=self._state.episode_id or "",
                difficulty=self._state.current_level,
                remaining_questions=self._questions_remaining,
                flags_so_far=self._flags_issued,
                phase="done",
                step_reward=step_reward,
                cumulative_reward=self._episode_reward,
                feedback=feedback,
                done=True,
                reward=self._episode_reward,
            )

        current_round = self._rounds[self._current_round_idx]

        # Build current turn text
        if self._phase == "question_response" and question_response_text:
            current_turn = (
                f"[Worker AI — Response to your question]:\n"
                f"{question_response_text}"
            )
        else:
            current_turn = (
                f"[User]: {current_round['user_msg']}\n\n"
                f"[Worker AI]: {current_round['worker_response']}"
            )

        return MultiTurnObservation(
            conversation_so_far=self._format_conversation(self._current_round_idx),
            current_turn=current_turn,
            current_turn_number=self._current_round_idx + 1,
            total_turns=len(self._rounds),
            task_domain=self._domain,
            task_id=self._state.episode_id or "",
            difficulty=self._state.current_level,
            remaining_questions=self._questions_remaining,
            flags_so_far=self._flags_issued,
            phase=self._phase,
            step_reward=step_reward,
            cumulative_reward=self._episode_reward,
            feedback=feedback,
            done=False,
            reward=step_reward,
        )

    def _format_conversation(self, up_to_round: int) -> str:
        """Format all conversation turns up to (but not including) the given round."""
        if up_to_round == 0:
            return "[Conversation start]"

        lines = [f"[OVERSIGHT SESSION — Domain: {self._domain}]\n"]
        for i in range(min(up_to_round, len(self._rounds))):
            r = self._rounds[i]
            lines.append(f"[Round {i + 1}]")
            lines.append(f"  User: {r['user_msg']}")
            lines.append(f"  Worker AI: {r['worker_response']}")
            lines.append("")
        return "\n".join(lines)

    def _end_episode(
        self, last_step_reward: float, last_step_feedback: str
    ) -> MultiTurnObservation:
        """Finalize the episode: compute end-of-episode bonuses."""
        self._episode_done = True

        # Compute end-of-episode bonuses (missed errors, efficiency)
        end_bonus, end_summary = compute_episode_end_bonus(
            flagged_error_rounds=self._flagged_error_rounds,
            all_rounds=self._rounds,
            rounds_completed=self._current_round_idx,
            total_rounds=len(self._rounds),
        )
        self._episode_reward += end_bonus

        # Track missed errors
        error_rounds = {
            i for i, r in enumerate(self._rounds) if r.get("has_error", False)
        }
        missed = error_rounds - self._flagged_error_rounds
        self._state.errors_missed += len(missed)
        for _ in missed:
            self._track_result("fn")

        combined_feedback = last_step_feedback
        if end_summary:
            combined_feedback += f" | Episode end: {end_summary}"
        combined_feedback += f" | Total reward: {self._episode_reward:.2f}"

        self._state.cumulative_reward += self._episode_reward
        self._phase = "done"

        return MultiTurnObservation(
            conversation_so_far=self._format_conversation(len(self._rounds)),
            current_turn="[Episode complete]",
            current_turn_number=len(self._rounds),
            total_turns=len(self._rounds),
            task_domain=self._domain,
            task_id=self._state.episode_id or "",
            difficulty=self._state.current_level,
            remaining_questions=self._questions_remaining,
            flags_so_far=self._flags_issued,
            phase="done",
            step_reward=last_step_reward,
            cumulative_reward=self._episode_reward,
            feedback=combined_feedback,
            done=True,
            reward=self._episode_reward,
        )

    def _track_result(self, result_type: str) -> None:
        """Track results for curriculum advancement."""
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
        """Auto-advance curriculum level based on rolling F1."""
        f1 = self._compute_rolling_f1()
        thresholds = {1: 0.60, 2: 0.65, 3: 0.70}
        current = self._state.current_level
        if current in thresholds and f1 > thresholds[current]:
            self._state.current_level = min(current + 1, 4)
            self._recent_results.clear()
