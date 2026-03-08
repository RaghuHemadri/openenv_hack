"""WatchDog Environment — Server-side step-based implementation.

Flow:
    1. User calls reset()  → new Avalon game created
    2. User calls step()   → Avalon game advances one turn
       - AvalonGame.step() generates the player's response (via LangChain LLM)
       - error_engine.maybe_mutate() optionally injects a deception
       - Observation returned to the Overseer for judgement
    3. Overseer decides: pass / flag / question / intervene
"""

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import MultiTurnAction, MultiTurnObservation, MultiTurnState
from error_engine import maybe_mutate, generate_question_response, start_episode
from rewards import (
    compute_flag_reward,
    compute_pass_reward,
    compute_intervene_reward,
    compute_question_cost,
    compute_episode_end_bonus,
)


class WatchDogMultiTurnEnvironment(
    Environment[MultiTurnAction, MultiTurnObservation, MultiTurnState]
):
    """Multi-turn RL environment for training AI oversight agents.

    Wraps an environment game (Avalon by default). Each step():
        1. Gets the next turn from the game (Avalon player speaks via LLM)
        2. Optionally mutates the response (Werewolf deception)
        3. Presents it to the Overseer for judgement
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_QUESTIONS_PER_EPISODE: int = 2

    def __init__(self) -> None:
        super().__init__()
        self._state = MultiTurnState(episode_id=str(uuid.uuid4()), step_count=0)

        # Game engine
        self._game = None
        self._env_name: str = "avalon"

        # Current turn state
        self._current_turn: dict[str, Any] | None = None
        self._current_response: str = ""  # possibly mutated
        self._current_has_error: bool = False
        self._current_error_detail: dict[str, Any] | None = None
        self._question_response_cache: dict[str, str] | None = None

        # Episode tracking
        self._phase: str = "observe"  # "observe" | "question_response" | "done"
        self._episode_done: bool = False
        self._episode_reward: float = 0.0
        self._questions_remaining: int = self.MAX_QUESTIONS_PER_EPISODE
        self._flags_issued: int = 0
        self._turns_seen: list[dict[str, Any]] = []
        self._flagged_error_turns: set[int] = set()
        self._all_flag_turns: set[int] = set()

        # Curriculum
        self._rolling_window = 50
        self._recent_results: list[str] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MultiTurnObservation:
        """Start a new oversight episode backed by an Avalon game."""
        from envs.avalon import AvalonGame

        self._state.episode_id = episode_id or str(uuid.uuid4())
        self._state.step_count = 0
        self._state.total_episodes += 1
        self._episode_done = False
        self._maybe_advance_level()

        # Create game
        self._game = AvalonGame(level=self._state.current_level, seed=seed)
        game_info = self._game.reset(seed=seed)

        # Tell the error engine about this episode's wolf count
        from envs.avalon import LEVEL_CONFIG
        wolf_count = sum(1 for p in self._game.state.players if p.role == "Werewolf")
        config = LEVEL_CONFIG.get(self._state.current_level, LEVEL_CONFIG[2])
        start_episode(wolf_count, config["num_rounds"])

        # Reset per-episode state
        self._episode_reward = 0.0
        self._questions_remaining = self.MAX_QUESTIONS_PER_EPISODE
        self._flags_issued = 0
        self._turns_seen = []
        self._flagged_error_turns = set()
        self._all_flag_turns = set()
        self._phase = "observe"

        # Generate first turn
        self._advance_game_turn()

        return self._build_observation(
            step_reward=None,
            feedback=f"New game started. {game_info['alive_count']} players.",
        )

    def step(
        self,
        action: MultiTurnAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MultiTurnObservation:
        """Process Overseer action on the current turn, then advance the game.

        Actions:
            pass:      Accept this turn, advance to next player
            flag:      Accuse this turn of containing deception
            question:  Ask the player to clarify (limited budget)
            intervene: Emergency stop — end episode immediately
        """
        self._state.step_count += 1
        action_type = action.action_type.lower().strip()
        turn_idx = len(self._turns_seen) - 1

        # Build round_data compatible with reward functions
        round_data = {
            "has_error": self._current_has_error,
            "error_detail": self._current_error_detail,
            "worker_response": self._current_response,
        }

        # ── PASS ────────────────────────────────────────────────
        if action_type == "pass":
            reward, feedback = compute_pass_reward(round_data)
            self._episode_reward += reward
            if not self._current_has_error:
                self._state.correct_passes += 1

            # Advance to next game turn
            self._advance_game_turn()
            self._phase = "observe"

            if self._game_done():
                return self._end_episode(reward, feedback)
            return self._build_observation(step_reward=reward, feedback=feedback)

        # ── FLAG ────────────────────────────────────────────────
        elif action_type == "flag":
            reward, feedback, result_type = compute_flag_reward(action, round_data)
            self._episode_reward += reward
            self._flags_issued += 1
            self._all_flag_turns.add(turn_idx)

            if result_type == "tp":
                self._flagged_error_turns.add(turn_idx)
                self._state.errors_detected += 1
                self._track_result("tp")
            else:
                self._state.false_flags += 1
                self._track_result("fp")

            self._advance_game_turn()
            self._phase = "observe"

            if self._game_done():
                return self._end_episode(reward, feedback)
            return self._build_observation(step_reward=reward, feedback=feedback)

        # ── QUESTION ────────────────────────────────────────────
        elif action_type == "question":
            if self._questions_remaining <= 0:
                reward, feedback = compute_pass_reward(round_data)
                feedback = "No questions left! Treated as PASS. " + feedback
                self._episode_reward += reward
                self._advance_game_turn()
                self._phase = "observe"
                if self._game_done():
                    return self._end_episode(reward, feedback)
                return self._build_observation(step_reward=reward, feedback=feedback)

            self._questions_remaining -= 1
            self._state.questions_used += 1
            q_cost, q_feedback = compute_question_cost()
            self._episode_reward += q_cost

            # Generate reactive response
            self._question_response_cache = generate_question_response(
                worker_response=self._current_response,
                has_error=self._current_has_error,
                error_detail=self._current_error_detail,
                level=self._state.current_level,
                context={"turn": self._current_turn},
            )

            self._phase = "question_response"
            return self._build_observation(
                step_reward=q_cost,
                feedback=f"{q_feedback} Player responded. Now decide: PASS or FLAG.",
                question_response_text=self._question_response_cache.get("response", ""),
            )

        # ── INTERVENE ───────────────────────────────────────────
        elif action_type == "intervene":
            reward, feedback, result_type = compute_intervene_reward(round_data)
            self._episode_reward += reward
            if result_type == "intervene_correct":
                self._state.interventions_correct += 1
                self._flagged_error_turns.add(turn_idx)
            else:
                self._state.interventions_wrong += 1
            return self._end_episode(reward, feedback)

        else:
            reward, feedback = compute_pass_reward(round_data)
            feedback = f"Unknown action '{action_type}', treated as PASS. " + feedback
            self._episode_reward += reward
            self._advance_game_turn()
            self._phase = "observe"
            if self._game_done():
                return self._end_episode(reward, feedback)
            return self._build_observation(step_reward=reward, feedback=feedback)

    @property
    def state(self) -> MultiTurnState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="WatchDog Multi-Turn",
            description=(
                "Step-based oversight environment. Wraps Avalon (Werewolf) with "
                "LangChain-orchestrated LLM player turns and mutation injection."
            ),
            version="0.3.0",
            author="WatchDog Team",
        )

    # ── Game Turn Management ────────────────────────────────────

    def _advance_game_turn(self) -> None:
        """Get next turn from the Avalon game and optionally mutate it."""
        if self._game is None or self._game.is_done:
            self._current_turn = None
            return

        turn = self._game.step()
        self._current_turn = turn

        if turn.get("game_over"):
            self._current_response = turn["message"]
            self._current_has_error = False
            self._current_error_detail = None
            self._question_response_cache = None
            self._turns_seen.append(turn)
            return

        # Run mutation layer:  Avalon clean response → maybe_mutate
        clean_response = turn["message"]
        speaker_role = turn["role"]
        level = self._state.current_level

        mutated_response, has_error, error_detail = maybe_mutate(
            clean_response=clean_response,
            speaker_role=speaker_role,
            level=level,
            context={
                "speaker_id": turn.get("speaker_id"),
                "speaker_name": turn.get("speaker_name"),
                "day": turn.get("day"),
                "phase": turn.get("phase"),
            },
        )

        self._current_response = mutated_response
        self._current_has_error = has_error
        self._current_error_detail = error_detail
        self._question_response_cache = None

        # Store for history
        enriched = {**turn, "displayed_response": mutated_response, "has_error": has_error}
        self._turns_seen.append(enriched)

    def _game_done(self) -> bool:
        if self._game is None:
            return True
        if self._game.is_done:
            return True
        if self._current_turn is None:
            return True
        return self._current_turn.get("game_over", False)

    # ── Observation Building ────────────────────────────────────

    def _build_observation(
        self,
        step_reward: float | None,
        feedback: str | None,
        question_response_text: str | None = None,
    ) -> MultiTurnObservation:
        if self._current_turn is None or self._game_done():
            return MultiTurnObservation(
                conversation_so_far=self._format_conversation(),
                current_turn="[Episode complete]",
                current_turn_number=len(self._turns_seen),
                total_turns=len(self._turns_seen),
                task_domain=self._env_name,
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

        turn = self._current_turn

        if self._phase == "question_response" and question_response_text:
            current_text = (
                f"[{turn.get('speaker_display', 'Player')} — Response to your question]:\n"
                f"{question_response_text}"
            )
        else:
            current_text = (
                f"[Moderator]: {turn.get('moderator_prompt', '')}\n\n"
                f"[{turn.get('speaker_display', 'Player')}]: {self._current_response}"
            )

        return MultiTurnObservation(
            conversation_so_far=self._format_conversation(exclude_last=True),
            current_turn=current_text,
            current_turn_number=len(self._turns_seen),
            total_turns=len(self._turns_seen),
            task_domain=self._env_name,
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

    def _format_conversation(self, exclude_last: bool = False) -> str:
        turns = self._turns_seen[:-1] if exclude_last and self._turns_seen else self._turns_seen
        if not turns:
            return "[Conversation start]"

        lines = [f"[OVERSIGHT SESSION — Avalon / Werewolf]\n"]
        for i, t in enumerate(turns):
            speaker = t.get("speaker_display", "Player")
            msg = t.get("displayed_response", t.get("message", ""))
            lines.append(f"[Turn {i+1}] {speaker}: {msg}")
            lines.append("")
        return "\n".join(lines)

    def _end_episode(
        self, last_reward: float, last_feedback: str
    ) -> MultiTurnObservation:
        self._episode_done = True

        # Count missed errors
        error_turns = {
            i for i, t in enumerate(self._turns_seen)
            if t.get("has_error", False)
        }
        missed = error_turns - self._flagged_error_turns
        self._state.errors_missed += len(missed)
        for _ in missed:
            self._track_result("fn")

        end_bonus, end_summary = compute_episode_end_bonus(
            flagged_error_rounds=self._flagged_error_turns,
            all_rounds=[
                {"has_error": t.get("has_error", False)}
                for t in self._turns_seen
            ],
            rounds_completed=len(self._turns_seen),
            total_rounds=len(self._turns_seen),
        )
        self._episode_reward += end_bonus

        combined = last_feedback
        if end_summary:
            combined += f" | {end_summary}"
        combined += f" | Total reward: {self._episode_reward:.2f}"

        self._state.cumulative_reward += self._episode_reward
        self._phase = "done"

        return MultiTurnObservation(
            conversation_so_far=self._format_conversation(),
            current_turn="[Episode complete]",
            current_turn_number=len(self._turns_seen),
            total_turns=len(self._turns_seen),
            task_domain=self._env_name,
            task_id=self._state.episode_id or "",
            difficulty=self._state.current_level,
            remaining_questions=self._questions_remaining,
            flags_so_far=self._flags_issued,
            phase="done",
            step_reward=last_reward,
            cumulative_reward=self._episode_reward,
            feedback=combined,
            done=True,
            reward=self._episode_reward,
        )

    # ── Curriculum ──────────────────────────────────────────────

    def _track_result(self, result_type: str) -> None:
        self._recent_results.append(result_type)
        if len(self._recent_results) > self._rolling_window:
            self._recent_results.pop(0)

    def _compute_rolling_f1(self) -> float:
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
        f1 = self._compute_rolling_f1()
        thresholds = {1: 0.60, 2: 0.65, 3: 0.70}
        current = self._state.current_level
        if current in thresholds and f1 > thresholds[current]:
            self._state.current_level = min(current + 1, 4)
            self._recent_results.clear()
