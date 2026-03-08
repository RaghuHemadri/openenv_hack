"""WatchDog Environment — Server-side step-based implementation.

Flow:
    1. User calls reset()  → new game via selected plugin (avalon/cicero)
    2. User calls step()   → plugin advances one turn, optionally mutated
    3. Overseer decides: pass / flag / question
"""

import importlib
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import (
    MultiTurnAction,
    MultiTurnObservation,
    MultiTurnState,
    agent_turn_to_dict,
)
from error_engine import generate_question_response, maybe_mutate, start_episode
from rewards import (
    compute_flag_reward,
    compute_pass_reward,
    compute_question_cost,
    compute_episode_end_bonus,
)


def _get_plugin(game_id: str):
    """Get plugin from registry by game_id."""
    try:
        from plugins import get_plugin
    except ImportError:
        from watchdog_env.plugins import get_plugin
    plugin = get_plugin(game_id)
    if plugin is None:
        raise RuntimeError(
            f"Plugin '{game_id}' not registered. Import plugins to register."
        )
    return plugin


def _get_plugin_config(game_id: str, level: int) -> Any:
    """Get plugin-specific config for the given level."""
    if game_id == "avalon":
        try:
            from plugins.avalon import AvalonConfig
        except ImportError:
            from watchdog_env.plugins.avalon import AvalonConfig
        return AvalonConfig(level=level)
    if game_id == "cicero":
        try:
            from plugins.cicero.cicero_config import CiceroConfig
        except ImportError:
            from watchdog_env.plugins.cicero.cicero_config import CiceroConfig
        return CiceroConfig(num_steps=5)
    raise ValueError(f"No config factory for game_id={game_id}")


class WatchDogMultiTurnEnvironment(
    Environment[MultiTurnAction, MultiTurnObservation, MultiTurnState]
):
    """Multi-turn RL environment for training AI oversight agents.

    Each step():
        1. Gets the next turn from the selected plugin (avalon/cicero)
        2. Optionally mutates the turn (avalon: Werewolf turns only)
        3. Presents it to the Overseer for judgement
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_QUESTIONS_PER_EPISODE: int = 2

    def __init__(
        self,
        game_id: str = "avalon",
        use_mutations: bool = True,
        use_llm: bool = True,
    ) -> None:
        super().__init__()
        self._state = MultiTurnState(episode_id=str(uuid.uuid4()), step_count=0)

        # Plugin selection
        self._game_id = game_id
        self._use_mutations = use_mutations
        self._use_llm = use_llm
        self._plugin = None
        self._env_name: str = game_id

        # Current turn state
        self._current_step: Any = None  # MultiAgentStep
        self._current_turn: dict[str, Any] | None = None
        self._current_response: str = ""
        self._current_has_error: bool = False
        self._current_error_detail: dict[str, Any] | None = None
        self._question_response_cache: dict[str, str] | None = None

        # Episode tracking
        self._phase: str = "observe"
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
        """Start a new oversight episode backed by the selected plugin."""
        import os
        if self._use_llm:
            os.environ.pop("WATCHDOG_AVALON_USE_TEMPLATE", None)
            os.environ.pop("WATCHDOG_CICERO_USE_TEMPLATE", None)
            try:
                m = importlib.import_module("watchdog_env.plugins.avalon.llm")
                if hasattr(m, "_llm_instance"):
                    m._llm_instance = None
            except ImportError:
                pass
        else:
            os.environ["WATCHDOG_AVALON_USE_TEMPLATE"] = "1"
            os.environ["WATCHDOG_CICERO_USE_TEMPLATE"] = "1"
        self._plugin = _get_plugin(self._game_id)
        self._state.episode_id = episode_id or str(uuid.uuid4())
        self._state.step_count = 0
        self._state.total_episodes += 1
        self._episode_done = False
        self._maybe_advance_level()

        config = _get_plugin_config(self._game_id, self._state.current_level)
        self._plugin.reset(seed=seed, config=config)
        plugin_state = self._plugin.get_state()
        game_state = plugin_state.metadata.get("game_state")
        alive_count = len(game_state.alive_players) if game_state else 2

        # Initialize mutation tracking
        if self._use_mutations:
            if self._game_id == "avalon" and game_state:
                wolf_count = len(game_state.alive_wolves) if hasattr(game_state, "alive_wolves") else 2
                cfg = plugin_state.config
                num_rounds = cfg.get_num_rounds() if hasattr(cfg, "get_num_rounds") else 2
                start_episode(game_id="avalon", wolf_count=wolf_count, num_rounds=num_rounds)
            elif self._game_id == "cicero":
                cfg = plugin_state.config
                num_steps = cfg.num_steps if hasattr(cfg, "num_steps") else 5
                start_episode(game_id="cicero", num_steps=num_steps)

        self._episode_reward = 0.0
        self._questions_remaining = self.MAX_QUESTIONS_PER_EPISODE
        self._flags_issued = 0
        self._turns_seen = []
        self._flagged_error_turns = set()
        self._all_flag_turns = set()
        self._phase = "observe"

        self._advance_game_turn()

        return self._build_observation(
            step_reward=None,
            feedback=f"New game started. {alive_count} players.",
        )

    def step(
        self,
        action: MultiTurnAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MultiTurnObservation:
        """Process Overseer action on the current turn, then advance the game."""
        self._state.step_count += 1
        action_type = action.action_type.lower().strip()
        turn_idx = len(self._turns_seen) - 1

        round_data = {
            "has_error": self._current_has_error,
            "error_detail": self._current_error_detail,
            "worker_response": self._current_response,
        }

        if action_type == "pass":
            reward, feedback = compute_pass_reward(round_data)
            self._episode_reward += reward
            if not self._current_has_error:
                self._state.correct_passes += 1
            self._advance_game_turn()
            self._phase = "observe"
            if self._game_done():
                return self._end_episode(reward, feedback)
            return self._build_observation(step_reward=reward, feedback=feedback)

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
            self._question_response_cache = generate_question_response(
                worker_response=self._current_response,
                has_error=self._current_has_error,
                error_detail=self._current_error_detail,
                level=self._state.current_level,
                context={"turn": self._current_turn},
                game_id=self._game_id,
            )
            self._phase = "question_response"
            return self._build_observation(
                step_reward=q_cost,
                feedback=f"{q_feedback} Player responded. Now decide: PASS or FLAG.",
                question_response_text=self._question_response_cache.get("response", ""),
            )

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
                "Step-based oversight environment. Uses Avalon (Werewolf) plugin "
                "with LangChain-orchestrated LLM player turns."
            ),
            version="0.4.0",
            author="WatchDog Team",
        )

    # ── Batched-episode support ─────────────────────────────────────

    def reset_deferred(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Reset environment without generating the first turn.

        For batched episode generation: call prepare_advance() + batch LLM +
        complete_advance() to produce the first turn.
        """
        import os
        if self._use_llm:
            os.environ.pop("WATCHDOG_AVALON_USE_TEMPLATE", None)
            os.environ.pop("WATCHDOG_CICERO_USE_TEMPLATE", None)
        else:
            os.environ["WATCHDOG_AVALON_USE_TEMPLATE"] = "1"
            os.environ["WATCHDOG_CICERO_USE_TEMPLATE"] = "1"
        self._plugin = _get_plugin(self._game_id)
        self._state.episode_id = episode_id or str(uuid.uuid4())
        self._state.step_count = 0
        self._state.total_episodes += 1
        self._episode_done = False
        self._maybe_advance_level()

        config = _get_plugin_config(self._game_id, self._state.current_level)
        self._plugin.reset(seed=seed, config=config)
        plugin_state = self._plugin.get_state()
        game_state = plugin_state.metadata.get("game_state")

        if self._use_mutations:
            if self._game_id == "avalon" and game_state:
                wolf_count = len(game_state.alive_wolves) if hasattr(game_state, "alive_wolves") else 2
                cfg = plugin_state.config
                num_rounds = cfg.get_num_rounds() if hasattr(cfg, "get_num_rounds") else 2
                start_episode(game_id="avalon", wolf_count=wolf_count, num_rounds=num_rounds)
            elif self._game_id == "cicero":
                cfg = plugin_state.config
                num_steps = cfg.num_steps if hasattr(cfg, "num_steps") else 5
                start_episode(game_id="cicero", num_steps=num_steps)

        self._episode_reward = 0.0
        self._questions_remaining = self.MAX_QUESTIONS_PER_EPISODE
        self._flags_issued = 0
        self._turns_seen = []
        self._flagged_error_turns = set()
        self._all_flag_turns = set()
        self._phase = "observe"
        self._pending_moderator_prompt: Optional[str] = None

    def prepare_advance(self):
        """Get LLM request info for the next game turn.

        Returns (speaker, game_state, moderator_prompt) tuple, or None if done.
        """
        if self._plugin is None:
            return None
        plugin_state = self._plugin.get_state()
        if plugin_state.done:
            return None
        step_index = len(self._turns_seen)
        info = self._plugin.prepare_next_turn(seed=None, step_index=step_index)
        if info is None:
            return None
        self._pending_moderator_prompt = info["moderator_prompt"]
        return (info["speaker"], info["game_state"], info["moderator_prompt"])

    def complete_advance(self, response: str) -> None:
        """Complete the next game turn with a pre-generated LLM response."""
        step_index = len(self._turns_seen)
        step = self._plugin.complete_turn(
            response, step_index, moderator_prompt=self._pending_moderator_prompt,
        )
        self._pending_moderator_prompt = None
        self._current_step = step
        self._apply_step_mutations(step)

    # ── Internal helpers ─────────────────────────────────────────

    def _apply_step_mutations(self, step) -> None:
        """Apply mutations to a MultiAgentStep and store the resulting turn."""
        if not step.turns:
            self._current_turn = None
            return

        turn = step.turns[-1]
        clean_response = turn.action_text
        displayed_response = clean_response
        has_error = False
        error_detail = None

        if self._use_mutations:
            if self._game_id == "avalon":
                speaker_role = turn.metadata.get("role", "")
                context = {
                    "turn": turn.metadata,
                    "speaker_id": turn.agent_id,
                    "day": turn.metadata.get("day"),
                    "round_idx": turn.metadata.get("round_idx"),
                }
                displayed_response, has_error, error_detail = maybe_mutate(
                    clean_response=clean_response,
                    speaker_role=speaker_role,
                    level=self._state.current_level,
                    context=context,
                    game_id="avalon",
                )
                plugin_state = self._plugin.get_state()
                game_state = plugin_state.metadata.get("game_state")
                if game_state and game_state.conversation_log and displayed_response != clean_response:
                    game_state.conversation_log[-1]["message"] = displayed_response
            elif self._game_id == "cicero":
                plugin_state = self._plugin.get_state()
                context = {
                    "turn": turn.metadata,
                    "speaker_id": turn.agent_id,
                    "step_index": len(self._turns_seen),
                    "season": turn.metadata.get("season"),
                    "region": turn.metadata.get("region"),
                    "domain_name": turn.metadata.get("domain_name"),
                    "domain_desc": turn.metadata.get("domain_desc"),
                    "counterpart": turn.metadata.get("counterpart"),
                }
                displayed_response, has_error, error_detail = maybe_mutate(
                    clean_response=clean_response,
                    speaker_role="",
                    level=self._state.current_level,
                    context=context,
                    game_id="cicero",
                )
                if plugin_state.conversation_log and displayed_response != clean_response:
                    plugin_state.conversation_log[-1]["message"] = displayed_response

        self._current_response = displayed_response
        self._current_has_error = has_error
        self._current_error_detail = error_detail
        self._question_response_cache = None

        turn_dict = agent_turn_to_dict(
            turn,
            has_error=has_error,
            displayed_response=displayed_response,
            error_detail=error_detail,
        )
        turn_dict["game_over"] = step.done
        turn_dict["winner"] = turn.metadata.get("winner")
        self._turns_seen.append(turn_dict)
        self._current_turn = turn_dict

    def _advance_game_turn(self) -> None:
        """Get next turn from the plugin. Optionally mutate (avalon: Werewolf turns)."""
        if self._plugin is None:
            self._current_turn = None
            return

        plugin_state = self._plugin.get_state()
        if plugin_state.done:
            self._current_turn = None
            return

        step_index = len(self._turns_seen)
        step = self._plugin.generate_step(seed=None, step_index=step_index)
        self._current_step = step
        self._apply_step_mutations(step)

    def _game_done(self) -> bool:
        if self._plugin is None:
            return True
        if self._plugin.get_state().done:
            return True
        if self._current_turn is None:
            return True
        return self._current_turn.get("game_over", False)

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
        domain = self._env_name.replace("_", " ").title()
        lines = [f"[OVERSIGHT SESSION — {domain}]\n"]
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
            all_rounds=[{"has_error": t.get("has_error", False)} for t in self._turns_seen],
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
