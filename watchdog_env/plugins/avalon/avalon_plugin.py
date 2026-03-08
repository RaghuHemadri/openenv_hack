"""Avalon (Werewolf) multi-agent plugin. Self-contained implementation.

Implements MultiAgentSystemPlugin with AgentTurn (display_name, moderator_prompt)
from shared models.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from watchdog_env.models import AgentTurn, MultiAgentConfig, MultiAgentState, MultiAgentStep
from watchdog_env.plugins.base import MultiAgentSystemPlugin

from .avalon_config import AvalonConfig, LEVEL_CONFIG
from .avalon_models import (
    GameState,
    Player,
    create_game,
    _DAY_EVENTS,
    _DAY_EVENTS_NO_DEATH,
    _DAY_OPENERS,
)
from .llm import _generate_player_response_llm

logger = logging.getLogger(__name__)


def _build_moderator_prompt(state: GameState, speaker: Player) -> str:
    """Build moderator prompt for this speaker's turn."""
    if state.round_idx == 0 and state.day == 1:
        event = random.choice(_DAY_EVENTS_NO_DEATH)
    elif state.round_idx == 0:
        last_killed = None
        if state.eliminated:
            last_id = state.eliminated[-1]
            p = next((x for x in state.players if x.player_id == last_id), None)
            if p:
                last_killed = p.display
        if last_killed:
            event = f"{last_killed} was found dead this morning — the Werewolves struck."
        else:
            event = random.choice(_DAY_EVENTS_NO_DEATH)
    else:
        event = random.choice(_DAY_EVENTS[1:])

    template = random.choice(_DAY_OPENERS)
    return template.format(day=state.day, event=event, speaker_disp=speaker.display)


def _setup_speaker_order(state: GameState) -> list[str]:
    """Randomise speaking order among alive players."""
    alive_ids = [p.player_id for p in state.alive_players]
    random.shuffle(alive_ids)
    return alive_ids


def _simulate_night(state: GameState) -> dict[str, Any]:
    """Simulate night phase: wolf kill, doctor save, police check."""
    alive = state.alive_players
    non_wolves = [p for p in alive if p.role != "Werewolf"]
    if non_wolves:
        kill_target = random.choice(non_wolves)
        state.night_kill = kill_target.player_id
    else:
        state.night_kill = None

    doctors = [p for p in alive if p.role == "Doctor"]
    if doctors:
        doctor = doctors[0]
        protectable = [p for p in alive if p.player_id != doctor.player_id]
        if protectable:
            save_target = random.choice(protectable)
            state.doctor_save = save_target.player_id
        else:
            state.doctor_save = None
    else:
        state.doctor_save = None

    police = [p for p in alive if p.role == "Police"]
    if police:
        seer = police[0]
        investigable = [p for p in alive if p.player_id != seer.player_id]
        if investigable:
            investigate = random.choice(investigable)
            state.police_result = {"target": investigate.player_id, "role": investigate.role}
    else:
        state.police_result = None

    if state.night_kill and state.night_kill != state.doctor_save:
        victim = next((p for p in state.players if p.player_id == state.night_kill), None)
        if victim:
            victim.alive = False
            state.eliminated.append(victim.player_id)

    state.day += 1
    state.phase = "day"
    state.round_idx = 0
    return {"killed": state.night_kill != state.doctor_save}


class AvalonPlugin(MultiAgentSystemPlugin):
    """Multi-agent Avalon (Werewolf) plugin. Self-contained implementation."""

    def __init__(self) -> None:
        self._state = MultiAgentState()

    def get_game_id(self) -> str:
        return "avalon"

    def get_display_name(self) -> str:
        return "Avalon (Werewolf)"

    def list_agent_ids(self) -> list[str]:
        game_state = self._state.metadata.get("game_state")
        if game_state is None:
            return []
        return [p.player_id for p in game_state.players]

    def reset(
        self,
        seed: int | None = None,
        config: MultiAgentConfig | None = None,
    ) -> None:
        if seed is not None:
            random.seed(seed)
        cfg = config if isinstance(config, AvalonConfig) else AvalonConfig()
        level = cfg.level if isinstance(cfg, AvalonConfig) else 2

        game_state = create_game(level=level, seed=seed)
        level_cfg = LEVEL_CONFIG.get(level, LEVEL_CONFIG[2])
        num_rounds = level_cfg.get("num_rounds", 2)
        max_rounds = num_rounds * len(game_state.alive_players)
        speaker_order = _setup_speaker_order(game_state)

        # conversation_log lives in game_state; wire state to use it as context
        self._state = MultiAgentState(
            step_index=0,
            turns_so_far=[],
            config=cfg,
            done=False,
            metadata={
                "game_state": game_state,
                "speaker_order": speaker_order,
                "speaker_idx": 0,
                "max_rounds": max_rounds,
                "total_turns": 0,
            },
            conversation_log=game_state.conversation_log,
        )

    def get_state(self) -> MultiAgentState:
        return self._state

    def prepare_next_turn(self, seed: int | None, step_index: int):
        """Resolve the next speaker without generating a response.

        Returns dict with speaker/game_state/moderator_prompt, or None if done.
        May simulate night phases as a side effect.
        Must be followed by complete_turn() with the LLM response.
        """
        if seed is not None:
            random.seed(seed)

        meta = self._state.metadata
        game_state: GameState = meta["game_state"]
        speaker_order: list[str] = meta["speaker_order"]
        speaker_idx: int = meta["speaker_idx"]
        max_rounds: int = meta["max_rounds"]
        total_turns: int = meta["total_turns"]

        if game_state.game_over or total_turns >= max_rounds:
            self._state.done = True
            return None

        # Skip dead players
        while speaker_idx < len(speaker_order):
            speaker_id = speaker_order[speaker_idx]
            speaker = next((p for p in game_state.players if p.player_id == speaker_id), None)
            if speaker and speaker.alive:
                break
            speaker_idx += 1

        if speaker_idx >= len(speaker_order):
            # Full rotation — simulate night
            _simulate_night(game_state)
            meta["speaker_order"] = _setup_speaker_order(game_state)
            meta["speaker_idx"] = 0
            return self.prepare_next_turn(seed, step_index)

        meta["speaker_idx"] = speaker_idx
        speaker = next((p for p in game_state.players if p.player_id == speaker_order[speaker_idx]), None)
        if speaker is None:
            self._state.done = True
            return None

        moderator_prompt = _build_moderator_prompt(game_state, speaker)
        return {
            "speaker": speaker,
            "game_state": game_state,
            "moderator_prompt": moderator_prompt,
        }

    def complete_turn(self, message: str, step_index: int, moderator_prompt: str | None = None) -> MultiAgentStep:
        """Finalize a turn with the given LLM response. Returns MultiAgentStep.

        Args:
            message: The generated player response text.
            step_index: Current step index.
            moderator_prompt: The prompt used to generate the response.
                If None, rebuilds it (may differ due to random.choice).
        """
        meta = self._state.metadata
        game_state: GameState = meta["game_state"]
        speaker_order: list[str] = meta["speaker_order"]
        speaker_idx: int = meta["speaker_idx"]
        total_turns: int = meta["total_turns"]
        max_rounds: int = meta["max_rounds"]

        speaker = next((p for p in game_state.players if p.player_id == speaker_order[speaker_idx]), None)
        if moderator_prompt is None:
            moderator_prompt = _build_moderator_prompt(game_state, speaker)

        turn = AgentTurn(
            agent_id=speaker.player_id,
            action_text=message,
            step_index=step_index,
            phase=game_state.phase,
            display_name=speaker.display,
            moderator_prompt=moderator_prompt,
            metadata={
                "speaker_name": speaker.name,
                "role": speaker.role,
                "day": game_state.day,
                "round_idx": game_state.round_idx,
                "game_over": game_state.game_over,
                "winner": game_state.winner,
            },
        )

        game_state.conversation_log.append({
            "speaker_id": speaker.player_id,
            "speaker_display": speaker.display,
            "message": message,
            "moderator_prompt": moderator_prompt,
        })
        game_state.round_idx += 1
        total_turns += 1

        # Advance speaker
        speaker_idx += 1
        if speaker_idx >= len(speaker_order):
            _simulate_night(game_state)
            meta["speaker_order"] = _setup_speaker_order(game_state)
            meta["speaker_idx"] = 0
        else:
            meta["speaker_idx"] = speaker_idx

        meta["total_turns"] = total_turns
        self._state.turns_so_far.append(turn)
        self._state.step_index = step_index + 1
        self._state.done = game_state.game_over or total_turns >= max_rounds

        return MultiAgentStep(
            turns=[turn],
            done=self._state.done,
            step_index=step_index,
            game_id=self.get_game_id(),
            task_id="",
            domain=self.get_game_id(),
            state=MultiAgentState(
                step_index=self._state.step_index,
                turns_so_far=list(self._state.turns_so_far),
                config=self._state.config,
                done=self._state.done,
                metadata=dict(meta),
                conversation_log=self._state.conversation_log,
            ),
        )

    def _make_done_step(self, step_index: int) -> MultiAgentStep:
        """Create a 'game over' MultiAgentStep."""
        meta = self._state.metadata
        game_state: GameState = meta["game_state"]
        total_turns: int = meta["total_turns"]

        turn = AgentTurn(
            agent_id="SYSTEM",
            action_text=(
                f"Game over. "
                f"{'The village wins!' if game_state.winner == 'village' else 'The Werewolves win!'}"
            ),
            step_index=step_index,
            phase="end",
            display_name="[SYSTEM] Moderator",
            moderator_prompt="",
            metadata={"game_over": True, "winner": game_state.winner},
        )
        self._state.turns_so_far.append(turn)
        self._state.done = True
        meta["total_turns"] = total_turns + 1
        return MultiAgentStep(
            turns=[turn],
            done=True,
            step_index=step_index,
            game_id=self.get_game_id(),
            task_id="",
            domain=self.get_game_id(),
            state=MultiAgentState(
                step_index=self._state.step_index + 1,
                turns_so_far=list(self._state.turns_so_far),
                config=self._state.config,
                done=True,
                metadata=dict(meta),
                conversation_log=self._state.conversation_log,
            ),
        )

    def generate_step(self, seed: int | None, step_index: int) -> MultiAgentStep:
        """Generate one game turn (resolve speaker + LLM + finalize)."""
        info = self.prepare_next_turn(seed, step_index)
        if info is None:
            return self._make_done_step(step_index)
        message = _generate_player_response_llm(
            info["speaker"], info["game_state"], info["moderator_prompt"],
        )
        return self.complete_turn(message, step_index, moderator_prompt=info["moderator_prompt"])
