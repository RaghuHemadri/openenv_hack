"""Avalon (Werewolf) step-based game engine."""

from __future__ import annotations

import logging
import random
from typing import Any

from .avalon_models import (
    GameState,
    Player,
    create_game,
    _DAY_EVENTS,
    _DAY_EVENTS_NO_DEATH,
    _DAY_OPENERS,
)
from .avalon_config import LEVEL_CONFIG
from .llm import _generate_player_response_llm

logger = logging.getLogger(__name__)


class AvalonGame:
    """Step-based Werewolf game engine.

    Usage:
        game = AvalonGame(level=2)
        game.reset()

        while not game.is_done:
            turn = game.step()
            # turn = { speaker, message, moderator_prompt, phase, day, ... }
    """

    def __init__(self, level: int = 1, seed: int | None = None):
        self.level = level
        self._seed = seed
        self.state: GameState | None = None
        self._speaker_order: list[str] = []
        self._speaker_idx: int = 0
        self._max_rounds: int = 0
        self._total_turns: int = 0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Start a new game. Returns the initial game info (public)."""
        s = seed if seed is not None else self._seed
        self.state = create_game(level=self.level, seed=s)

        config = LEVEL_CONFIG.get(self.level, LEVEL_CONFIG[2])
        num_rotations = config["num_rounds"]
        self._max_rounds = num_rotations * len(self.state.alive_players)
        self._total_turns = 0
        self._setup_speaker_order()

        return self._public_game_info()

    def step(self) -> dict[str, Any]:
        """Advance one turn: pick speaker, generate their response, log it."""
        assert self.state is not None, "Call reset() first"

        if self.is_done:
            return self._done_turn()

        # Pick current speaker
        speaker_id = self._speaker_order[self._speaker_idx]
        speaker = self.state.get_player(speaker_id)

        # Skip dead players
        while speaker and not speaker.alive:
            self._advance_speaker()
            if self.is_done:
                return self._done_turn()
            speaker_id = self._speaker_order[self._speaker_idx]
            speaker = self.state.get_player(speaker_id)

        if speaker is None:
            return self._done_turn()

        # Build moderator prompt
        moderator_prompt = self._build_moderator_prompt(speaker)

        # Generate player response via LLM (sequential call)
        message = _generate_player_response_llm(speaker, self.state, moderator_prompt)

        # Log the turn
        turn_entry = {
            "speaker_id": speaker.player_id,
            "speaker_name": speaker.name,
            "speaker_display": speaker.display,
            "role": speaker.role,
            "message": message,
            "moderator_prompt": moderator_prompt,
            "day": self.state.day,
            "phase": self.state.phase,
            "round_idx": self.state.round_idx,
        }
        self.state.conversation_log.append(turn_entry)
        self.state.round_idx += 1
        self._total_turns += 1

        # Advance to next speaker
        self._advance_speaker()

        return {
            **turn_entry,
            "game_over": self.state.game_over,
            "winner": self.state.winner,
            "alive_players": [p.to_dict() for p in self.state.alive_players],
        }

    @property
    def is_done(self) -> bool:
        if self.state is None:
            return True
        if self.state.game_over:
            return True
        if self.state.round_idx >= self._max_rounds:
            return True
        if self._total_turns >= self._max_rounds:
            return True
        return False

    # ── Night Phase Simulation ──────────────────────────────────

    def simulate_night(self) -> dict[str, Any]:
        """Simulate a night phase: wolf kill, doctor save, police check."""
        assert self.state is not None
        alive = self.state.alive_players

        # Wolves choose a target (random among non-wolves)
        non_wolves = [p for p in alive if p.role != "Werewolf"]
        if non_wolves:
            kill_target = random.choice(non_wolves)
            self.state.night_kill = kill_target.player_id
        else:
            self.state.night_kill = None

        # Doctor protects someone
        doctors = [p for p in alive if p.role == "Doctor"]
        if doctors:
            doctor = doctors[0]
            protectable = [p for p in alive if p.player_id != doctor.player_id]
            if protectable:
                save_target = random.choice(protectable)
                self.state.doctor_save = save_target.player_id
            else:
                self.state.doctor_save = None
        else:
            self.state.doctor_save = None

        # Police investigates someone
        police = [p for p in alive if p.role == "Police"]
        if police:
            seer = police[0]
            investigable = [p for p in alive if p.player_id != seer.player_id]
            if investigable:
                investigate = random.choice(investigable)
                self.state.police_result = {
                    "target": investigate.player_id,
                    "role": investigate.role,
                }
        else:
            self.state.police_result = None

        # Resolve: if doctor saved the kill target, nobody dies
        killed_name = None
        if self.state.night_kill and self.state.night_kill != self.state.doctor_save:
            victim = self.state.get_player(self.state.night_kill)
            if victim:
                victim.alive = False
                self.state.eliminated.append(victim.player_id)
                killed_name = victim.display

        # Advance day
        self.state.day += 1
        self.state.phase = "day"
        self.state.round_idx = 0
        self._setup_speaker_order()

        return {
            "night_kill_target": self.state.night_kill,
            "doctor_save_target": self.state.doctor_save,
            "saved": self.state.night_kill == self.state.doctor_save,
            "killed": killed_name,
            "police_result": self.state.police_result,
        }

    # ── Internal Helpers ────────────────────────────────────────

    def _setup_speaker_order(self):
        """Randomise the speaking order among alive players."""
        assert self.state is not None
        alive_ids = [p.player_id for p in self.state.alive_players]
        random.shuffle(alive_ids)
        self._speaker_order = alive_ids
        self._speaker_idx = 0

    def _advance_speaker(self):
        """Move to the next speaker. Trigger night phase after full rotation."""
        self._speaker_idx += 1
        if self._speaker_idx >= len(self._speaker_order):
            # Full rotation complete — run night phase before next day
            night_result = self.simulate_night()
            logger.info(
                "Night %d complete: killed=%s, saved=%s",
                self.state.day - 1,
                night_result.get("killed"),
                night_result.get("saved"),
            )

    def _build_moderator_prompt(self, speaker: Player) -> str:
        """Build the moderator's prompt for this speaker's turn."""
        assert self.state is not None

        if self.state.round_idx == 0 and self.state.day == 1:
            event = random.choice(_DAY_EVENTS_NO_DEATH)
        elif self.state.round_idx == 0:
            last_killed = None
            if self.state.eliminated:
                last_id = self.state.eliminated[-1]
                p = self.state.get_player(last_id)
                if p:
                    last_killed = p.display
            if last_killed:
                event = (
                    f"{last_killed} was found dead this morning "
                    f"— the Werewolves struck."
                )
            else:
                event = random.choice(_DAY_EVENTS_NO_DEATH)
        else:
            event = random.choice(_DAY_EVENTS[1:])

        template = random.choice(_DAY_OPENERS)
        return template.format(
            day=self.state.day,
            event=event,
            speaker_disp=speaker.display,
        )

    def _public_game_info(self) -> dict[str, Any]:
        """Return public information about the game (no hidden roles)."""
        assert self.state is not None
        return {
            "players": [
                {"player_id": p.player_id, "name": p.name, "alive": p.alive}
                for p in self.state.players
            ],
            "day": self.state.day,
            "phase": self.state.phase,
            "alive_count": len(self.state.alive_players),
            "level": self.level,
        }

    def _done_turn(self) -> dict[str, Any]:
        """Return a terminal turn."""
        assert self.state is not None
        return {
            "speaker_id": "SYSTEM",
            "speaker_name": "Moderator",
            "speaker_display": "[SYSTEM] Moderator",
            "role": "system",
            "message": (
                f"Game over. "
                f"{'The village wins!' if self.state.winner == 'village' else 'The Werewolves win!'}"
            ),
            "moderator_prompt": "",
            "day": self.state.day,
            "phase": "end",
            "round_idx": self.state.round_idx,
            "game_over": True,
            "winner": self.state.winner,
            "alive_players": [p.to_dict() for p in self.state.alive_players],
        }
