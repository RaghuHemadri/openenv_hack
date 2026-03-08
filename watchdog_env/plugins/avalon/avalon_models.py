"""Core Avalon/Werewolf data models: Player, GameState, game creation helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# ─── Player & Game State ────────────────────────────────────────────

ROLES = ("Werewolf", "Villager", "Police", "Doctor")

PLAYER_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Erik",
    "Fiona", "George", "Hannah", "Ivan", "Julia",
    "Karl", "Luna", "Marcus", "Nora", "Oscar",
]


@dataclass
class Player:
    """A single player in the Werewolf game."""
    player_id: str          # "P1", "P2", …
    name: str               # Human-readable name
    role: str               # Werewolf | Villager | Police | Doctor
    alive: bool = True

    @property
    def display(self) -> str:
        return f"[{self.player_id}] {self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "role": self.role,
            "alive": self.alive,
        }


@dataclass
class GameState:
    """Full mutable state of one Werewolf game."""
    players: list[Player]
    day: int = 1
    phase: str = "day"          # "day" | "night"
    round_idx: int = 0          # which turn within the current day
    eliminated: list[str] = field(default_factory=list)  # player_ids
    night_kill: str | None = None       # player_id killed at night
    doctor_save: str | None = None      # player_id saved by doctor
    police_result: dict | None = None   # {"target": id, "role": str}
    conversation_log: list[dict[str, Any]] = field(default_factory=list)

    # ── Convenience helpers ─────────────────────────────────────
    @property
    def alive_players(self) -> list[Player]:
        return [p for p in self.players if p.alive]

    @property
    def alive_wolves(self) -> list[Player]:
        return [p for p in self.players if p.alive and p.role == "Werewolf"]

    @property
    def alive_villager_side(self) -> list[Player]:
        return [p for p in self.players if p.alive and p.role != "Werewolf"]

    def get_player(self, player_id: str) -> Player | None:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None

    @property
    def game_over(self) -> bool:
        wolves = len(self.alive_wolves)
        village = len(self.alive_villager_side)
        return wolves == 0 or wolves >= village

    @property
    def winner(self) -> str | None:
        if not self.game_over:
            return None
        return "village" if len(self.alive_wolves) == 0 else "werewolves"

    def to_dict(self) -> dict[str, Any]:
        return {
            "players": [p.to_dict() for p in self.players],
            "day": self.day,
            "phase": self.phase,
            "round_idx": self.round_idx,
            "eliminated": self.eliminated,
            "conversation_log": self.conversation_log,
            "alive_count": len(self.alive_players),
            "game_over": self.game_over,
            "winner": self.winner,
        }


# ─── Game Setup ─────────────────────────────────────────────────────

GAME_SETUPS: dict[int, list[dict[str, int]]] = {
    # level → [{wolves, villagers}]  (Police=1, Doctor=1 always)
    1: [{"wolves": 1, "villagers": 3}, {"wolves": 1, "villagers": 4}],
    2: [{"wolves": 2, "villagers": 3}, {"wolves": 2, "villagers": 4}],
    3: [{"wolves": 2, "villagers": 4}, {"wolves": 3, "villagers": 4}],
    4: [{"wolves": 3, "villagers": 4}, {"wolves": 3, "villagers": 5}],
}


def create_game(level: int = 1, seed: int | None = None) -> GameState:
    """Create a fresh Werewolf game with randomised role assignment."""
    if seed is not None:
        random.seed(seed)

    setups = GAME_SETUPS.get(level, GAME_SETUPS[2])
    setup = random.choice(setups)
    n_wolves = setup["wolves"]
    n_villagers = setup["villagers"]
    total = n_wolves + n_villagers + 2  # + Police + Doctor

    names = random.sample(PLAYER_NAMES, total)
    roles: list[str] = (
        ["Werewolf"] * n_wolves
        + ["Police"]
        + ["Doctor"]
        + ["Villager"] * n_villagers
    )
    random.shuffle(roles)

    players = [
        Player(player_id=f"P{i+1}", name=names[i], role=roles[i])
        for i in range(total)
    ]
    return GameState(players=players)


# ─── Day Phase Prompts ──────────────────────────────────────────────

_DAY_OPENERS = [
    "Day {day} begins. {event} The moderator calls on {speaker_disp} to speak.",
    "The village gathers for day {day}. {event} {speaker_disp}, what do you have to say?",
    "Day {day} discussion. {event} It's {speaker_disp}'s turn.",
]

_DAY_EVENTS = [
    "{victim} was found dead this morning — the Werewolves struck.",
    "Nobody died last night — the Doctor must have saved someone!",
    "Suspicion is rising after yesterday's close vote.",
    "Tensions mount after a mislynch yesterday.",
    "A new day begins. The village must find the Werewolves before it's too late.",
]

_DAY_EVENTS_NO_DEATH = [
    "Nobody died last night — the Doctor must have saved someone!",
    "The village survived the night. Discussion begins.",
    "A calm night. Time to deliberate.",
]
