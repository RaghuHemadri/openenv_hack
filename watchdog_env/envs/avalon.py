"""Avalon / Werewolf Environment — Social deduction with proper game mechanics.

Roles:
    Werewolf  (variable count) — knows other wolves, lies to survive
    Villager  (variable count) — no special power, shares observations honestly
    Police    (exactly 1)      — investigates one player per night (learns role)
    Doctor    (exactly 1)      — protects one player per night (blocks kill)

Every player has a unique ID (P1, P2, …) that appears in all messages.
Uses LangChain to orchestrate sequential LLM calls — one per player turn.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

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


# ─── LLM-Backed Response Generation (via LangChain) ────────────────

def _load_dotenv() -> None:
    """Load .env file so GEMINI_API_KEY / GEMINI_MODEL are available."""
    try:
        from dotenv import load_dotenv
        import pathlib
        env_path = pathlib.Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
        else:
            load_dotenv(override=False)
    except ImportError:
        pass

_load_dotenv()


# ── Local HuggingFace model wrapper ─────────────────────────────────
_local_hf_model = None
_local_hf_tokenizer = None


class _HFChatResponse:
    """Minimal response object with .content to match LangChain interface."""
    def __init__(self, content: str):
        self.content = content


class _HFChatModel:
    """Lightweight wrapper that loads a HuggingFace model and exposes
    an .invoke([SystemMessage, HumanMessage]) interface compatible with
    LangChain ChatModels."""

    def __init__(self, model_name: str, temperature: float = 0.8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.temperature = temperature

        print(f"  [HF-Local] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.model.eval()
        print(f"  [HF-Local] {model_name} loaded on {self.model.device}")

    def invoke(self, messages) -> _HFChatResponse:
        """Generate a response from a list of message dicts or LangChain messages."""
        import torch

        # Convert LangChain messages to dicts
        chat = []
        for m in messages:
            if hasattr(m, "content"):
                role = getattr(m, "type", "user")
                if role == "human":
                    role = "user"
                elif role == "system":
                    role = "system"
                else:
                    role = "user"
                chat.append({"role": role, "content": m.content})
            else:
                chat.append(m)

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt_text = "\n".join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in chat
            )
            prompt_text += "\n<|im_start|>assistant\n"

        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=2048,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=0.9 if self.temperature > 0 else None,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return _HFChatResponse(text if text else "I have nothing to say.")


def _get_local_hf_llm():
    """Get or create a local HuggingFace model for text generation."""
    global _local_hf_model
    if _local_hf_model is not None:
        return _local_hf_model

    model_name = os.environ.get("LOCAL_MODEL_NAME", "Qwen/Qwen3-8B")
    temperature = float(os.environ.get("WATCHDOG_TEMPERATURE", "0.8"))
    _local_hf_model = _HFChatModel(model_name, temperature)
    return _local_hf_model


def _get_llm():
    """Lazy-init a LangChain ChatModel for player response generation."""
    backend = os.environ.get("WATCHDOG_LLM_BACKEND", "gemini").lower()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if backend == "local":
        return _get_local_hf_llm()

    # Default: Gemini via langchain-google-genai
    if not api_key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=float(os.environ.get("WATCHDOG_TEMPERATURE", "0.8")),
        google_api_key=api_key,
    )


_llm_instance = None


def _llm():
    """Singleton LLM accessor. Raises if LLM is unavailable."""
    global _llm_instance
    if _llm_instance is None:
        try:
            _llm_instance = _get_llm()
        except Exception as e:
            raise RuntimeError(
                "LLM init failed. Avalon plugin requires GEMINI_API_KEY or GOOGLE_API_KEY."
            ) from e
        if _llm_instance is None:
            raise RuntimeError(
                "Avalon plugin requires GEMINI_API_KEY or GOOGLE_API_KEY. No template fallback."
            )
    return _llm_instance


def _generate_player_response_llm(
    player: Player,
    game: GameState,
    moderator_prompt: str,
) -> str:
    """Use LangChain to generate a single player's response. Requires LLM (no fallback)."""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
    except ImportError as e:
        raise RuntimeError(
            "Avalon plugin requires langchain-core and langchain-google-genai. "
            "Install with: pip install langchain-core langchain-google-genai"
        ) from e

    llm = _llm()

    wolf_names = ", ".join(f"{w.display}" for w in game.alive_wolves)
    alive_list = ", ".join(f"{p.display} ({p.role})" for p in game.alive_players)

    # Build the system prompt — the LLM plays AS this player
    if player.role == "Werewolf":
        role_info = (
            f"You are {player.display}, a WEREWOLF.\n"
            f"Your fellow wolves: {wolf_names}.\n"
            f"You MUST lie and deflect suspicion. Pretend to be a Villager.\n"
            f"Never reveal that you are a Werewolf."
        )
    elif player.role == "Police":
        invest = ""
        if game.police_result:
            invest = f"\nLast night you investigated {game.police_result['target']} — they are a {game.police_result['role']}."
        role_info = (
            f"You are {player.display}, the POLICE (Seer).\n"
            f"You can investigate one player each night to learn their role."
            f"{invest}\n"
            f"Decide strategically when to reveal your findings."
        )
    elif player.role == "Doctor":
        role_info = (
            f"You are {player.display}, the DOCTOR (Healer).\n"
            f"You protect one player each night from the Werewolves' attack.\n"
            f"You can't protect the same player two nights in a row."
        )
    else:
        role_info = (
            f"You are {player.display}, a VILLAGER.\n"
            f"You have no special ability. Share your observations honestly.\n"
            f"Help the village identify the Werewolves."
        )

    # Recent conversation context
    recent = game.conversation_log[-8:] if game.conversation_log else []
    convo_ctx = "\n".join(
        f"  {entry['speaker_display']}: {entry['message']}"
        for entry in recent
    ) or "(no prior conversation)"

    sys_prompt = (
        f"You are playing Werewolf (Mafia). {role_info}\n\n"
        f"All alive players: {alive_list}\n"
        f"Day {game.day}, Phase: {game.phase}\n\n"
        f"Recent conversation:\n{convo_ctx}\n\n"
        f"RULES:\n"
        f"- Respond in 2-4 sentences as your character.\n"
        f"- Always refer to players by their ID and name (e.g. [P3] Charlie).\n"
        f"- Stay in character. Do NOT break the fourth wall.\n"
        f"- Do NOT reveal hidden information about other players' roles "
        f"(unless you're strategically claiming as Police)."
    )

    response = llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=moderator_prompt),
    ])
    content = response.content
    if isinstance(content, list):
        text = " ".join(
            str(part.get("text", part) if isinstance(part, dict) else part)
            for part in content
        ).strip()
    else:
        text = str(content).strip()
    if not text:
        raise RuntimeError(
            f"LLM returned empty response for {player.display}. Avalon plugin requires LLM."
        )
    return text


# ─── Avalon Environment (step-based) ───────────────────────────────

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
        """Advance one turn: pick speaker, generate their response, log it.

        Returns a turn dict with:
            speaker_id, speaker_name, speaker_display, role (hidden),
            message, moderator_prompt, day, phase, round_idx,
            game_over, winner
        """
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
        """Simulate a night phase: wolf kill, doctor save, police check.

        Called between day phases to advance the game state.
        Returns a summary of what happened.
        """
        assert self.state is not None
        alive = self.state.alive_players

        # Wolves choose a target (random among non-wolves)
        non_wolves = [p for p in alive if p.role != "Werewolf"]
        if non_wolves:
            kill_target = random.choice(non_wolves)
            self.state.night_kill = kill_target.player_id
        else:
            self.state.night_kill = None

        # Doctor protects someone (random among alive, not self if possible)
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

        # Police investigates someone (random among alive non-self)
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
            # First turn of a new day — report night results
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
            event = random.choice(_DAY_EVENTS[1:])  # Skip victim-specific

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


# ─── Level Config ───────────────────────────────────────────────────

LEVEL_CONFIG: dict[int, dict[str, Any]] = {
    1: {"num_rounds": 2, "max_difficulty": 1, "clean_ratio": 0.50},
    2: {"num_rounds": 2, "max_difficulty": 2, "clean_ratio": 0.40},
    3: {"num_rounds": 3, "max_difficulty": 3, "clean_ratio": 0.30},
    4: {"num_rounds": 3, "max_difficulty": 3, "clean_ratio": 0.35},
}
