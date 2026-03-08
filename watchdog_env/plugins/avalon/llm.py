"""LLM infrastructure for Avalon & shared game-play model.

Supports two backends (configured via WATCHDOG_LLM_BACKEND env var):
  - "local"  (default): Qwen3 8B loaded via transformers + bitsandbytes 4-bit
  - "gemini": Google Gemini via langchain-google-genai (kept but not default)

The game-play model is frozen (inference only). For trainable mutation model
see watchdog_env.mutations.llm_backend.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any

from .avalon_models import GameState, Player

logger = logging.getLogger(__name__)


# ─── .env loader ────────────────────────────────────────────────────

def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        env_path = pathlib.Path(__file__).resolve().parents[3] / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
        else:
            load_dotenv(override=False)
    except ImportError:
        pass

_load_dotenv()


# ─── Unified chat response ──────────────────────────────────────────

class ChatResponse:
    """Minimal response with .content — compatible with LangChain interface."""
    def __init__(self, content: str):
        self.content = content


# ─── Local HuggingFace game-play model (Qwen3 8B, 4-bit, frozen) ────

_local_model_instance = None


class GamePlayModel:
    """Frozen local model for game play (Avalon / Cicero).

    Loads Qwen/Qwen3-8B in bf16 for fast inference on high-VRAM GPUs.
    Provides invoke() and invoke_batch() with the same interface as LangChain.
    """

    def __init__(self, model_name: str | None = None, temperature: float = 0.8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name or os.environ.get("LOCAL_MODEL_NAME", "Qwen/Qwen3-8B")
        self.temperature = temperature

        logger.info("Loading game-play model %s (bf16 + flash_attention_2)...", self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        logger.info("Game-play model loaded: %s", self.model_name)

    def _messages_to_prompt(self, messages) -> str:
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
            elif isinstance(m, dict):
                chat.append(m)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
            )
        return (
            "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in chat)
            + "\n<|im_start|>assistant\n"
        )

    def invoke(self, messages) -> ChatResponse:
        import torch
        prompt_text = self._messages_to_prompt(messages)
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
        return ChatResponse(text if text else "I have nothing to say.")

    def invoke_batch(self, messages_list: list) -> list[ChatResponse]:
        import torch
        if len(messages_list) == 1:
            return [self.invoke(messages_list[0])]
        prompt_texts = [self._messages_to_prompt(msgs) for msgs in messages_list]
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
        )
        self.tokenizer.padding_side = orig_padding_side
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_lengths = inputs["attention_mask"].sum(dim=1)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=0.9 if self.temperature > 0 else None,
            )
        results = []
        for i in range(len(messages_list)):
            gen_ids = output_ids[i][input_lengths[i]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(ChatResponse(text if text else "I have nothing to say."))
        return results


def get_game_play_model() -> GamePlayModel:
    """Singleton accessor for the shared game-play model."""
    global _local_model_instance
    if _local_model_instance is None:
        model_name = os.environ.get("LOCAL_MODEL_NAME", "Qwen/Qwen3-8B")
        temperature = float(os.environ.get("WATCHDOG_TEMPERATURE", "0.8"))
        _local_model_instance = GamePlayModel(model_name, temperature)
    return _local_model_instance


# ─── Gemini backend (kept as option, not default) ───────────────────

def _get_gemini_llm():
    """Return Gemini ChatModel via langchain-google-genai. Requires API key."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=float(os.environ.get("WATCHDOG_TEMPERATURE", "0.8")),
        google_api_key=api_key,
    )


# ─── Unified LLM accessor ──────────────────────────────────────────

_llm_instance = None


def _get_llm():
    """Get the configured LLM backend. Default: local Qwen3 8B."""
    backend = os.environ.get("WATCHDOG_LLM_BACKEND", "local").lower()
    if backend == "gemini":
        llm = _get_gemini_llm()
        if llm is not None:
            return llm
        logger.warning("Gemini requested but no API key found. Falling back to local model.")
    return get_game_play_model()


def _llm():
    """Singleton LLM accessor."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = _get_llm()
    return _llm_instance


# ─── Player response generation ─────────────────────────────────────

def _build_player_system_prompt(
    player: Player, game: GameState, alive_list: str, wolf_names: str,
) -> str:
    """Build system prompt for a player in Werewolf."""
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

    recent = game.conversation_log[-8:] if game.conversation_log else []
    convo_ctx = "\n".join(
        f"  {entry['speaker_display']}: {entry['message']}"
        for entry in recent
    ) or "(no prior conversation)"

    return (
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


def _generate_player_response_llm(
    player: Player,
    game: GameState,
    moderator_prompt: str,
) -> str:
    """Generate a single player's response using the configured LLM backend."""
    llm = _llm()

    wolf_names = ", ".join(f"{w.display}" for w in game.alive_wolves)
    alive_list = ", ".join(f"{p.display} ({p.role})" for p in game.alive_players)

    sys_prompt = _build_player_system_prompt(player, game, alive_list, wolf_names)

    # Use dict messages — works with both local GamePlayModel and LangChain
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": moderator_prompt},
    ]
    response = llm.invoke(messages)
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
            f"LLM returned empty response for {player.display}."
        )
    return text
