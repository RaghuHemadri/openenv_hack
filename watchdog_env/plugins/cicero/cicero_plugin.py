"""Cicero multi-agent plugin: Diplomacy-style negotiation via LangChain + Gemini.

Implements all MultiAgentSystemPlugin methods. generate_step is based on state history
(turns_so_far) for LLM context. Requires GEMINI_API_KEY or GOOGLE_API_KEY for live runs.
"""

from __future__ import annotations

import os
import random
from typing import Any

from watchdog_env.models import AgentTurn, MultiAgentConfig, MultiAgentState, MultiAgentStep
from watchdog_env.plugins.base import (
    MultiAgentSystemPlugin,
    append_to_conversation_log,
    get_conversation_log,
)
from watchdog_env.plugins.cicero.cicero_config import CICERO_POWERS, CiceroConfig


def _get_langchain_llm():
    """Return ChatGoogleGenerativeAI. Raises if API key missing or langchain unavailable."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Cicero plugin requires GEMINI_API_KEY or GOOGLE_API_KEY. No template fallback."
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.85,
        google_api_key=api_key,
    )


def _format_conversation_log(entries: list[dict]) -> str:
    """Format conversation_log entries as transcript for LLM context."""
    return "\n".join(
        f"{e.get('speaker_display', e.get('speaker_id', '?'))}: {e.get('message', '')}"
        for e in entries
    )


# Diplomacy game context (from diplomacy_cicero: 1914 Europe, seven powers, Supply Centers)
_DIPLOMACY_CONTEXT = """
Diplomacy is a strategic board game set in 1914 Europe. The board has 56 land regions and 19 sea regions.
Seven Great Powers: Austria-Hungary, England, France, Germany, Italy, Russia, Turkey.
Each power controls regions and units. Supply Centers (SCs) determine unit count—more SCs means more units.
Goal: control more than half of all SCs by moving units and convincing other players to support you.
Key regions: Vienna, Warsaw, Moscow, Constantinople, London, Paris, Berlin, Rome, Serbia, Bulgaria, Galicia, Ukraine.
Seasons: Spring 1901, Fall 1901, Spring 1902, etc. Orders are written each season; negotiation happens before orders.
"""

_NEGOTIATION_DOMAINS = [
    ("alliance_negotiation", "proposing or renewing an alliance for mutual support"),
    ("move_coordination", "coordinating army/fleet moves for the coming season"),
    ("supply_center_deal", "negotiating who gets which Supply Center or territory"),
    ("support_request", "asking for support on a specific move or defense"),
    ("threat_assessment", "discussing a rival power's moves or a potential stab"),
]


class CiceroPlugin(MultiAgentSystemPlugin):
    """Multi-agent Diplomacy (Cicero) plugin. All methods implemented."""

    def __init__(self) -> None:
        self._state = MultiAgentState()

    def get_game_id(self) -> str:
        return "cicero"

    def get_display_name(self) -> str:
        return "Cicero (Diplomacy negotiation)"

    def list_agent_ids(self) -> list[str]:
        return list(CICERO_POWERS)

    def reset(
        self,
        seed: int | None = None,
        config: MultiAgentConfig | None = None,
    ) -> None:
        if seed is not None:
            random.seed(seed)
        cfg = config if isinstance(config, CiceroConfig) else CiceroConfig()
        self._state = MultiAgentState(
            step_index=0,
            turns_so_far=[],
            config=cfg,
            done=False,
            conversation_log=[],
        )

    def get_state(self) -> MultiAgentState:
        return self._state

    def generate_step(self, seed: int | None, step_index: int) -> MultiAgentStep:
        if seed is not None:
            random.seed(seed)
        cfg = self._state.config
        if isinstance(cfg, CiceroConfig):
            num_steps = cfg.num_steps
            powers = cfg.get_powers()
            model_name = cfg.model_name
            temperature = cfg.temperature
        else:
            num_steps = 3
            powers = list(CICERO_POWERS)
            model_name = "gemini-2.5-flash"
            temperature = 0.85

        done = step_index >= num_steps - 1
        llm = _get_langchain_llm()

        # Use conversation_log for context
        transcript_so_far = _format_conversation_log(get_conversation_log(self._state))
        regions = [
            "Vienna", "Warsaw", "Constantinople", "London", "Paris", "Berlin", "Rome",
            "Serbia", "Bulgaria", "Galicia", "Ukraine", "North Sea", "Mediterranean",
        ]
        seasons = ["Spring 1901", "Fall 1901", "Spring 1902", "Fall 1902", "Spring 1903"]
        region = random.choice(regions)
        season = random.choice(seasons)
        domain_name, domain_desc = random.choice(_NEGOTIATION_DOMAINS)

        turns: list[AgentTurn] = []
        participating = powers[: min(2, len(powers))] if step_index == 0 else powers[:2]
        for i, power in enumerate(participating):
            other = participating[(i + 1) % len(participating)]
            system = (
                f"You are {power} in a Diplomacy game (1914 Europe). "
                f"You are in a private, open-domain negotiation with {other}. "
                f"{_DIPLOMACY_CONTEXT.strip()}\n\n"
                "Stay in character as that power. Use natural diplomatic language: propose alliances, "
                "coordinate moves, discuss Supply Centers, or respond to offers. "
                "Keep your message 1–4 sentences. Output only your message, no prefix or quotes."
            )
            if transcript_so_far:
                user = (
                    f"Season: {season}. Region of interest: {region}. "
                    f"Topic: {domain_desc}.\n\n"
                    f"Conversation so far:\n{transcript_so_far}\n\n"
                    f"You are {power}. Reply to {other} in character. Output only your message."
                )
            else:
                user = (
                    f"Season: {season}. Region of interest: {region}. "
                    f"Topic: {domain_desc}.\n\n"
                    f"You are {power} opening the conversation with {other}. "
                    f"Send your first message (proposal, offer, or diplomatic overture). Output only your message."
                )

            from langchain_core.messages import HumanMessage, SystemMessage
            langchain_messages = [SystemMessage(content=system), HumanMessage(content=user)]

            response = llm.invoke(langchain_messages)
            text = response.content if hasattr(response, "content") else str(response)
            if not (text and isinstance(text, str)):
                raise RuntimeError(
                    f"Cicero plugin: LLM returned empty response for {power}. No fallback."
                )
            text = text.strip()

            append_to_conversation_log(
                self._state,
                speaker_id=power,
                speaker_display=power,
                message=text,
            )
            turns.append(
                AgentTurn(
                    agent_id=power,
                    action_text=text,
                    step_index=step_index,
                    display_name=power,
                    metadata={
                        "season": season,
                        "region": region,
                        "domain_name": domain_name,
                        "domain_desc": domain_desc,
                        "counterpart": other,
                    },
                )
            )
            transcript_so_far = _format_conversation_log(get_conversation_log(self._state))

        self._state.step_index = step_index + 1
        self._state.turns_so_far.extend(turns)
        self._state.done = done

        return MultiAgentStep(
            turns=turns,
            done=done,
            step_index=step_index,
            game_id=self.get_game_id(),
            state=MultiAgentState(
                step_index=self._state.step_index,
                turns_so_far=list(self._state.turns_so_far),
                config=self._state.config,
                done=self._state.done,
                conversation_log=list(self._state.conversation_log),
            ),
        )
