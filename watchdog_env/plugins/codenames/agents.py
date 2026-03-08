"""Agent implementations for Codenames players.

Four distinct agents: red_spymaster, red_operative, blue_spymaster, blue_operative.
Each agent has role-specific prompts and visibility into the game state.

Uses shared local Qwen3 8B game-play model from avalon/llm.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

from watchdog_env.plugins.codenames.game_state import CodenamesGameState, ClueRecord

logger = logging.getLogger(__name__)

AgentRole = Literal["red_spymaster", "red_operative", "blue_spymaster", "blue_operative"]


class AgentActionError(Exception):
    """Raised when agent fails to produce a valid action."""
    pass


@dataclass
class ClueAction:
    """A clue action from a spymaster."""
    clue_word: str
    clue_number: int
    reasoning: str = ""


@dataclass
class GuessAction:
    """A guess action from an operative."""
    guessed_word: str
    reasoning: str = ""
    pass_turn: bool = False


def _get_llm():
    """Get Gemini if API key present, otherwise local model."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=float(os.environ.get("WATCHDOG_TEMPERATURE", "0.8")),
            google_api_key=api_key,
        )
    from watchdog_env.plugins.avalon.llm import get_game_play_model
    return get_game_play_model()


def _format_clue_history(clue_history: list[ClueRecord]) -> str:
    """Format clue history for prompts."""
    if not clue_history:
        return "No clues given yet."
    
    lines = []
    for clue in clue_history:
        lines.append(f"  - {clue.team.upper()}: \"{clue.clue_word}\" {clue.clue_number}")
    return "\n".join(lines)


def _format_guess_history(guess_history: list, current_turn: int | None = None) -> str:
    """Format guess history for prompts."""
    if not guess_history:
        return "No guesses made yet."
    
    lines = []
    for guess in guess_history:
        if current_turn is not None and guess.turn_number != current_turn:
            continue
        result = "CORRECT" if guess.correct else f"WRONG ({guess.actual_type})"
        lines.append(f"  - {guess.team.upper()} guessed \"{guess.guessed_word}\": {result}")
    
    return "\n".join(lines) if lines else "No guesses this turn."


def _build_spymaster_prompt(state: CodenamesGameState, team: str) -> str:
    """Build the prompt for a spymaster agent."""
    board_view = state.get_board_for_spymaster()
    
    team_words = state.get_remaining_words(team)
    opponent_team = "blue" if team == "red" else "red"
    opponent_words = state.get_remaining_words(opponent_team)
    assassin_word = [w for w, t in state.board.assignments.items() if t == "assassin"][0]
    neutral_words = [w for w, t in state.board.assignments.items() if t == "neutral" and w not in state.revealed_words]
    
    # Get word interactions info if available
    interactions_info = ""
    if state.board.interactions and state.board.interactions.polysemes:
        polysemes_on_board = [w for w in state.board.interactions.polysemes if w in state.board.words]
        if polysemes_on_board:
            interactions_info = f"\nWords with multiple meanings (be careful!): {', '.join(polysemes_on_board)}"
    
    grid_str = "\n".join([" | ".join(row) for row in state.board.grid])
    
    return f"""You are the {team.upper()} SPYMASTER in a game of Codenames.

BOARD (5x5 grid):
{grid_str}

YOUR TEAM'S WORDS (to help your operative find): {', '.join(team_words)}
OPPONENT'S WORDS (AVOID): {', '.join(opponent_words)}
NEUTRAL WORDS: {', '.join(neutral_words)}
ASSASSIN (MUST AVOID - instant loss): {assassin_word}
{interactions_info}

REVEALED WORDS: {', '.join(state.revealed_words) if state.revealed_words else 'None'}

CLUE HISTORY:
{_format_clue_history(state.clue_history)}

RULES:
1. Give a ONE-WORD clue and a NUMBER indicating how many words on the board relate to it
2. Your clue CANNOT be any word on the board (or a form of it)
3. Try to connect MULTIPLE of your team's words with one clue
4. Avoid clues that might lead to opponent words or the ASSASSIN
5. The number should reflect how many words you intend to hint at

Think strategically about word associations. Consider:
- Which words can be connected by a single clue?
- What clues might accidentally point to opponent or assassin words?
- Are there any "trap" words with multiple meanings?

Respond with ONLY a JSON object in this exact format:
{{"clue": "YOUR_CLUE_WORD", "number": N, "reasoning": "Brief explanation of your strategy"}}

The clue must be a single word, all uppercase. The number must be 1 or higher."""


def _build_operative_prompt(state: CodenamesGameState, team: str) -> str:
    """Build the prompt for an operative agent."""
    board_view = state.get_board_for_operative()
    
    current_clue = state.current_clue
    if not current_clue:
        raise AgentActionError("No clue available for operative")
    
    # Get unrevealed words
    unrevealed = [w for w in state.board.words if w not in state.revealed_words]
    
    grid_str = "\n".join([" | ".join(row) for row in state.board.grid])
    
    # Mark revealed words in display
    revealed_info = state.get_revealed_words_by_type()
    
    max_guesses = current_clue.clue_number + 1
    remaining_guesses = max_guesses - state.guesses_this_turn
    
    guesses_this_turn = _format_guess_history(state.guess_history, state.turn_number)
    
    return f"""You are the {team.upper()} OPERATIVE in a game of Codenames.

BOARD (5x5 grid):
{grid_str}

REVEALED WORDS:
  - Red team: {', '.join(revealed_info['red']) if revealed_info['red'] else 'None'}
  - Blue team: {', '.join(revealed_info['blue']) if revealed_info['blue'] else 'None'}
  - Neutral: {', '.join(revealed_info['neutral']) if revealed_info['neutral'] else 'None'}
  - Assassin: {', '.join(revealed_info['assassin']) if revealed_info['assassin'] else 'Not revealed'}

UNREVEALED WORDS: {', '.join(unrevealed)}

CURRENT CLUE: "{current_clue.clue_word}" {current_clue.clue_number}

GUESSES THIS TURN:
{guesses_this_turn}

GUESSES REMAINING: {remaining_guesses} (clue number + 1 bonus guess)

CLUE HISTORY:
{_format_clue_history(state.clue_history)}

RULES:
1. Guess ONE unrevealed word that you think relates to the clue "{current_clue.clue_word}"
2. You can guess up to {current_clue.clue_number} + 1 words (the +1 is a bonus)
3. If you guess wrong or hit neutral, your turn ends immediately
4. If you hit the ASSASSIN, your team LOSES the game
5. You can PASS to end your turn safely (especially if unsure)

Think about:
- What words on the board relate to "{current_clue.clue_word}"?
- Consider the number {current_clue.clue_number} - that many words should relate to the clue
- Avoid words that might be dangerous (opponent or assassin)
- Look at previous clues for context

Respond with ONLY a JSON object in this exact format:
{{"guess": "WORD_TO_GUESS", "reasoning": "Why this word relates to the clue"}}

OR to pass your turn:
{{"pass": true, "reasoning": "Why you're choosing to pass"}}

The word must be from the unrevealed words list, all uppercase."""


def _parse_clue_response(response_text: str) -> ClueAction:
    """Parse spymaster's clue response.
    
    Raises:
        AgentActionError: If parsing fails
    """
    try:
        text = response_text.strip()
        # Handle markdown code blocks
        if "```" in text:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)
        
        # Try to find JSON object
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        
        data = json.loads(text)
        clue_word = str(data.get("clue", "")).upper().strip()
        clue_number = int(data.get("number", 1))
        reasoning = str(data.get("reasoning", ""))
        
        if not clue_word:
            raise AgentActionError("Empty clue word in response")
        if clue_number < 1:
            raise AgentActionError(f"Invalid clue number: {clue_number}")
        
        return ClueAction(clue_word=clue_word, clue_number=clue_number, reasoning=reasoning)
    except json.JSONDecodeError as e:
        raise AgentActionError(f"Failed to parse clue response as JSON: {e}") from e
    except (KeyError, ValueError, TypeError) as e:
        raise AgentActionError(f"Invalid clue response format: {e}") from e


def _parse_guess_response(response_text: str) -> GuessAction:
    """Parse operative's guess response.
    
    Raises:
        AgentActionError: If parsing fails
    """
    try:
        text = response_text.strip()
        # Handle markdown code blocks
        if "```" in text:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)
        
        # Try to find JSON object
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        
        data = json.loads(text)
        
        # Check for pass
        if data.get("pass", False):
            return GuessAction(guessed_word="", reasoning=str(data.get("reasoning", "")), pass_turn=True)
        
        guessed_word = str(data.get("guess", "")).upper().strip()
        reasoning = str(data.get("reasoning", ""))
        
        if not guessed_word:
            raise AgentActionError("Empty guess word in response")
        
        return GuessAction(guessed_word=guessed_word, reasoning=reasoning, pass_turn=False)
    except json.JSONDecodeError as e:
        raise AgentActionError(f"Failed to parse guess response as JSON: {e}") from e
    except (KeyError, ValueError, TypeError) as e:
        raise AgentActionError(f"Invalid guess response format: {e}") from e


class CodenamesAgent:
    """Agent that plays a specific role in Codenames.
    
    Supports local (Qwen3 8B) or Gemini backends.
    """
    
    def __init__(self, role: AgentRole):
        self.role = role
        self.team = "red" if role.startswith("red") else "blue"
        self.is_spymaster = "spymaster" in role
    
    def get_action(self, state: CodenamesGameState) -> ClueAction | GuessAction:
        """Get the agent's action based on current game state.
        
        Raises:
            AgentActionError: If agent fails to produce a valid action
        """
        llm = _get_llm()
        
        if self.is_spymaster:
            return self._get_clue_action(state, llm)
        else:
            return self._get_guess_action(state, llm)
    
    def _get_clue_action(self, state: CodenamesGameState, llm) -> ClueAction:
        """Get a clue action from the spymaster.
        
        Raises:
            AgentActionError: If clue generation fails
        """
        prompt = _build_spymaster_prompt(state, self.team)
        
        system_content = (
            "You are playing Codenames as a Spymaster. "
            "Your goal is to help your teammate find your team's words "
            "while avoiding the opponent's words and the assassin. "
            "Respond only with the requested JSON format."
        )
        
        # Use dict messages — works with both local GamePlayModel and LangChain
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                response_text = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            else:
                response_text = str(content)
            
            if not response_text.strip():
                raise AgentActionError("LLM returned empty response for clue")
            
            action = _parse_clue_response(response_text)
            
            # Validate clue is not a board word
            board_words = set(w.upper() for w in state.board.words)
            if action.clue_word in board_words:
                raise AgentActionError(f"Clue '{action.clue_word}' is on the board - invalid clue")
            
            return action
            
        except AgentActionError:
            raise
        except Exception as e:
            raise AgentActionError(f"LLM clue generation failed: {e}") from e
    
    def _get_guess_action(self, state: CodenamesGameState, llm) -> GuessAction:
        """Get a guess action from the operative.
        
        Raises:
            AgentActionError: If guess generation fails
        """
        prompt = _build_operative_prompt(state, self.team)
        
        system_content = (
            "You are playing Codenames as an Operative. "
            "Your goal is to guess your team's words based on clues from your Spymaster. "
            "Avoid the assassin word at all costs. "
            "Respond only with the requested JSON format."
        )
        
        # Use dict messages — works with both local GamePlayModel and LangChain
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                response_text = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            else:
                response_text = str(content)
            
            if not response_text.strip():
                raise AgentActionError("LLM returned empty response for guess")
            
            action = _parse_guess_response(response_text)
            
            # Validate guess is on the board and unrevealed (unless passing)
            if not action.pass_turn:
                unrevealed = set(w.upper() for w in state.board.words if w not in state.revealed_words)
                if action.guessed_word not in unrevealed:
                    raise AgentActionError(f"Invalid guess '{action.guessed_word}' - not an unrevealed word")
            
            return action
            
        except AgentActionError:
            raise
        except Exception as e:
            raise AgentActionError(f"LLM guess generation failed: {e}") from e


def create_agents() -> dict[str, CodenamesAgent]:
    """Create all four agents for a Codenames game."""
    roles: list[AgentRole] = ["red_spymaster", "red_operative", "blue_spymaster", "blue_operative"]
    return {role: CodenamesAgent(role) for role in roles}
