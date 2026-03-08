"""Codenames multi-agent plugin: 4-player word guessing game.

Implements MultiAgentSystemPlugin interface for Codenames board game.

Supports two backends (configured via WATCHDOG_LLM_BACKEND env var):
  - "local"  (default): shared Qwen3 8B game-play model from avalon/llm.py
  - "gemini": Google Gemini via langchain-google-genai (requires API key)
"""

from __future__ import annotations

import logging
import random
from typing import Any

from watchdog_env.models import AgentTurn, MultiAgentConfig, MultiAgentState, MultiAgentStep
from watchdog_env.plugins.base import (
    MultiAgentSystemPlugin,
    append_to_conversation_log,
    get_conversation_log,
)
from watchdog_env.plugins.codenames.codenames_config import CODENAMES_AGENTS, CodenamesConfig
from watchdog_env.plugins.codenames.board_generator import (
    generate_board,
    BoardAssignment,
    BoardGenerationError,
)
from watchdog_env.plugins.codenames.game_state import CodenamesGameState
from watchdog_env.plugins.codenames.agents import (
    CodenamesAgent,
    create_agents,
    ClueAction,
    GuessAction,
    AgentActionError,
)

logger = logging.getLogger(__name__)


def _get_agent_display_name(agent_id: str) -> str:
    """Get display name for an agent ID."""
    display_names = {
        "red_spymaster": "Red Spymaster",
        "red_operative": "Red Operative",
        "blue_spymaster": "Blue Spymaster",
        "blue_operative": "Blue Operative",
    }
    return display_names.get(agent_id, agent_id)


class CodenamesPlugin(MultiAgentSystemPlugin):
    """Multi-agent Codenames plugin with 4 players (2 teams).
    
    Game flow:
    1. Red Spymaster gives clue
    2. Red Operative guesses (can make multiple guesses)
    3. Blue Spymaster gives clue
    4. Blue Operative guesses
    5. Repeat until one team wins or max turns reached
    
    Win conditions:
    - Find all your team's words
    - Opponent hits the assassin
    
    Lose conditions:
    - Hit the assassin
    - Opponent finds all their words first
    """

    def __init__(self) -> None:
        self._state = MultiAgentState()
        self._game_state: CodenamesGameState | None = None
        self._agents: dict[str, CodenamesAgent] = {}
        self._config: CodenamesConfig | None = None

    def get_game_id(self) -> str:
        return "codenames"

    def get_display_name(self) -> str:
        return "Codenames (4-player word game)"

    def list_agent_ids(self) -> list[str]:
        return list(CODENAMES_AGENTS)

    def reset(
        self,
        seed: int | None = None,
        config: MultiAgentConfig | None = None,
    ) -> None:
        """Initialize a new game with the given seed and config."""
        if seed is not None:
            random.seed(seed)
        
        # Parse config
        cfg = config if isinstance(config, CodenamesConfig) else CodenamesConfig()
        cfg.validate()
        self._config = cfg
        
        # Generate board (uses WATCHDOG_LLM_BACKEND for LLM selection)
        board = generate_board(
            seed=seed,
            complexity_level=cfg.complexity_level,
            red_words=cfg.red_words,
            blue_words=cfg.blue_words,
            neutral_words=cfg.neutral_words,
            assassin_words=cfg.assassin_words,
        )
        
        # Initialize game state
        self._game_state = CodenamesGameState(
            board=board,
            current_team=cfg.starting_team,
            current_phase="clue",
            max_turns=cfg.max_turns,
        )
        
        # Create agents (uses WATCHDOG_LLM_BACKEND for LLM selection)
        self._agents = create_agents()
        
        # Initialize plugin state with conversation_log (matching Cicero pattern)
        self._state = MultiAgentState(
            step_index=0,
            turns_so_far=[],
            config=cfg,
            done=False,
            conversation_log=[],
            metadata={
                "game_id": "codenames",
                "board_words": board.words,
                "starting_team": cfg.starting_team,
            },
        )

    def get_state(self) -> MultiAgentState:
        return self._state

    def get_game_state(self) -> CodenamesGameState | None:
        """Get the internal Codenames game state (for testing/debugging)."""
        return self._game_state

    def generate_step(self, seed: int | None, step_index: int) -> MultiAgentStep:
        """Generate one step of the game.
        
        Each step is one agent's action:
        - Spymaster giving a clue
        - Operative making a guess (or passing)
        
        Multiple guess steps may occur in sequence for the same operative.
        """
        if seed is not None:
            random.seed(seed)
        
        if self._game_state is None or self._config is None:
            # Return empty step if not initialized
            return MultiAgentStep(
                turns=[],
                done=True,
                step_index=step_index,
                game_id=self.get_game_id(),
            )
        
        game = self._game_state
        
        # Check if game is already over
        if game.game_over:
            return self._finalize_step(step_index, done=True, message="Game already over")
        
        # Get current agent
        current_agent_id = game.get_current_agent_id()
        agent = self._agents.get(current_agent_id)
        
        if agent is None:
            return self._finalize_step(step_index, done=True, message=f"Unknown agent: {current_agent_id}")
        
        # Get agent's action
        action = agent.get_action(game)
        
        turns: list[AgentTurn] = []
        display_name = _get_agent_display_name(current_agent_id)
        
        if game.current_phase == "clue":
            # Spymaster giving clue
            if isinstance(action, ClueAction):
                game.process_clue(action.clue_word, action.clue_number, action.reasoning)
                
                action_text = f"CLUE: \"{action.clue_word}\" {action.clue_number}"
                if action.reasoning:
                    action_text += f" (Reasoning: {action.reasoning})"
                
                turn = AgentTurn(
                    agent_id=current_agent_id,
                    action_text=action_text,
                    step_index=step_index,
                    phase="clue",
                    display_name=display_name,
                    metadata={
                        "clue_word": action.clue_word,
                        "clue_number": action.clue_number,
                        "reasoning": action.reasoning,
                        "team": game.current_team,
                        "role": "Spymaster",
                        "phase": "clue",
                    },
                )
                turns.append(turn)
                
                # Add to conversation log (matching Cicero pattern)
                append_to_conversation_log(
                    self._state,
                    speaker_id=current_agent_id,
                    speaker_display=display_name,
                    message=action_text,
                    phase="clue",
                    team=game.current_team,
                )
            
        else:  # guess phase
            # Operative making guess
            if isinstance(action, GuessAction):
                if action.pass_turn:
                    # Operative passes
                    game.pass_turn()
                    
                    action_text = f"PASS: Ending turn. (Reasoning: {action.reasoning})"
                    turn = AgentTurn(
                        agent_id=current_agent_id,
                        action_text=action_text,
                        step_index=step_index,
                        phase="guess_pass",
                        display_name=display_name,
                        metadata={
                            "pass": True,
                            "reasoning": action.reasoning,
                            "team": agent.team,
                            "role": "Operative",
                            "phase": "guess_pass",
                        },
                    )
                    turns.append(turn)
                    
                    append_to_conversation_log(
                        self._state,
                        speaker_id=current_agent_id,
                        speaker_display=display_name,
                        message=action_text,
                        phase="guess_pass",
                        team=agent.team,
                    )
                else:
                    # Process the guess
                    continue_guessing, result_message = game.process_guess(
                        action.guessed_word, action.reasoning
                    )
                    
                    action_text = f"GUESS: \"{action.guessed_word}\" - {result_message}"
                    if action.reasoning:
                        action_text += f" (Reasoning: {action.reasoning})"
                    
                    turn = AgentTurn(
                        agent_id=current_agent_id,
                        action_text=action_text,
                        step_index=step_index,
                        phase="guess",
                        display_name=display_name,
                        metadata={
                            "guessed_word": action.guessed_word,
                            "result": result_message,
                            "reasoning": action.reasoning,
                            "team": agent.team,
                            "continue_guessing": continue_guessing,
                            "role": "Operative",
                            "phase": "guess",
                        },
                    )
                    turns.append(turn)
                    
                    append_to_conversation_log(
                        self._state,
                        speaker_id=current_agent_id,
                        speaker_display=display_name,
                        message=action_text,
                        phase="guess",
                        team=agent.team,
                        guessed_word=action.guessed_word,
                        result=result_message,
                    )
                    
                    # If wrong guess or max guesses reached, end turn
                    if not continue_guessing and not game.game_over:
                        game.end_turn()
        
        # Update state
        self._state.step_index = step_index + 1
        self._state.turns_so_far.extend(turns)
        self._state.done = game.game_over
        
        if game.game_over:
            self._state.metadata["winner"] = game.winner
            self._state.metadata["game_over_reason"] = game.game_over_reason
        
        return MultiAgentStep(
            turns=turns,
            done=game.game_over,
            step_index=step_index,
            game_id=self.get_game_id(),
            state=self._create_state_snapshot(),
        )

    def _finalize_step(self, step_index: int, done: bool, message: str = "") -> MultiAgentStep:
        """Create a final step when game ends or error occurs."""
        self._state.step_index = step_index + 1
        self._state.done = done
        
        return MultiAgentStep(
            turns=[],
            done=done,
            step_index=step_index,
            game_id=self.get_game_id(),
            state=self._create_state_snapshot(),
        )

    def _create_state_snapshot(self) -> MultiAgentState:
        """Create a snapshot of current state for the step output."""
        return MultiAgentState(
            step_index=self._state.step_index,
            turns_so_far=list(self._state.turns_so_far),
            config=self._state.config,
            done=self._state.done,
            metadata=dict(self._state.metadata),
            conversation_log=list(self._state.conversation_log),
        )

    def get_full_game_state(self) -> dict[str, Any]:
        """Get the complete serialized game state for recording."""
        if self._game_state is None:
            return {}
        
        return {
            "game_state": self._game_state.to_dict(),
            "plugin_state": {
                "step_index": self._state.step_index,
                "done": self._state.done,
                "turns_count": len(self._state.turns_so_far),
                "metadata": self._state.metadata,
            },
        }
