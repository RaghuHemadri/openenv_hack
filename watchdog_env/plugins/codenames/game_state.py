"""Game state tracking for Codenames.

Manages board state, revealed words, turn history, and game outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from watchdog_env.plugins.codenames.board_generator import BoardAssignment
from watchdog_env.plugins.codenames.word_interactions import WordInteractions


PhaseType = Literal["clue", "guess"]
TeamType = Literal["red", "blue"]
WordType = Literal["red", "blue", "neutral", "assassin"]


@dataclass
class ClueRecord:
    """Record of a clue given by a spymaster."""
    
    team: TeamType
    clue_word: str
    clue_number: int
    turn_number: int
    reasoning: str = ""  # Optional reasoning from the agent
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "clue_word": self.clue_word,
            "clue_number": self.clue_number,
            "turn_number": self.turn_number,
            "reasoning": self.reasoning,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClueRecord":
        return cls(
            team=data["team"],
            clue_word=data["clue_word"],
            clue_number=data["clue_number"],
            turn_number=data["turn_number"],
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class GuessRecord:
    """Record of a guess made by an operative."""
    
    team: TeamType
    guessed_word: str
    actual_type: WordType
    correct: bool  # True if guessed own team's word
    turn_number: int
    guess_number: int  # Which guess in this turn (1, 2, 3, ...)
    reasoning: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "guessed_word": self.guessed_word,
            "actual_type": self.actual_type,
            "correct": self.correct,
            "turn_number": self.turn_number,
            "guess_number": self.guess_number,
            "reasoning": self.reasoning,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuessRecord":
        return cls(
            team=data["team"],
            guessed_word=data["guessed_word"],
            actual_type=data["actual_type"],
            correct=data["correct"],
            turn_number=data["turn_number"],
            guess_number=data["guess_number"],
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class CodenamesGameState:
    """Complete game state for a Codenames game."""
    
    # Board state
    board: BoardAssignment
    revealed_words: set[str] = field(default_factory=set)
    
    # Game progress
    current_team: TeamType = "red"
    current_phase: PhaseType = "clue"
    turn_number: int = 0
    current_clue: ClueRecord | None = None
    guesses_this_turn: int = 0
    
    # History
    clue_history: list[ClueRecord] = field(default_factory=list)
    guess_history: list[GuessRecord] = field(default_factory=list)
    
    # Game outcome
    winner: TeamType | None = None
    game_over: bool = False
    game_over_reason: str | None = None  # "all_words", "assassin", "max_turns"
    
    # Config reference
    max_turns: int = 20
    
    def get_remaining_words(self, team: TeamType) -> list[str]:
        """Get unrevealed words for a team."""
        return [
            w for w, t in self.board.assignments.items()
            if t == team and w not in self.revealed_words
        ]
    
    def get_revealed_words_by_type(self) -> dict[str, list[str]]:
        """Get revealed words grouped by type."""
        result: dict[str, list[str]] = {"red": [], "blue": [], "neutral": [], "assassin": []}
        for word in self.revealed_words:
            word_type = self.board.assignments.get(word, "neutral")
            result[word_type].append(word)
        return result
    
    def get_board_for_spymaster(self) -> dict[str, Any]:
        """Get board view for spymaster (sees all assignments)."""
        return {
            "grid": self.board.grid,
            "assignments": self.board.assignments,
            "revealed": list(self.revealed_words),
            "interactions": self.board.interactions.to_dict(),
        }
    
    def get_board_for_operative(self) -> dict[str, Any]:
        """Get board view for operative (only sees revealed words)."""
        visible_assignments = {
            w: t for w, t in self.board.assignments.items()
            if w in self.revealed_words
        }
        return {
            "grid": self.board.grid,
            "revealed": list(self.revealed_words),
            "revealed_assignments": visible_assignments,
        }
    
    def get_current_agent_id(self) -> str:
        """Get the current agent ID based on team and phase."""
        if self.current_phase == "clue":
            return f"{self.current_team}_spymaster"
        else:
            return f"{self.current_team}_operative"
    
    def process_clue(self, clue_word: str, clue_number: int, reasoning: str = "") -> None:
        """Process a spymaster's clue."""
        clue = ClueRecord(
            team=self.current_team,
            clue_word=clue_word,
            clue_number=clue_number,
            turn_number=self.turn_number,
            reasoning=reasoning,
        )
        self.current_clue = clue
        self.clue_history.append(clue)
        self.current_phase = "guess"
        self.guesses_this_turn = 0
    
    def process_guess(self, guessed_word: str, reasoning: str = "") -> tuple[bool, str]:
        """Process an operative's guess.
        
        Returns:
            (continue_guessing, result_message)
            continue_guessing: True if the operative can continue guessing
            result_message: Description of what happened
        """
        guessed_word = guessed_word.upper()
        
        if guessed_word in self.revealed_words:
            return False, f"Word '{guessed_word}' was already revealed"
        
        if guessed_word not in self.board.assignments:
            return False, f"Word '{guessed_word}' is not on the board"
        
        actual_type = self.board.assignments[guessed_word]
        correct = actual_type == self.current_team
        
        self.guesses_this_turn += 1
        guess = GuessRecord(
            team=self.current_team,
            guessed_word=guessed_word,
            actual_type=actual_type,
            correct=correct,
            turn_number=self.turn_number,
            guess_number=self.guesses_this_turn,
            reasoning=reasoning,
        )
        self.guess_history.append(guess)
        self.revealed_words.add(guessed_word)
        
        # Check for assassin
        if actual_type == "assassin":
            self.game_over = True
            self.winner = "blue" if self.current_team == "red" else "red"
            self.game_over_reason = "assassin"
            return False, f"ASSASSIN! {self.current_team.upper()} team loses!"
        
        # Check for all words found
        red_remaining = len(self.get_remaining_words("red"))
        blue_remaining = len(self.get_remaining_words("blue"))
        
        if red_remaining == 0:
            self.game_over = True
            self.winner = "red"
            self.game_over_reason = "all_words"
            return False, "RED team found all their words and wins!"
        
        if blue_remaining == 0:
            self.game_over = True
            self.winner = "blue"
            self.game_over_reason = "all_words"
            return False, "BLUE team found all their words and wins!"
        
        # Check if guess was correct
        if correct:
            # Can continue if haven't exceeded clue number + 1 bonus
            max_guesses = (self.current_clue.clue_number + 1) if self.current_clue else 1
            if self.guesses_this_turn < max_guesses:
                return True, f"Correct! '{guessed_word}' is {self.current_team.upper()}. You may continue guessing."
            else:
                return False, f"Correct! '{guessed_word}' is {self.current_team.upper()}. Maximum guesses reached."
        else:
            if actual_type == "neutral":
                return False, f"'{guessed_word}' is NEUTRAL. Turn ends."
            else:
                opponent = "blue" if self.current_team == "red" else "red"
                return False, f"'{guessed_word}' is {opponent.upper()}'s word! Turn ends."
    
    def end_turn(self) -> None:
        """End the current team's turn and switch to the other team."""
        self.current_team = "blue" if self.current_team == "red" else "red"
        self.current_phase = "clue"
        self.current_clue = None
        self.guesses_this_turn = 0
        self.turn_number += 1
        
        # Check for max turns
        if self.turn_number >= self.max_turns:
            self.game_over = True
            red_remaining = len(self.get_remaining_words("red"))
            blue_remaining = len(self.get_remaining_words("blue"))
            if red_remaining < blue_remaining:
                self.winner = "red"
            elif blue_remaining < red_remaining:
                self.winner = "blue"
            else:
                self.winner = None  # Draw
            self.game_over_reason = "max_turns"
    
    def pass_turn(self) -> None:
        """Operative passes (ends guessing phase)."""
        self.end_turn()
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize game state to dictionary."""
        return {
            "board": self.board.to_dict(),
            "revealed_words": list(self.revealed_words),
            "current_team": self.current_team,
            "current_phase": self.current_phase,
            "turn_number": self.turn_number,
            "current_clue": self.current_clue.to_dict() if self.current_clue else None,
            "guesses_this_turn": self.guesses_this_turn,
            "clue_history": [c.to_dict() for c in self.clue_history],
            "guess_history": [g.to_dict() for g in self.guess_history],
            "winner": self.winner,
            "game_over": self.game_over,
            "game_over_reason": self.game_over_reason,
            "max_turns": self.max_turns,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodenamesGameState":
        """Deserialize from dictionary."""
        board = BoardAssignment.from_dict(data["board"])
        current_clue = ClueRecord.from_dict(data["current_clue"]) if data.get("current_clue") else None
        
        return cls(
            board=board,
            revealed_words=set(data.get("revealed_words", [])),
            current_team=data.get("current_team", "red"),
            current_phase=data.get("current_phase", "clue"),
            turn_number=data.get("turn_number", 0),
            current_clue=current_clue,
            guesses_this_turn=data.get("guesses_this_turn", 0),
            clue_history=[ClueRecord.from_dict(c) for c in data.get("clue_history", [])],
            guess_history=[GuessRecord.from_dict(g) for g in data.get("guess_history", [])],
            winner=data.get("winner"),
            game_over=data.get("game_over", False),
            game_over_reason=data.get("game_over_reason"),
            max_turns=data.get("max_turns", 20),
        )
    
    def get_game_summary(self) -> str:
        """Get a human-readable summary of the game state."""
        lines = [
            f"Turn {self.turn_number} | {self.current_team.upper()}'s {self.current_phase.upper()} phase",
            f"Red remaining: {len(self.get_remaining_words('red'))} | Blue remaining: {len(self.get_remaining_words('blue'))}",
        ]
        
        if self.current_clue:
            lines.append(f"Current clue: {self.current_clue.clue_word} {self.current_clue.clue_number}")
            lines.append(f"Guesses made: {self.guesses_this_turn}")
        
        if self.game_over:
            lines.append(f"GAME OVER: {self.winner.upper() if self.winner else 'DRAW'} ({self.game_over_reason})")
        
        return "\n".join(lines)
