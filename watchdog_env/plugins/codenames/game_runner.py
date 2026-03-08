"""Game runner for Codenames that plays full games and records all interactions.

Provides utilities to run games, record transcripts, and save game histories.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from watchdog_env.plugins.codenames.codenames_config import CodenamesConfig
from watchdog_env.plugins.codenames.codenames_plugin import CodenamesPlugin
from watchdog_env.plugins.base import AgentTurn

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """Record of a single step in the game."""
    
    step_index: int
    agent_id: str
    action_text: str
    phase: str
    metadata: dict[str, Any] = field(default_factory=dict)
    game_state_snapshot: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "agent_id": self.agent_id,
            "action_text": self.action_text,
            "phase": self.phase,
            "metadata": self.metadata,
            "game_state_snapshot": self.game_state_snapshot,
        }


@dataclass
class GameRecord:
    """Complete record of a Codenames game."""
    
    game_id: str
    config: CodenamesConfig
    seed: int | None
    start_time: str
    end_time: str | None = None
    winner: str | None = None
    game_over_reason: str | None = None
    total_steps: int = 0
    total_turns: int = 0
    steps: list[StepRecord] = field(default_factory=list)
    initial_board: dict[str, Any] = field(default_factory=dict)
    final_state: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "config": {
                "board_size": self.config.board_size,
                "red_words": self.config.red_words,
                "blue_words": self.config.blue_words,
                "neutral_words": self.config.neutral_words,
                "assassin_words": self.config.assassin_words,
                "starting_team": self.config.starting_team,
                "max_turns": self.config.max_turns,
                "complexity_level": self.config.complexity_level,
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
            },
            "seed": self.seed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "winner": self.winner,
            "game_over_reason": self.game_over_reason,
            "total_steps": self.total_steps,
            "total_turns": self.total_turns,
            "steps": [s.to_dict() for s in self.steps],
            "initial_board": self.initial_board,
            "final_state": self.final_state,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str | Path) -> None:
        """Save the game record to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.to_json())
        logger.info("Saved game record to %s", filepath)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameRecord":
        """Load a game record from a dictionary."""
        config_data = data.get("config", {})
        config = CodenamesConfig(
            board_size=config_data.get("board_size", 25),
            red_words=config_data.get("red_words", 9),
            blue_words=config_data.get("blue_words", 8),
            neutral_words=config_data.get("neutral_words", 7),
            assassin_words=config_data.get("assassin_words", 1),
            starting_team=config_data.get("starting_team", "red"),
            max_turns=config_data.get("max_turns", 20),
            complexity_level=config_data.get("complexity_level", 2),
            model_name=config_data.get("model_name", "gemini-2.0-flash"),
            temperature=config_data.get("temperature", 0.7),
        )
        
        steps = [
            StepRecord(
                step_index=s["step_index"],
                agent_id=s["agent_id"],
                action_text=s["action_text"],
                phase=s["phase"],
                metadata=s.get("metadata", {}),
                game_state_snapshot=s.get("game_state_snapshot"),
            )
            for s in data.get("steps", [])
        ]
        
        return cls(
            game_id=data.get("game_id", ""),
            config=config,
            seed=data.get("seed"),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time"),
            winner=data.get("winner"),
            game_over_reason=data.get("game_over_reason"),
            total_steps=data.get("total_steps", 0),
            total_turns=data.get("total_turns", 0),
            steps=steps,
            initial_board=data.get("initial_board", {}),
            final_state=data.get("final_state", {}),
        )
    
    @classmethod
    def load(cls, filepath: str | Path) -> "GameRecord":
        """Load a game record from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)


class CodenamesGameRunner:
    """Runner to play Codenames games and record all interactions."""
    
    def __init__(
        self,
        record_state_snapshots: bool = True,
        max_steps: int = 200,  # Safety limit
    ):
        self.record_state_snapshots = record_state_snapshots
        self.max_steps = max_steps
    
    def run_game(
        self,
        config: CodenamesConfig | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> GameRecord:
        """Run a complete Codenames game.
        
        Args:
            config: Game configuration (uses defaults if None)
            seed: Random seed for reproducibility
            verbose: Print game progress to console
        
        Returns:
            GameRecord with all interactions recorded
        """
        if config is None:
            config = CodenamesConfig()
        
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        # Initialize plugin
        plugin = CodenamesPlugin()
        plugin.reset(seed=seed, config=config)
        
        # Create game record
        game_id = f"codenames_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed}"
        start_time = datetime.now().isoformat()
        
        # Record initial board state
        initial_state = plugin.get_full_game_state()
        initial_board = initial_state.get("game_state", {}).get("board", {})
        
        record = GameRecord(
            game_id=game_id,
            config=config,
            seed=seed,
            start_time=start_time,
            initial_board=initial_board,
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Codenames Game: {game_id}")
            print(f"Complexity Level: {config.complexity_level}")
            print(f"Seed: {seed}")
            print(f"{'='*60}\n")
            
            # Print initial board
            game_state = plugin.get_game_state()
            if game_state:
                print("Board:")
                for row in game_state.board.grid:
                    print("  " + " | ".join(f"{w:12}" for w in row))
                print()
        
        # Run game loop
        step_index = 0
        while not plugin.get_state().done and step_index < self.max_steps:
            step = plugin.generate_step(seed=seed, step_index=step_index)
            
            for turn in step.turns:
                step_record = StepRecord(
                    step_index=turn.step_index,
                    agent_id=turn.agent_id,
                    action_text=turn.action_text,
                    phase=turn.phase,
                    metadata=dict(turn.metadata),
                )
                
                if self.record_state_snapshots:
                    step_record.game_state_snapshot = plugin.get_full_game_state()
                
                record.steps.append(step_record)
                
                if verbose:
                    team = "RED" if "red" in turn.agent_id else "BLUE"
                    role = "Spymaster" if "spymaster" in turn.agent_id else "Operative"
                    print(f"[Step {step_index}] {team} {role}: {turn.action_text}")
            
            step_index += 1
            
            if step.done:
                break
        
        # Record final state
        final_state = plugin.get_full_game_state()
        game_state = plugin.get_game_state()
        
        record.end_time = datetime.now().isoformat()
        record.total_steps = step_index
        record.final_state = final_state
        
        if game_state:
            record.winner = game_state.winner
            record.game_over_reason = game_state.game_over_reason
            record.total_turns = game_state.turn_number
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Game Over!")
            if record.winner:
                print(f"Winner: {record.winner.upper()}")
            else:
                print("Result: DRAW")
            print(f"Reason: {record.game_over_reason}")
            print(f"Total Steps: {record.total_steps}")
            print(f"Total Turns: {record.total_turns}")
            print(f"{'='*60}\n")
        
        return record
    
    def run_multiple_games(
        self,
        num_games: int,
        config: CodenamesConfig | None = None,
        base_seed: int | None = None,
        verbose: bool = False,
        save_dir: str | Path | None = None,
    ) -> list[GameRecord]:
        """Run multiple games and optionally save records.
        
        Args:
            num_games: Number of games to run
            config: Game configuration (uses defaults if None)
            base_seed: Base seed (game i uses base_seed + i)
            verbose: Print progress
            save_dir: Directory to save game records
        
        Returns:
            List of GameRecords
        """
        if base_seed is None:
            base_seed = random.randint(0, 2**31 - 1)
        
        records = []
        
        for i in range(num_games):
            seed = base_seed + i
            
            if verbose:
                print(f"\nRunning game {i+1}/{num_games} (seed={seed})...")
            
            record = self.run_game(config=config, seed=seed, verbose=verbose)
            records.append(record)
            
            if save_dir:
                save_path = Path(save_dir) / f"{record.game_id}.json"
                record.save(save_path)
        
        # Print summary
        if verbose and records:
            red_wins = sum(1 for r in records if r.winner == "red")
            blue_wins = sum(1 for r in records if r.winner == "blue")
            draws = sum(1 for r in records if r.winner is None)
            
            print(f"\n{'='*60}")
            print(f"Summary of {num_games} games:")
            print(f"  Red wins: {red_wins} ({100*red_wins/num_games:.1f}%)")
            print(f"  Blue wins: {blue_wins} ({100*blue_wins/num_games:.1f}%)")
            print(f"  Draws: {draws} ({100*draws/num_games:.1f}%)")
            
            assassin_losses = sum(1 for r in records if r.game_over_reason == "assassin")
            all_words_wins = sum(1 for r in records if r.game_over_reason == "all_words")
            max_turns_ends = sum(1 for r in records if r.game_over_reason == "max_turns")
            
            print(f"\nEnd reasons:")
            print(f"  All words found: {all_words_wins}")
            print(f"  Assassin hit: {assassin_losses}")
            print(f"  Max turns reached: {max_turns_ends}")
            print(f"{'='*60}\n")
        
        return records


def run_demo_game(complexity_level: int = 2, verbose: bool = True) -> GameRecord:
    """Run a single demo game with default settings.
    
    This is a convenience function for quick testing.
    """
    config = CodenamesConfig(complexity_level=complexity_level)
    runner = CodenamesGameRunner()
    return runner.run_game(config=config, verbose=verbose)


def print_game_transcript(record: GameRecord) -> None:
    """Print a formatted transcript of a game."""
    print(f"\n{'='*70}")
    print(f"CODENAMES GAME TRANSCRIPT")
    print(f"Game ID: {record.game_id}")
    print(f"Started: {record.start_time}")
    print(f"Ended: {record.end_time}")
    print(f"{'='*70}\n")
    
    print("CONFIGURATION:")
    print(f"  Complexity: {record.config.complexity_level}")
    print(f"  Red words: {record.config.red_words}")
    print(f"  Blue words: {record.config.blue_words}")
    print(f"  Starting team: {record.config.starting_team}")
    print()
    
    print("INITIAL BOARD:")
    grid = record.initial_board.get("grid", [])
    assignments = record.initial_board.get("assignments", {})
    for row in grid:
        row_str = " | ".join(f"{w:12}" for w in row)
        print(f"  {row_str}")
    print()
    
    print("WORD ASSIGNMENTS:")
    for team in ["red", "blue", "neutral", "assassin"]:
        words = [w for w, t in assignments.items() if t == team]
        print(f"  {team.upper()}: {', '.join(words)}")
    print()
    
    print("GAME TRANSCRIPT:")
    print("-" * 70)
    
    current_turn = -1
    for step in record.steps:
        turn = step.metadata.get("turn_number", step.step_index // 2)
        if turn != current_turn:
            current_turn = turn
            print(f"\n--- Turn {turn} ---")
        
        team = "RED" if "red" in step.agent_id else "BLUE"
        role = "Spymaster" if "spymaster" in step.agent_id else "Operative"
        print(f"[{team} {role}] {step.action_text}")
    
    print(f"\n{'-'*70}")
    print(f"RESULT: {record.winner.upper() if record.winner else 'DRAW'}")
    print(f"Reason: {record.game_over_reason}")
    print(f"Total turns: {record.total_turns}")
    print(f"Total steps: {record.total_steps}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Run a demo game when executed directly
    import sys
    
    complexity = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    record = run_demo_game(complexity_level=complexity, verbose=True)
    
    # Optionally save
    if len(sys.argv) > 2:
        record.save(sys.argv[2])
