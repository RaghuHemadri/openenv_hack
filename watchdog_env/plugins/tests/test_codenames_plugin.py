"""Tests for the Codenames multi-agent plugin.

Tests cover:
- Plugin registration and basic methods
- Configuration validation
- Game state transitions (using mock board)
- Word interactions serialization
- Conversation log (matching Cicero pattern)
- Integration tests

Uses shared local Qwen3 8B game-play model from avalon/llm.py.
Tests marked with SKIP_WITHOUT_GPU require a GPU for the local model.
"""

from __future__ import annotations

import pytest
from watchdog_env.plugins.codenames.codenames_config import CodenamesConfig, CODENAMES_AGENTS
from watchdog_env.plugins.codenames.codenames_plugin import CodenamesPlugin
from watchdog_env.plugins.codenames.board_generator import (
    generate_board,
    BoardAssignment,
    BoardGenerationError,
)
from watchdog_env.plugins.codenames.game_state import CodenamesGameState, ClueRecord, GuessRecord
from watchdog_env.plugins.codenames.word_interactions import (
    WordInteractions,
    WordRelation,
    ThematicCluster,
)
from watchdog_env.plugins.codenames.agents import (
    CodenamesAgent,
    create_agents,
    ClueAction,
    GuessAction,
    AgentActionError,
)
from watchdog_env.plugins.codenames.game_runner import (
    CodenamesGameRunner,
    GameRecord,
)
from watchdog_env.plugins.base import get_conversation_log
from watchdog_env.plugins.registry import get_plugin, list_game_ids


def _has_gpu():
    """Check if GPU is available for local model."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Skip tests that require GPU for local model
SKIP_WITHOUT_GPU = pytest.mark.skipif(
    not _has_gpu(),
    reason="GPU not available for local model"
)


def _create_mock_board() -> BoardAssignment:
    """Create a mock board for testing without API."""
    words = [
        "APPLE", "BANK", "CARD", "DOOR", "ENGINE",
        "FIRE", "GHOST", "HOTEL", "ICE", "JAZZ",
        "KING", "LAMP", "MOON", "NIGHT", "OPERA",
        "PIANO", "QUEEN", "RIVER", "STAR", "TREE",
        "UMBRELLA", "VIOLIN", "WATER", "YARD", "ZEBRA",
    ]
    
    assignments = {}
    # Assassin
    assignments["ZEBRA"] = "assassin"
    # Red team (9)
    for w in words[0:9]:
        assignments[w] = "red"
    # Blue team (8)
    for w in words[9:17]:
        assignments[w] = "blue"
    # Neutral (7)
    for w in words[17:24]:
        assignments[w] = "neutral"
    
    interactions = WordInteractions(words=words)
    
    grid = [words[i*5:(i+1)*5] for i in range(5)]
    
    return BoardAssignment(
        words=words,
        assignments=assignments,
        interactions=interactions,
        grid=grid,
    )


# ============================================================================
# Plugin Registration Tests
# ============================================================================

class TestPluginRegistration:
    """Test plugin registration and basic interface."""
    
    def test_plugin_registered(self):
        """Codenames plugin should be registered."""
        assert "codenames" in list_game_ids()
    
    def test_get_plugin(self):
        """Should be able to get Codenames plugin by ID."""
        plugin = get_plugin("codenames")
        assert plugin is not None
        assert plugin.get_game_id() == "codenames"
    
    def test_game_id(self):
        """Plugin should return correct game ID."""
        plugin = CodenamesPlugin()
        assert plugin.get_game_id() == "codenames"
    
    def test_display_name(self):
        """Plugin should return display name."""
        plugin = CodenamesPlugin()
        assert "Codenames" in plugin.get_display_name()
    
    def test_list_agent_ids(self):
        """Plugin should list all 4 agent IDs."""
        plugin = CodenamesPlugin()
        agents = plugin.list_agent_ids()
        assert len(agents) == 4
        assert "red_spymaster" in agents
        assert "red_operative" in agents
        assert "blue_spymaster" in agents
        assert "blue_operative" in agents


# ============================================================================
# Configuration Tests
# ============================================================================

class TestCodenamesConfig:
    """Test game configuration."""
    
    def test_default_config(self):
        """Default config should have valid values."""
        config = CodenamesConfig()
        assert config.board_size == 25
        assert config.red_words == 9
        assert config.blue_words == 8
        assert config.neutral_words == 7
        assert config.assassin_words == 1
        assert config.starting_team == "red"
        assert config.complexity_level == 2
    
    def test_validate_correct_total(self):
        """Config with correct word totals should validate."""
        config = CodenamesConfig()
        config.validate()  # Should not raise
    
    def test_validate_incorrect_total(self):
        """Config with incorrect word totals should fail validation."""
        config = CodenamesConfig(red_words=10)  # Total = 26 != 25
        with pytest.raises(ValueError, match="must equal board_size"):
            config.validate()
    
    def test_validate_invalid_team(self):
        """Config with invalid starting team should fail validation."""
        config = CodenamesConfig(starting_team="green")  # type: ignore
        with pytest.raises(ValueError, match="must be 'red' or 'blue'"):
            config.validate()
    
    def test_validate_invalid_complexity(self):
        """Config with invalid complexity should fail validation."""
        config = CodenamesConfig(complexity_level=5)
        with pytest.raises(ValueError, match="must be 1, 2, or 3"):
            config.validate()


# ============================================================================
# Board Generation Tests
# ============================================================================

@SKIP_WITHOUT_GPU
class TestBoardGeneration:
    """Test board generation functionality (requires API)."""
    
    def test_generate_board_basic(self):
        """Should generate board with basic complexity."""
        board = generate_board(seed=42, complexity_level=1)
        assert len(board.words) == 25
        assert len(board.assignments) == 25
        assert len(board.grid) == 5
        assert all(len(row) == 5 for row in board.grid)
    
    def test_generate_board_medium(self):
        """Should generate board with medium complexity."""
        board = generate_board(seed=42, complexity_level=2)
        assert len(board.words) == 25
        assert board.interactions is not None
    
    def test_generate_board_complex(self):
        """Should generate board with complex complexity."""
        board = generate_board(seed=42, complexity_level=3)
        assert len(board.words) == 25
        assert board.interactions is not None
    
    def test_board_assignments_correct_counts(self):
        """Board should have correct number of each word type."""
        board = generate_board(seed=42)
        
        red_count = sum(1 for t in board.assignments.values() if t == "red")
        blue_count = sum(1 for t in board.assignments.values() if t == "blue")
        neutral_count = sum(1 for t in board.assignments.values() if t == "neutral")
        assassin_count = sum(1 for t in board.assignments.values() if t == "assassin")
        
        assert red_count == 9
        assert blue_count == 8
        assert neutral_count == 7
        assert assassin_count == 1


# ============================================================================
# Word Interactions Tests
# ============================================================================

class TestWordInteractions:
    """Test word interaction functionality."""
    
    def test_word_interactions_serialization(self):
        """Word interactions should serialize and deserialize correctly."""
        original = WordInteractions(
            words=["BANK", "RIVER", "CURRENT"],
            relations={
                "BANK": WordRelation("BANK", ["RIVER"], "polyseme", ["finance", "nature"]),
            },
            clusters=[ThematicCluster("water", ["BANK", "RIVER", "CURRENT"])],
            polysemes=["BANK", "CURRENT"],
            false_friends=[("BANK", "SUIT")],
            assassin_traps=["CURRENT"],
        )
        
        data = original.to_dict()
        restored = WordInteractions.from_dict(data)
        
        assert restored.words == original.words
        assert restored.polysemes == original.polysemes
        assert len(restored.clusters) == len(original.clusters)
        assert len(restored.relations) == len(original.relations)
    
    def test_get_related_words(self):
        """Should get related words for a word."""
        interactions = WordInteractions(
            words=["BANK", "RIVER"],
            relations={
                "BANK": WordRelation("BANK", ["RIVER", "CURRENT"], "polyseme"),
            },
        )
        
        related = interactions.get_related_words("BANK")
        assert "RIVER" in related
        assert "CURRENT" in related
    
    def test_get_trap_level(self):
        """Should get trap level for a word."""
        interactions = WordInteractions(
            words=["BANK"],
            relations={
                "BANK": WordRelation("BANK", [], "polyseme", [], trap_level=2),
            },
        )
        
        assert interactions.get_trap_level("BANK") == 2
        assert interactions.get_trap_level("UNKNOWN") == 0


# ============================================================================
# Game State Tests (using mock board)
# ============================================================================

class TestGameState:
    """Test game state management using mock board."""
    
    @pytest.fixture
    def game_state(self):
        """Create a game state for testing."""
        board = _create_mock_board()
        return CodenamesGameState(board=board)
    
    def test_initial_state(self, game_state):
        """Initial state should be correct."""
        assert game_state.current_team == "red"
        assert game_state.current_phase == "clue"
        assert game_state.turn_number == 0
        assert not game_state.game_over
        assert game_state.winner is None
    
    def test_get_remaining_words(self, game_state):
        """Should get correct remaining words for each team."""
        red_words = game_state.get_remaining_words("red")
        blue_words = game_state.get_remaining_words("blue")
        
        assert len(red_words) == 9
        assert len(blue_words) == 8
    
    def test_process_clue(self, game_state):
        """Processing clue should update state correctly."""
        game_state.process_clue("ANIMAL", 3, "Three animal words")
        
        assert game_state.current_phase == "guess"
        assert game_state.current_clue is not None
        assert game_state.current_clue.clue_word == "ANIMAL"
        assert game_state.current_clue.clue_number == 3
        assert len(game_state.clue_history) == 1
    
    def test_process_correct_guess(self, game_state):
        """Correct guess should allow continued guessing."""
        game_state.process_clue("TEST", 2)
        
        # Get a red word
        red_words = game_state.get_remaining_words("red")
        word = red_words[0]
        
        can_continue, message = game_state.process_guess(word)
        
        assert can_continue
        assert word in game_state.revealed_words
        assert len(game_state.get_remaining_words("red")) == 8
    
    def test_process_wrong_guess_ends_turn(self, game_state):
        """Wrong guess should end turn."""
        game_state.process_clue("TEST", 2)
        
        # Get a blue word (wrong for red team)
        blue_words = game_state.get_remaining_words("blue")
        word = blue_words[0]
        
        can_continue, message = game_state.process_guess(word)
        
        assert not can_continue
        assert word in game_state.revealed_words
    
    def test_assassin_ends_game(self, game_state):
        """Hitting assassin should end game."""
        game_state.process_clue("TEST", 1)
        
        # Get assassin word
        assassin_word = [w for w, t in game_state.board.assignments.items() if t == "assassin"][0]
        
        can_continue, message = game_state.process_guess(assassin_word)
        
        assert not can_continue
        assert game_state.game_over
        assert game_state.winner == "blue"  # Opponent wins
        assert game_state.game_over_reason == "assassin"
    
    def test_all_words_found_wins(self, game_state):
        """Finding all team words should win the game."""
        game_state.process_clue("TEST", 10)  # High number for testing
        
        red_words = list(game_state.get_remaining_words("red"))
        for word in red_words:
            can_continue, message = game_state.process_guess(word)
            if game_state.game_over:
                break
        
        assert game_state.game_over
        assert game_state.winner == "red"
        assert game_state.game_over_reason == "all_words"
    
    def test_end_turn_switches_team(self, game_state):
        """Ending turn should switch to other team."""
        assert game_state.current_team == "red"
        
        game_state.process_clue("TEST", 1)
        game_state.end_turn()
        
        assert game_state.current_team == "blue"
        assert game_state.current_phase == "clue"
        assert game_state.turn_number == 1
    
    def test_state_serialization(self, game_state):
        """Game state should serialize and deserialize correctly."""
        game_state.process_clue("TEST", 2)
        
        data = game_state.to_dict()
        restored = CodenamesGameState.from_dict(data)
        
        assert restored.current_team == game_state.current_team
        assert restored.current_phase == game_state.current_phase
        assert restored.turn_number == game_state.turn_number
        assert len(restored.clue_history) == len(game_state.clue_history)


# ============================================================================
# Agent Tests
# ============================================================================

class TestAgentBasics:
    """Test basic agent functionality."""
    
    def test_create_agents(self):
        """Should create all 4 agents."""
        agents = create_agents()
        assert len(agents) == 4
        assert "red_spymaster" in agents
        assert "red_operative" in agents
        assert "blue_spymaster" in agents
        assert "blue_operative" in agents
    
    def test_agent_roles(self):
        """Agents should have correct roles."""
        agents = create_agents()
        
        assert agents["red_spymaster"].is_spymaster
        assert agents["red_spymaster"].team == "red"
        
        assert not agents["red_operative"].is_spymaster
        assert agents["red_operative"].team == "red"
        
        assert agents["blue_spymaster"].is_spymaster
        assert agents["blue_spymaster"].team == "blue"


@SKIP_WITHOUT_GPU
class TestAgentsWithLLM:
    """Test agent functionality with API."""
    
    def test_spymaster_clue(self):
        """Spymaster should generate valid clue."""
        board = generate_board(seed=42)
        game_state = CodenamesGameState(board=board)
        
        agent = CodenamesAgent("red_spymaster")
        action = agent.get_action(game_state)
        
        assert isinstance(action, ClueAction)
        assert action.clue_word
        assert action.clue_number >= 1
    
    def test_operative_guess(self):
        """Operative should generate valid guess."""
        board = generate_board(seed=42)
        game_state = CodenamesGameState(board=board)
        game_state.process_clue("TEST", 2)
        
        agent = CodenamesAgent("red_operative")
        action = agent.get_action(game_state)
        
        assert isinstance(action, GuessAction)
        # Either a valid word guess or a pass
        if not action.pass_turn:
            unrevealed = set(game_state.board.words) - game_state.revealed_words
            assert action.guessed_word in unrevealed


# ============================================================================
# Plugin Integration Tests
# ============================================================================

@SKIP_WITHOUT_GPU
class TestPluginIntegration:
    """Test the plugin as a whole (requires API)."""
    
    def test_reset_and_generate_step(self):
        """Should reset and generate steps."""
        plugin = CodenamesPlugin()
        plugin.reset(seed=42, config=CodenamesConfig(complexity_level=1))
        
        # First step should be red spymaster clue
        step0 = plugin.generate_step(seed=42, step_index=0)
        
        assert len(step0.turns) >= 1
        assert step0.turns[0].agent_id == "red_spymaster"
        assert "CLUE" in step0.turns[0].action_text
    
    def test_state_tracking(self):
        """Plugin should track state correctly."""
        plugin = CodenamesPlugin()
        plugin.reset(seed=42)
        
        state = plugin.get_state()
        assert state.step_index == 0
        assert len(state.turns_so_far) == 0
        
        plugin.generate_step(seed=42, step_index=0)
        
        state = plugin.get_state()
        assert state.step_index == 1
        assert len(state.turns_so_far) >= 1
    
    def test_game_state_access(self):
        """Should be able to access internal game state."""
        plugin = CodenamesPlugin()
        plugin.reset(seed=42)
        
        game_state = plugin.get_game_state()
        assert game_state is not None
        assert len(game_state.board.words) == 25
    
    def test_full_game_state(self):
        """Should get full serialized game state."""
        plugin = CodenamesPlugin()
        plugin.reset(seed=42)
        
        full_state = plugin.get_full_game_state()
        assert "game_state" in full_state
        assert "plugin_state" in full_state
    
    def test_conversation_log_cleared_on_reset(self):
        """Reset clears conversation_log (matching Cicero pattern)."""
        plugin = CodenamesPlugin()
        plugin.reset(seed=1, config=CodenamesConfig(complexity_level=1))
        plugin.generate_step(seed=1, step_index=0)
        plugin.reset(seed=99, config=CodenamesConfig(complexity_level=1))
        assert len(get_conversation_log(plugin.get_state())) == 0
    
    def test_conversation_log_in_step_state(self):
        """Each step returns state with conversation_log; entries have speaker_id, message."""
        plugin = CodenamesPlugin()
        plugin.reset(seed=1, config=CodenamesConfig(complexity_level=1))
        step = plugin.generate_step(seed=1, step_index=0)
        assert step.state is not None
        assert isinstance(step.state.conversation_log, list)
        log = step.state.conversation_log
        for entry in log:
            assert "speaker_id" in entry and "message" in entry
            assert isinstance(entry["message"], str) and len(entry["message"]) > 0
    
    def test_conversation_log_accumulates(self):
        """Conversation log accumulates across steps."""
        plugin = CodenamesPlugin()
        plugin.reset(seed=1, config=CodenamesConfig(complexity_level=1))
        
        plugin.generate_step(seed=1, step_index=0)
        log1 = get_conversation_log(plugin.get_state())
        assert len(log1) >= 1
        
        plugin.generate_step(seed=1, step_index=1)
        log2 = get_conversation_log(plugin.get_state())
        assert len(log2) >= 2


# ============================================================================
# Game Runner Tests
# ============================================================================

@SKIP_WITHOUT_GPU
class TestGameRunner:
    """Test game runner functionality (requires API)."""
    
    def test_run_game_produces_record(self):
        """Running a game should produce a GameRecord."""
        runner = CodenamesGameRunner(max_steps=10)
        record = runner.run_game(seed=42, verbose=False)
        
        assert isinstance(record, GameRecord)
        assert record.game_id
        assert record.start_time
        assert len(record.steps) > 0
    
    def test_game_record_serialization(self):
        """Game record should serialize to JSON."""
        runner = CodenamesGameRunner(max_steps=5)
        record = runner.run_game(seed=42, verbose=False)
        
        json_str = record.to_json()
        assert json_str
        assert "game_id" in json_str
        assert "steps" in json_str
    
    def test_game_record_save_load(self, tmp_path):
        """Game record should save and load correctly."""
        runner = CodenamesGameRunner(max_steps=5)
        record = runner.run_game(seed=42, verbose=False)
        
        filepath = tmp_path / "test_game.json"
        record.save(filepath)
        
        loaded = GameRecord.load(filepath)
        assert loaded.game_id == record.game_id
        assert loaded.seed == record.seed
        assert len(loaded.steps) == len(record.steps)


# ============================================================================
# Full Integration Tests with API
# ============================================================================

@SKIP_WITHOUT_GPU
class TestFullIntegration:
    """Full integration tests that require LLM (local or Gemini)."""
    
    def test_board_generation_with_llm(self):
        """Board generation should use LLM."""
        board = generate_board(seed=42, complexity_level=2)
        assert len(board.words) == 25
        assert board.interactions is not None
    
    def test_agent_with_llm(self):
        """Agent should use LLM for decisions."""
        board = generate_board(seed=42)
        game_state = CodenamesGameState(board=board)
        
        agent = CodenamesAgent("red_spymaster")
        action = agent.get_action(game_state)
        
        assert isinstance(action, ClueAction)
        # LLM should provide reasoning
        assert action.reasoning
    
    def test_full_game_with_llm(self):
        """Should be able to run a full game with LLM."""
        runner = CodenamesGameRunner(max_steps=50)
        record = runner.run_game(
            config=CodenamesConfig(complexity_level=2),
            seed=42,
            verbose=False
        )
        
        assert len(record.steps) > 0
        # Game should have some meaningful interactions
        clue_steps = [s for s in record.steps if "CLUE" in s.action_text]
        guess_steps = [s for s in record.steps if "GUESS" in s.action_text]
        
        assert len(clue_steps) > 0
        assert len(guess_steps) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
