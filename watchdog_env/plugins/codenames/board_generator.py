"""Board generator for Codenames.

Generates word boards with complex semantic relationships based on complexity level.

Uses shared local Qwen3 8B game-play model from avalon/llm.py.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import os
from watchdog_env.plugins.codenames.word_interactions import (
    WordInteractions,
    WordRelation,
    ThematicCluster,
)

logger = logging.getLogger(__name__)


class BoardGenerationError(Exception):
    """Raised when board generation fails."""
    pass


@dataclass
class BoardAssignment:
    """Complete board with word assignments and interactions."""
    
    words: list[str]
    assignments: dict[str, str]  # word -> "red"/"blue"/"neutral"/"assassin"
    interactions: WordInteractions
    grid: list[list[str]] = field(default_factory=list)  # 5x5 grid representation
    
    def get_team_words(self, team: str) -> list[str]:
        """Get all words assigned to a team."""
        return [w for w, t in self.assignments.items() if t == team]
    
    def get_unrevealed_team_words(self, team: str, revealed: set[str]) -> list[str]:
        """Get unrevealed words for a team."""
        return [w for w in self.get_team_words(team) if w not in revealed]
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "words": self.words,
            "assignments": self.assignments,
            "interactions": self.interactions.to_dict(),
            "grid": self.grid,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoardAssignment":
        """Deserialize from dictionary."""
        return cls(
            words=data["words"],
            assignments=data["assignments"],
            interactions=WordInteractions.from_dict(data.get("interactions", {"words": data["words"]})),
            grid=data.get("grid", []),
        )


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


def _build_generation_prompt(complexity_level: int, board_size: int = 25) -> str:
    """Build the prompt for Gemini to generate a Codenames board."""
    
    if complexity_level == 1:
        return f"""Generate {board_size} single English words for a Codenames word game board.

Requirements:
- All words must be common, concrete nouns or verbs
- No proper nouns, no compound words, no phrases
- Words should be diverse in category (animals, objects, places, actions, etc.)
- Easy to understand and visualize

Return ONLY a JSON object with this exact format:
{{"words": ["WORD1", "WORD2", ...], "clusters": [], "polysemes": [], "relations": {{}}}}

All words must be UPPERCASE."""

    elif complexity_level == 2:
        return f"""Generate {board_size} English words for a Codenames word game board with MEDIUM complexity.

Requirements:
- Include 2-3 thematic clusters (4-5 words each) that share hidden themes
- Include 3-4 polysemes (words with multiple meanings like BANK, CELL, PITCH)
- Some words should have overlapping semantic domains
- Mix of concrete and abstract nouns

Examples of good thematic clusters:
- Water theme: BANK, RIVER, CURRENT, STREAM, WAVE
- Music theme: PITCH, NOTE, SHARP, FLAT, SCALE

Return ONLY a JSON object with this exact format:
{{
    "words": ["WORD1", "WORD2", ...],
    "clusters": [
        {{"theme": "theme_name", "words": ["W1", "W2", ...], "secondary_themes": ["alt_theme"]}}
    ],
    "polysemes": ["BANK", "CELL", ...],
    "relations": {{
        "WORD": {{"related_words": ["W1", "W2"], "domains": ["domain1", "domain2"], "relation_type": "polyseme"}}
    }}
}}

All words must be UPPERCASE."""

    else:  # complexity_level == 3
        return f"""Generate {board_size} English words for a Codenames word game board with HIGH complexity.

Requirements:
- Include 4-5 thematic clusters with OVERLAPPING themes (words belong to multiple clusters)
- Include 6-8 polysemes with multiple semantic domains
- Include "trap words" that seem related to common themes but have dangerous secondary meanings
- Include "false friends" - word pairs that seem related but have different meanings
- Words should create strategic dilemmas for players

Examples of complex interactions:
- BANK: finance + nature (river bank) + action (bank shot)
- CURRENT: water + electricity + time (current events)
- CELL: biology + prison + phone + battery
- FALSE FRIENDS: SUIT and TIE (seem related but different domains)

Return ONLY a JSON object with this exact format:
{{
    "words": ["WORD1", "WORD2", ...],
    "clusters": [
        {{"theme": "main_theme", "words": ["W1", "W2", ...], "secondary_themes": ["alt1", "alt2"]}}
    ],
    "polysemes": ["BANK", "CELL", "CURRENT", ...],
    "false_friends": [["WORD1", "WORD2"], ["WORD3", "WORD4"]],
    "relations": {{
        "WORD": {{
            "related_words": ["W1", "W2"],
            "domains": ["domain1", "domain2", "domain3"],
            "relation_type": "polyseme",
            "trap_level": 2
        }}
    }},
    "assassin_traps": ["WORD_NEAR_ASSASSIN"]
}}

trap_level: 0=safe, 1=mild, 2=moderate, 3=dangerous

All words must be UPPERCASE."""


def _parse_llm_response(response_text: str, complexity_level: int) -> WordInteractions:
    """Parse the LLM response into WordInteractions.
    
    Raises:
        BoardGenerationError: If parsing fails
    """
    try:
        # Clean up response - extract JSON if wrapped in markdown
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            text = "\n".join(json_lines)
        
        data = json.loads(text)
        words = [w.upper() for w in data.get("words", [])]
        
        if len(words) < 25:
            raise BoardGenerationError(f"Not enough words generated: {len(words)} (need 25)")
        
        interactions = WordInteractions(words=words[:25])
        
        # Parse clusters
        for cluster_data in data.get("clusters", []):
            interactions.clusters.append(ThematicCluster(
                theme=cluster_data.get("theme", ""),
                words=[w.upper() for w in cluster_data.get("words", [])],
                secondary_themes=cluster_data.get("secondary_themes", []),
            ))
        
        # Parse relations
        for word, rel_data in data.get("relations", {}).items():
            word = word.upper()
            interactions.relations[word] = WordRelation(
                word=word,
                related_words=[w.upper() for w in rel_data.get("related_words", [])],
                relation_type=rel_data.get("relation_type", "semantic"),
                domains=rel_data.get("domains", []),
                trap_level=rel_data.get("trap_level", 0),
            )
        
        interactions.polysemes = [w.upper() for w in data.get("polysemes", [])]
        interactions.false_friends = [
            (pair[0].upper(), pair[1].upper()) 
            for pair in data.get("false_friends", []) 
            if len(pair) == 2
        ]
        interactions.assassin_traps = [w.upper() for w in data.get("assassin_traps", [])]
        
        return interactions
        
    except json.JSONDecodeError as e:
        raise BoardGenerationError(f"Failed to parse LLM response as JSON: {e}") from e
    except (KeyError, TypeError) as e:
        raise BoardGenerationError(f"Invalid LLM response format: {e}") from e


def generate_board(
    seed: int | None = None,
    complexity_level: int = 2,
    red_words: int = 9,
    blue_words: int = 8,
    neutral_words: int = 7,
    assassin_words: int = 1,
    model_name: str | None = None,
    temperature: float | None = None,
) -> BoardAssignment:
    """Generate a complete Codenames board with assignments.
    
    Args:
        seed: Random seed for reproducibility
        complexity_level: 1=basic, 2=medium, 3=complex
        red_words: Number of red team words
        blue_words: Number of blue team words
        neutral_words: Number of neutral words
        assassin_words: Number of assassin words
        model_name: (deprecated, ignored) Model configured via WATCHDOG_LLM_BACKEND
        temperature: (deprecated, ignored) Temperature configured via env vars
    
    Returns:
        BoardAssignment with words, team assignments, and semantic interactions
    
    Raises:
        BoardGenerationError: If board generation fails
    """
    if seed is not None:
        random.seed(seed)
    
    board_size = red_words + blue_words + neutral_words + assassin_words
    
    # Get LLM (local Qwen3 or Gemini based on WATCHDOG_LLM_BACKEND)
    llm = _get_llm()
    
    prompt = _build_generation_prompt(complexity_level, board_size)
    
    # Use dict messages — works with both local GamePlayModel and LangChain
    system_content = (
        "You are a word game designer creating boards for Codenames. "
        "Generate creative word lists with interesting semantic relationships. "
        "Respond only with the requested JSON format."
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    
    try:
        response = llm.invoke(messages)
        
        # Handle both string and list content (newer langchain versions return list for multimodal)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            # Extract text from list of content blocks
            response_text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        else:
            response_text = str(content)
        
        if not response_text.strip():
            raise BoardGenerationError("LLM returned empty response for board generation")
        
        interactions = _parse_llm_response(response_text, complexity_level)
        
    except BoardGenerationError:
        raise
    except Exception as e:
        raise BoardGenerationError(f"LLM generation failed: {e}") from e
    
    words = list(interactions.words)
    random.shuffle(words)
    
    # Assign words to teams
    assignments: dict[str, str] = {}
    
    # If we have assassin traps, try to make one the assassin
    assassin_candidates = [w for w in interactions.assassin_traps if w in words]
    if assassin_candidates:
        assassin_word = random.choice(assassin_candidates)
        words.remove(assassin_word)
        words.insert(0, assassin_word)  # Put at front to be assigned as assassin
    
    idx = 0
    for _ in range(assassin_words):
        assignments[words[idx]] = "assassin"
        idx += 1
    
    for _ in range(red_words):
        assignments[words[idx]] = "red"
        idx += 1
    
    for _ in range(blue_words):
        assignments[words[idx]] = "blue"
        idx += 1
    
    for _ in range(neutral_words):
        assignments[words[idx]] = "neutral"
        idx += 1
    
    # Shuffle the final word order for the grid
    random.shuffle(words)
    
    # Create 5x5 grid representation
    grid = []
    for i in range(5):
        row = words[i * 5:(i + 1) * 5]
        grid.append(row)
    
    return BoardAssignment(
        words=words,
        assignments=assignments,
        interactions=interactions,
        grid=grid,
    )


def regenerate_board_with_same_words(
    words: list[str],
    interactions: WordInteractions,
    seed: int | None = None,
    red_words: int = 9,
    blue_words: int = 8,
    neutral_words: int = 7,
    assassin_words: int = 1,
) -> BoardAssignment:
    """Regenerate team assignments for existing words."""
    if seed is not None:
        random.seed(seed)
    
    shuffled_words = list(words)
    random.shuffle(shuffled_words)
    
    assignments: dict[str, str] = {}
    idx = 0
    
    for _ in range(assassin_words):
        assignments[shuffled_words[idx]] = "assassin"
        idx += 1
    
    for _ in range(red_words):
        assignments[shuffled_words[idx]] = "red"
        idx += 1
    
    for _ in range(blue_words):
        assignments[shuffled_words[idx]] = "blue"
        idx += 1
    
    for _ in range(neutral_words):
        assignments[shuffled_words[idx]] = "neutral"
        idx += 1
    
    random.shuffle(shuffled_words)
    grid = []
    for i in range(5):
        grid.append(shuffled_words[i * 5:(i + 1) * 5])
    
    return BoardAssignment(
        words=shuffled_words,
        assignments=assignments,
        interactions=interactions,
        grid=grid,
    )
