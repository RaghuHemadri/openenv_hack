"""Word interaction definitions for complex semantic relationships in Codenames.

Defines thematic clusters, polysemes, false friends, and semantic traps
to increase game complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WordRelation:
    """Represents a semantic relationship between words."""
    
    word: str
    related_words: list[str] = field(default_factory=list)
    relation_type: str = "semantic"  # semantic, thematic, homonym, false_friend, trap
    domains: list[str] = field(default_factory=list)  # e.g., ["finance", "nature"]
    trap_level: int = 0  # 0=none, 1=mild, 2=moderate, 3=dangerous


@dataclass
class ThematicCluster:
    """A group of words sharing a hidden theme."""
    
    theme: str
    words: list[str]
    secondary_themes: list[str] = field(default_factory=list)


@dataclass
class WordInteractions:
    """Container for all word interactions on a board."""
    
    words: list[str]
    relations: dict[str, WordRelation] = field(default_factory=dict)
    clusters: list[ThematicCluster] = field(default_factory=list)
    polysemes: list[str] = field(default_factory=list)  # Words with multiple meanings
    false_friends: list[tuple[str, str]] = field(default_factory=list)  # Pairs that seem related but aren't
    assassin_traps: list[str] = field(default_factory=list)  # Words semantically close to assassin
    
    def get_related_words(self, word: str) -> list[str]:
        """Get all words related to the given word."""
        if word in self.relations:
            return self.relations[word].related_words
        return []
    
    def get_trap_level(self, word: str) -> int:
        """Get the trap level for a word (how close to assassin)."""
        if word in self.relations:
            return self.relations[word].trap_level
        return 0
    
    def get_word_domains(self, word: str) -> list[str]:
        """Get the semantic domains for a word."""
        if word in self.relations:
            return self.relations[word].domains
        return []
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "words": self.words,
            "relations": {
                w: {
                    "related_words": r.related_words,
                    "relation_type": r.relation_type,
                    "domains": r.domains,
                    "trap_level": r.trap_level,
                }
                for w, r in self.relations.items()
            },
            "clusters": [
                {"theme": c.theme, "words": c.words, "secondary_themes": c.secondary_themes}
                for c in self.clusters
            ],
            "polysemes": self.polysemes,
            "false_friends": self.false_friends,
            "assassin_traps": self.assassin_traps,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordInteractions":
        """Deserialize from dictionary."""
        interactions = cls(words=data.get("words", []))
        
        for word, rel_data in data.get("relations", {}).items():
            interactions.relations[word] = WordRelation(
                word=word,
                related_words=rel_data.get("related_words", []),
                relation_type=rel_data.get("relation_type", "semantic"),
                domains=rel_data.get("domains", []),
                trap_level=rel_data.get("trap_level", 0),
            )
        
        for cluster_data in data.get("clusters", []):
            interactions.clusters.append(ThematicCluster(
                theme=cluster_data.get("theme", ""),
                words=cluster_data.get("words", []),
                secondary_themes=cluster_data.get("secondary_themes", []),
            ))
        
        interactions.polysemes = data.get("polysemes", [])
        interactions.false_friends = [tuple(pair) for pair in data.get("false_friends", [])]
        interactions.assassin_traps = data.get("assassin_traps", [])
        
        return interactions
