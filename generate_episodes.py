#!/usr/bin/env python3
"""
Generate Episodes — Unmutated vs Mutated
==========================================
Produces one clean (unmutated) and one mutated Avalon episode,
showing exactly what the overseer agent would see.

Usage:
    source ~/.venv/bin/activate
    cd /mnt/c/Users/rhvam/Desktop/not_os/openenv_hack
    PYTHONPATH=watchdog_env python generate_episodes.py
"""

import os
import sys
import random
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "watchdog_env"))

from watchdog_env.envs.avalon import AvalonGame, LEVEL_CONFIG
from watchdog_env.error_engine import maybe_mutate, start_episode

SEPARATOR = "=" * 72
THIN_SEP = "-" * 60


def generate_episode(seed: int, level: int, force_no_mutation: bool = False):
    """Run one Avalon episode and collect all turns.

    Args:
        seed: Random seed for reproducibility
        level: Curriculum difficulty 1-4
        force_no_mutation: If True, skip the mutation layer entirely (clean episode)

    Returns:
        List of turn dicts, each containing the display info + ground truth.
    """
    random.seed(seed)
    game = AvalonGame(level=level, seed=seed)
    game.reset(seed=seed)

    config = LEVEL_CONFIG.get(level, LEVEL_CONFIG[2])
    wolf_count = sum(1 for p in game.state.players if p.role == "Werewolf")
    start_episode(wolf_count, config["num_rounds"])

    turns = []
    while not game.is_done:
        turn = game.step()
        if turn.get("game_over"):
            turns.append({
                "speaker": turn.get("speaker_display", "Moderator"),
                "role": turn.get("role", "system"),
                "message": turn["message"],
                "displayed_response": turn["message"],
                "has_error": False,
                "error_detail": None,
                "moderator_prompt": turn.get("moderator_prompt", ""),
                "game_over": True,
            })
            break

        clean_response = turn["message"]

        if force_no_mutation:
            displayed = clean_response
            has_error = False
            error_detail = None
        else:
            displayed, has_error, error_detail = maybe_mutate(
                clean_response=clean_response,
                speaker_role=turn["role"],
                level=level,
                context={
                    "speaker_id": turn.get("speaker_id"),
                    "speaker_name": turn.get("speaker_name"),
                    "day": turn.get("day"),
                    "phase": turn.get("phase"),
                },
            )

        turns.append({
            "speaker": turn.get("speaker_display", "Player"),
            "role": turn.get("role", "unknown"),
            "message": clean_response,
            "displayed_response": displayed,
            "has_error": has_error,
            "error_detail": error_detail,
            "moderator_prompt": turn.get("moderator_prompt", ""),
            "game_over": False,
        })

    return turns


def print_episode(turns, title, show_ground_truth=False):
    """Pretty-print an episode."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

    for i, t in enumerate(turns):
        if t.get("game_over"):
            print(f"\n  [GAME OVER] {t['message']}")
            break

        print(f"\n  Turn {i + 1}  |  {t['speaker']}  |  Role: {t['role']}")
        print(f"  {THIN_SEP}")

        if t["moderator_prompt"]:
            for line in t["moderator_prompt"].split("\n"):
                print(f"  [Moderator]: {line}")
            print()

        for line in t["displayed_response"].split("\n"):
            print(f"  [{t['speaker']}]: {line}")

        if show_ground_truth:
            if t["has_error"]:
                detail = t["error_detail"] or {}
                print(f"  >>> MUTATED — {detail.get('mutation_name', 'unknown')} "
                      f"(difficulty {detail.get('difficulty', '?')}, "
                      f"type: {detail.get('type', '?')})")
                if detail.get("description"):
                    print(f"      Description: {detail['description']}")
                if detail.get("original"):
                    print(f"      Original:  {detail['original'][:120]}")
                if detail.get("corrupted"):
                    print(f"      Corrupted: {detail['corrupted'][:120]}")
            else:
                print(f"  >>> CLEAN")

    n_errors = sum(1 for t in turns if t.get("has_error"))
    n_total = len([t for t in turns if not t.get("game_over")])
    print(f"\n  {THIN_SEP}")
    print(f"  Total turns: {n_total}  |  Mutations: {n_errors}")
    print(SEPARATOR)


def main():
    SEED = 42

    # Use LLM for mutation injection, template for player responses (fast)
    # Keep API key available for the mutation LLM backend
    os.environ["WATCHDOG_USE_LLM"] = "1"

    # Force template fallback for player response generation (fast, no LLM needed)
    from watchdog_env.envs import avalon as _avalon_mod
    _avalon_mod._llm_instance = None
    # Temporarily hide the key so langchain doesn't try to init (slow)
    _saved_key = os.environ.pop("GEMINI_API_KEY", None)
    _saved_gkey = os.environ.pop("GOOGLE_API_KEY", None)
    # Import triggers _load_dotenv, so we need to prevent re-load
    # Player LLM uses langchain which needs the key at init time
    # We'll restore it right after the avalon singleton is set

    # ── Episode 1: Unmutated (clean) ────────────────────────────
    print("\n" + "▶" * 36)
    print("  GENERATING UNMUTATED EPISODE (all clean)")
    print("▶" * 36)

    clean_turns = generate_episode(seed=SEED, level=2, force_no_mutation=True)
    print_episode(clean_turns, "UNMUTATED EPISODE — All player responses are genuine",
                  show_ground_truth=True)

    # Restore API keys for LLM-based mutation injection
    if _saved_key:
        os.environ["GEMINI_API_KEY"] = _saved_key
    if _saved_gkey:
        os.environ["GOOGLE_API_KEY"] = _saved_gkey

    # Reset the mutation engine so it picks up the API key
    from watchdog_env import error_engine
    error_engine._registry = None
    error_engine._mutator = None

    # ── Episode 2: Mutated (with deception injection) ───────────
    # Run at difficulty 2 (moderate) so mutations are non-trivial
    print("\n" + "▶" * 36)
    print("  GENERATING MUTATED EPISODE (with deception injection)")
    print("▶" * 36)

    mutated_turns = generate_episode(seed=SEED, level=2, force_no_mutation=False)
    print_episode(mutated_turns,
                  "MUTATED EPISODE — Some Werewolf turns contain injected deception",
                  show_ground_truth=True)

    # ── Side-by-side diff for mutated turns ─────────────────────
    print(f"\n{SEPARATOR}")
    print("  SIDE-BY-SIDE COMPARISON (mutated turns only)")
    print(SEPARATOR)

    for i, (c, m) in enumerate(zip(clean_turns, mutated_turns)):
        if m.get("has_error"):
            print(f"\n  Turn {i + 1}: {m['speaker']} ({m['role']})")
            print(f"  {THIN_SEP}")
            print(f"  CLEAN:   {c['message'][:200]}")
            print(f"  MUTATED: {m['displayed_response'][:200]}")
            detail = m.get("error_detail", {})
            print(f"  TYPE: {detail.get('mutation_name', '?')} "
                  f"(difficulty {detail.get('difficulty', '?')})")
            print()

    # ── Curriculum demo ─────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  CURRICULUM DIFFICULTY SCALING DEMO")
    print(f"  (Shows how mutation difficulty increases with training episodes)")
    print(SEPARATOR)

    # Use template-only for curriculum demo (fast) 
    os.environ["WATCHDOG_USE_LLM"] = "0"
    from watchdog_env import error_engine
    error_engine._registry = None
    error_engine._mutator = None
    # Also reset player LLM to avoid slow API calls
    _avalon_mod._llm_instance = None
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]

    for level in [1, 2, 3]:
        lvl_seed = SEED + level * 100
        lvl_turns = generate_episode(seed=lvl_seed, level=level, force_no_mutation=False)
        n_mutations = sum(1 for t in lvl_turns if t.get("has_error"))
        mutation_types = [
            t["error_detail"]["mutation_name"]
            for t in lvl_turns if t.get("has_error") and t.get("error_detail")
        ]
        difficulties = [
            t["error_detail"].get("difficulty", "?")
            for t in lvl_turns if t.get("has_error") and t.get("error_detail")
        ]
        print(f"\n  Level {level} (max_difficulty={LEVEL_CONFIG[level]['max_difficulty']}, "
              f"clean_ratio={LEVEL_CONFIG[level]['clean_ratio']})")
        print(f"    Mutations injected: {n_mutations}")
        if mutation_types:
            print(f"    Types: {', '.join(mutation_types)}")
            print(f"    Difficulties: {difficulties}")
        else:
            print(f"    (no mutations this run — try different seed)")

    print(f"\n{SEPARATOR}")
    print("  DONE — Episodes generated successfully")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
