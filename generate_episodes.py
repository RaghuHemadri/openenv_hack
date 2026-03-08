#!/usr/bin/env python3
"""
Generate Episodes — Unmutated vs Mutated
==========================================
Produces one clean (unmutated) and one mutated episode for Avalon, Cicero, or Codenames,
showing exactly what the overseer agent would see.

Usage:
    source ~/.venv/bin/activate
    cd /mnt/c/Users/rhvam/Desktop/not_os/openenv_hack
    PYTHONPATH=watchdog_env python generate_episodes.py
    
    # For different games:
    PYTHONPATH=watchdog_env python generate_episodes.py --game cicero
    PYTHONPATH=watchdog_env python generate_episodes.py --game codenames
"""

import argparse
import os
import sys
import random
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "watchdog_env"))

from watchdog_env.error_engine import maybe_mutate, start_episode

SEPARATOR = "=" * 72
THIN_SEP = "-" * 60


def generate_avalon_episode(seed: int, level: int, force_no_mutation: bool = False):
    """Run one Avalon episode and collect all turns."""
    from watchdog_env.envs.avalon import AvalonGame, LEVEL_CONFIG
    
    random.seed(seed)
    game = AvalonGame(level=level, seed=seed)
    game.reset(seed=seed)

    config = LEVEL_CONFIG.get(level, LEVEL_CONFIG[2])
    wolf_count = sum(1 for p in game.state.players if p.role == "Werewolf")
    start_episode(game_id="avalon", wolf_count=wolf_count, num_rounds=config["num_rounds"])

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
                game_id="avalon",
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


def generate_cicero_episode(seed: int, level: int, force_no_mutation: bool = False):
    """Run one Cicero episode and collect all turns."""
    from watchdog_env.plugins.cicero import CiceroPlugin, CiceroConfig
    
    random.seed(seed)
    plugin = CiceroPlugin()
    config = CiceroConfig(num_steps=5)
    plugin.reset(seed=seed, config=config)
    
    start_episode(game_id="cicero", num_steps=config.num_steps)
    
    turns = []
    step_index = 0
    while not plugin.get_state().done:
        step = plugin.generate_step(seed=seed + step_index, step_index=step_index)
        
        for turn in step.turns:
            clean_response = turn.action_text
            
            if force_no_mutation:
                displayed = clean_response
                has_error = False
                error_detail = None
            else:
                displayed, has_error, error_detail = maybe_mutate(
                    clean_response=clean_response,
                    speaker_role="",
                    level=level,
                    context={
                        "speaker_id": turn.agent_id,
                        "step_index": step_index,
                        "season": turn.metadata.get("season"),
                        "region": turn.metadata.get("region"),
                    },
                    game_id="cicero",
                )
            
            turns.append({
                "speaker": turn.display_name or turn.agent_id,
                "role": "power",
                "message": clean_response,
                "displayed_response": displayed,
                "has_error": has_error,
                "error_detail": error_detail,
                "moderator_prompt": f"Season: {turn.metadata.get('season', 'Unknown')}",
                "game_over": step.done,
            })
        
        step_index += 1
        if step_index > 10:
            break
    
    return turns


def generate_codenames_episode(seed: int, level: int, force_no_mutation: bool = False):
    """Run one Codenames episode and collect all turns."""
    from watchdog_env.plugins.codenames import CodenamesPlugin, CodenamesConfig
    
    random.seed(seed)
    plugin = CodenamesPlugin()
    config = CodenamesConfig(complexity_level=level, max_turns=15)
    plugin.reset(seed=seed, config=config)
    
    start_episode(game_id="codenames", num_turns=15)
    
    turns = []
    step_index = 0
    while not plugin.get_state().done:
        step = plugin.generate_step(seed=seed + step_index, step_index=step_index)
        
        for turn in step.turns:
            clean_response = turn.action_text
            
            if force_no_mutation:
                displayed = clean_response
                has_error = False
                error_detail = None
            else:
                displayed, has_error, error_detail = maybe_mutate(
                    clean_response=clean_response,
                    speaker_role="",
                    level=level,
                    context={
                        "speaker_id": turn.agent_id,
                        "step_index": step_index,
                        "phase": turn.metadata.get("phase"),
                        "team": turn.metadata.get("team"),
                        "role": turn.metadata.get("role"),
                    },
                    game_id="codenames",
                )
            
            turns.append({
                "speaker": turn.display_name or turn.agent_id,
                "role": turn.metadata.get("role", "player"),
                "message": clean_response,
                "displayed_response": displayed,
                "has_error": has_error,
                "error_detail": error_detail,
                "moderator_prompt": f"Phase: {turn.metadata.get('phase', 'Unknown')} | Team: {turn.metadata.get('team', 'Unknown')}",
                "game_over": step.done,
            })
        
        step_index += 1
        if step_index > 30:
            break
    
    return turns


def generate_episode(seed: int, level: int, force_no_mutation: bool = False, game_id: str = "avalon"):
    """Run one episode and collect all turns.

    Args:
        seed: Random seed for reproducibility
        level: Curriculum difficulty 1-4
        force_no_mutation: If True, skip the mutation layer entirely (clean episode)
        game_id: Game type - "avalon", "cicero", or "codenames"

    Returns:
        List of turn dicts, each containing the display info + ground truth.
    """
    if game_id == "cicero":
        return generate_cicero_episode(seed, level, force_no_mutation)
    elif game_id == "codenames":
        return generate_codenames_episode(seed, level, force_no_mutation)
    else:
        return generate_avalon_episode(seed, level, force_no_mutation)


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
    parser = argparse.ArgumentParser(description="Generate clean and mutated episodes")
    parser.add_argument("--game", default="avalon", choices=["avalon", "cicero", "codenames"],
                        help="Game type to generate episodes for")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--level", type=int, default=2, help="Difficulty level (1-4)")
    args = parser.parse_args()
    
    SEED = args.seed
    GAME_ID = args.game
    LEVEL = args.level

    # Use LLM for mutation injection, template for player responses (fast)
    os.environ["WATCHDOG_USE_LLM"] = "1"

    # Temporarily hide the key for fast generation
    _saved_key = os.environ.pop("GEMINI_API_KEY", None)
    _saved_gkey = os.environ.pop("GOOGLE_API_KEY", None)

    # ── Episode 1: Unmutated (clean) ────────────────────────────
    print("\n" + "▶" * 36)
    print(f"  GENERATING UNMUTATED {GAME_ID.upper()} EPISODE (all clean)")
    print("▶" * 36)

    clean_turns = generate_episode(seed=SEED, level=LEVEL, force_no_mutation=True, game_id=GAME_ID)
    print_episode(clean_turns, f"UNMUTATED {GAME_ID.upper()} EPISODE — All player responses are genuine",
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
    print("\n" + "▶" * 36)
    print(f"  GENERATING MUTATED {GAME_ID.upper()} EPISODE (with deception injection)")
    print("▶" * 36)

    mutated_turns = generate_episode(seed=SEED, level=LEVEL, force_no_mutation=False, game_id=GAME_ID)
    
    game_desc = {
        "avalon": "Some Werewolf turns contain injected deception",
        "cicero": "Some negotiation turns contain injected deception",
        "codenames": "Some clue/guess turns contain injected deception",
    }
    print_episode(mutated_turns,
                  f"MUTATED {GAME_ID.upper()} EPISODE — {game_desc.get(GAME_ID, 'Contains mutations')}",
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
    print(f"  CURRICULUM DIFFICULTY SCALING DEMO ({GAME_ID.upper()})")
    print(f"  (Shows how mutation difficulty increases with training episodes)")
    print(SEPARATOR)

    # Use template-only for curriculum demo (fast) 
    os.environ["WATCHDOG_USE_LLM"] = "0"
    from watchdog_env import error_engine
    error_engine._registry = None
    error_engine._mutator = None
    # Clear API keys to force template mode
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]

    # Get level config for the game
    if GAME_ID == "avalon":
        from watchdog_env.envs.avalon import LEVEL_CONFIG
    elif GAME_ID == "cicero":
        from watchdog_env.error_engine import CICERO_LEVEL_CONFIG as LEVEL_CONFIG
    else:
        from watchdog_env.error_engine import CODENAMES_LEVEL_CONFIG as LEVEL_CONFIG

    for level in [1, 2, 3]:
        lvl_seed = SEED + level * 100
        lvl_turns = generate_episode(seed=lvl_seed, level=level, force_no_mutation=False, game_id=GAME_ID)
        n_mutations = sum(1 for t in lvl_turns if t.get("has_error"))
        mutation_types = [
            t["error_detail"]["mutation_name"]
            for t in lvl_turns if t.get("has_error") and t.get("error_detail")
        ]
        difficulties = [
            t["error_detail"].get("difficulty", "?")
            for t in lvl_turns if t.get("has_error") and t.get("error_detail")
        ]
        
        config = LEVEL_CONFIG.get(level, LEVEL_CONFIG.get(2, {}))
        max_diff = config.get("max_difficulty", "?")
        clean_ratio = config.get("clean_ratio", "?")
        
        print(f"\n  Level {level} (max_difficulty={max_diff}, clean_ratio={clean_ratio})")
        print(f"    Mutations injected: {n_mutations}")
        if mutation_types:
            print(f"    Types: {', '.join(mutation_types)}")
            print(f"    Difficulties: {difficulties}")
        else:
            print(f"    (no mutations this run — try different seed)")

    print(f"\n{SEPARATOR}")
    print(f"  DONE — {GAME_ID.upper()} episodes generated successfully")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
