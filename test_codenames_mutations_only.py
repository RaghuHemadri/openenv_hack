#!/usr/bin/env python3
"""
Temporary test file for Test 18: Codenames with Mutations Enabled

Run:
    PYTHONPATH=watchdog_env python test_codenames_mutations_only.py
"""

import os
import sys

# Use LLM for Codenames
os.environ.pop("WATCHDOG_CODENAMES_USE_TEMPLATE", None)

_root = os.path.dirname(os.path.abspath(__file__))
_watchdog_env = os.path.join(_root, "watchdog_env")
sys.path.insert(0, _root)
sys.path.insert(0, _watchdog_env)

from server.watchdog_environment import WatchDogMultiTurnEnvironment
from models import MultiTurnAction, MultiTurnObservation


SEPARATOR = "=" * 70
THIN_SEP = "-" * 50


def header(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def show_obs(obs: MultiTurnObservation, step_label: str = "") -> None:
    prefix = f"[{step_label}] " if step_label else ""
    print(f"\n{prefix}Turn {obs.current_turn_number}/{obs.total_turns} "
          f"| Phase: {obs.phase} | Domain: {obs.task_domain} "
          f"| Difficulty: {obs.difficulty}")
    print(f"  Questions left: {obs.remaining_questions} | Flags: {obs.flags_so_far}")

    if obs.current_turn and obs.current_turn != "[Episode complete]":
        print(f"\n  {THIN_SEP}")
        for line in obs.current_turn.split("\n"):
            print(f"  | {line}")
        print(f"  {THIN_SEP}")

    if obs.feedback:
        print(f"  Feedback: {obs.feedback}")
    if obs.step_reward is not None:
        print(f"  Step reward: {obs.step_reward:+.2f}")
    if obs.cumulative_reward is not None:
        print(f"  Cumulative reward: {obs.cumulative_reward:+.2f}")
    if obs.done:
        print(f"  >>> EPISODE DONE <<<")


def show_ground_truth(env: WatchDogMultiTurnEnvironment) -> None:
    print("\n  Ground truth (hidden from Overseer):")
    for i, t in enumerate(env._turns_seen):
        status = "ERROR" if t.get("has_error") else "CLEAN"
        speaker = t.get("speaker_display", "Player")
        detail = " [mutated]" if t.get("has_error") else ""
        print(f"    Turn {i + 1}: {status} — {speaker}{detail}")
    num_errors = sum(1 for t in env._turns_seen if t.get("has_error"))
    if num_errors == 0:
        print("    (No mutations — all turns are clean)")


def test_codenames_with_mutations():
    """Codenames with use_mutations=True: mutations can occur, at least one per episode."""
    header("Test 18: Codenames with Mutations Enabled")

    try:
        from watchdog_env.plugins import get_plugin
        assert get_plugin("codenames") is not None
    except (ImportError, AssertionError):
        print("  Skipped: Codenames plugin not available")
        return

    env = WatchDogMultiTurnEnvironment(game_id="codenames", use_mutations=True, use_llm=True)
    obs = env.reset(seed=1)
    show_obs(obs, "RESET")
    assert obs.task_domain == "codenames", f"Expected task_domain=codenames, got {obs.task_domain}"

    step = 0
    while not obs.done:
        step += 1
        action = MultiTurnAction(action_type="pass")
        obs = env.step(action)
        show_obs(obs, f"Step {step}")

        if step > 30:
            print("  WARN: Exceeded 30 steps, stopping")
            break

    show_ground_truth(env)
    num_errors = sum(1 for t in env._turns_seen if t.get("has_error"))
    print(f"\n  Summary: {len(env._turns_seen)} turns, mutations: {num_errors}")
    assert num_errors >= 1, (
        "Codenames with mutations enabled should guarantee at least one mutation per episode"
    )
    print("  ✓ Test 18 PASSED")


if __name__ == "__main__":
    header("Test 18: Codenames with Mutations")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("  WARNING: GEMINI_API_KEY or GOOGLE_API_KEY not set")
        print("  This test requires the Gemini API key to run.")
        sys.exit(1)
    else:
        print("  API key found — running test")
    
    try:
        test_codenames_with_mutations()
        print("\n" + SEPARATOR)
        print("  TEST PASSED!")
        print(SEPARATOR)
    except Exception as e:
        print(f"\n  ✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
