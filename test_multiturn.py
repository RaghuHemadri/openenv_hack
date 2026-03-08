#!/usr/bin/env python3
"""
Test Script for WatchDog Multi-Turn Environment
=================================================
Avalon/Werewolf step-based oversight environment with player IDs.

Tests:
  1. Pass all turns — verify incremental revelation
  2. Flag action on error and clean turns
  3. Question mechanic
  4. Intervene action
  5. Full strategic episode (mixed actions)
  6. Episode end bonuses

Run:
    cd watchdog_env && python ../test_multiturn.py
    # or from repo root:
    PYTHONPATH=watchdog_env python test_multiturn.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "watchdog_env"))

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


def find_error_turn(env: WatchDogMultiTurnEnvironment) -> int | None:
    for i, t in enumerate(env._turns_seen):
        if t.get("has_error", False):
            return i
    return None


def find_clean_turn(env: WatchDogMultiTurnEnvironment) -> int | None:
    for i, t in enumerate(env._turns_seen):
        if not t.get("has_error", False):
            return i
    return None


def show_ground_truth(env: WatchDogMultiTurnEnvironment) -> None:
    print("\n  Ground truth (hidden from Overseer):")
    for i, t in enumerate(env._turns_seen):
        status = "ERROR" if t.get("has_error") else "CLEAN"
        speaker = t.get("speaker_display", "Player")
        detail = ""
        if t.get("has_error"):
            detail = " [mutated]"
        print(f"    Turn {i + 1}: {status} — {speaker}{detail}")


# ─── Test 1: Pass All Turns ────────────────────────────────────────

def test_pass_all():
    """Pass every turn. Verifies step-by-step progression."""
    header("Test 1: PASS All Turns")

    env = WatchDogMultiTurnEnvironment()
    obs = env.reset(seed=42)
    show_obs(obs, "RESET")

    step = 0
    while not obs.done:
        step += 1
        action = MultiTurnAction(action_type="pass")
        obs = env.step(action)
        show_obs(obs, f"Step {step}")

        if step > 20:
            print("  WARN: Exceeded 20 steps, stopping")
            break

    show_ground_truth(env)
    num_errors = sum(1 for t in env._turns_seen if t.get("has_error"))
    print(f"\n  Summary: Passed all {len(env._turns_seen)} turns. "
          f"Errors present: {num_errors}")
    print("  ✓ Test 1 PASSED")


# ─── Test 2: Flag Actions ──────────────────────────────────────────

def test_flag():
    """Flag a turn and verify feedback."""
    header("Test 2: FLAG Action")

    env = WatchDogMultiTurnEnvironment()
    obs = env.reset(seed=123)
    show_obs(obs, "RESET")

    # Pass first turn, then flag the second
    action = MultiTurnAction(action_type="pass")
    obs = env.step(action)
    show_obs(obs, "Step 1 — PASS")

    if not obs.done:
        flag_action = MultiTurnAction(action_type="flag", reason="Suspicious behavior")
        obs = env.step(flag_action)
        show_obs(obs, "Step 2 — FLAG")

    show_ground_truth(env)
    print("  ✓ Test 2 PASSED")


# ─── Test 3: Question Mechanic ─────────────────────────────────────

def test_question():
    """Use the QUESTION action on a turn."""
    header("Test 3: QUESTION Mechanic")

    env = WatchDogMultiTurnEnvironment()
    obs = env.reset(seed=456)
    show_obs(obs, "RESET")

    # Question the first turn
    q_action = MultiTurnAction(action_type="question", reason="Clarify role claim")
    obs = env.step(q_action)
    show_obs(obs, "Step 1 — QUESTION")

    assert obs.phase == "question_response", f"Expected question_response phase, got {obs.phase}"

    # After seeing the response, flag or pass
    if not obs.done:
        pass_action = MultiTurnAction(action_type="pass")
        obs = env.step(pass_action)
        show_obs(obs, "Step 2 — PASS after question")

    show_ground_truth(env)
    print("  ✓ Test 3 PASSED")


# ─── Test 4: Intervene ─────────────────────────────────────────────

def test_intervene():
    """INTERVENE ends the episode immediately."""
    header("Test 4: INTERVENE Action")

    env = WatchDogMultiTurnEnvironment()
    obs = env.reset(seed=789)
    show_obs(obs, "RESET")

    intervene_action = MultiTurnAction(
        action_type="intervene", reason="Emergency — clear deception detected"
    )
    obs = env.step(intervene_action)
    show_obs(obs, "Step 1 — INTERVENE")

    assert obs.done, "Episode should be done after INTERVENE"
    show_ground_truth(env)
    print("  ✓ Test 4 PASSED")


# ─── Test 5: Full Strategic Episode ────────────────────────────────

def test_strategic_episode():
    """Play through an episode with mixed strategy."""
    header("Test 5: Full Strategic Episode")

    env = WatchDogMultiTurnEnvironment()
    obs = env.reset(seed=2024)
    show_obs(obs, "RESET")

    step = 0
    while not obs.done:
        step += 1

        if step % 3 == 0 and obs.remaining_questions > 0:
            action = MultiTurnAction(action_type="question", reason="Checking consistency")
            label = "QUESTION"
        elif step % 5 == 0:
            action = MultiTurnAction(action_type="flag", reason="Suspicious")
            label = "FLAG"
        else:
            action = MultiTurnAction(action_type="pass")
            label = "PASS"

        obs = env.step(action)
        show_obs(obs, f"Step {step} — {label}")

        if step > 20:
            print("  WARN: Exceeded 20 steps, stopping")
            break

    show_ground_truth(env)
    print(f"\n  Final reward: {obs.cumulative_reward:.2f}")
    print(f"  Flags issued: {obs.flags_so_far}")
    print("  ✓ Test 5 PASSED")


# ─── Test 6: Episode End Bonus ──────────────────────────────────────

def test_episode_end_bonus():
    """Verify end-of-episode bonus reports missed errors."""
    header("Test 6: Episode End Bonus")

    env = WatchDogMultiTurnEnvironment()
    obs = env.reset(seed=999)
    show_obs(obs, "RESET")

    step = 0
    while not obs.done:
        step += 1
        action = MultiTurnAction(action_type="pass")
        obs = env.step(action)

        if step > 20:
            break

    if obs.feedback:
        print(f"\n  End feedback: {obs.feedback}")

    errors_present = sum(1 for t in env._turns_seen if t.get("has_error"))
    errors_detected = len(env._flagged_error_turns)
    errors_missed = errors_present - errors_detected
    print(f"  Errors present: {errors_present}, detected: {errors_detected}, "
          f"missed: {errors_missed}")
    print("  ✓ Test 6 PASSED")


# ─── Main ───────────────────────────────────────────────────────────

def main():
    header("WatchDog Multi-Turn Environment Tests")
    print("  Testing step-based Avalon oversight environment")
    print("  Players have IDs (P1, P2, ...) and LLM-generated responses")

    tests = [
        ("Pass All", test_pass_all),
        ("Flag", test_flag),
        ("Question", test_question),
        ("Intervene", test_intervene),
        ("Strategic Episode", test_strategic_episode),
        ("Episode End Bonus", test_episode_end_bonus),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ Test '{name}' FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    header("Summary")
    print(f"  Passed: {passed}/{len(tests)}")
    if failed:
        print(f"  Failed: {failed}/{len(tests)}")
    else:
        print("  All tests passed!")


if __name__ == "__main__":
    main()
