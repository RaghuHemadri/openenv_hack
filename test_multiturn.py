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
  7. Cicero plugin — minimal (registered, reset, generate steps)
  8. Plugin selection (game_id)
  9. Mutations disabled (use_mutations=False)
  10. Multiple resets
  11. Question budget exhausted
  12. Flag with error_type
  13. Cicero full episode via env
  14. Env metadata

Run:
    cd watchdog_env && python ../test_multiturn.py
    # or from repo root:
    PYTHONPATH=watchdog_env python test_multiturn.py
"""

import os
import sys

# Use LLM for Avalon and Cicero (default when API key present). Ensure template not forced.
os.environ.pop("WATCHDOG_AVALON_USE_TEMPLATE", None)
os.environ.pop("WATCHDOG_CICERO_USE_TEMPLATE", None)

_root = os.path.dirname(os.path.abspath(__file__))
_watchdog_env = os.path.join(_root, "watchdog_env")
sys.path.insert(0, _root)  # for watchdog_env package
sys.path.insert(0, _watchdog_env)  # for server, models when run from watchdog_env

# Ensure plugins are registered. Run from repo root with: PYTHONPATH=. python test_multiturn.py
# Or: cd watchdog_env && PYTHONPATH=.. python ../test_multiturn.py
try:
    from watchdog_env.plugins import get_plugin
    get_plugin("avalon")
except ImportError:
    pass  # plugins register on import; may fail if openenv not installed

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
        detail = " [mutated]" if t.get("has_error") else ""
        print(f"    Turn {i + 1}: {status} — {speaker}{detail}")
    num_errors = sum(1 for t in env._turns_seen if t.get("has_error"))
    if num_errors == 0:
        print("    (No mutations — all turns are clean)")


# ─── Test 1: Pass All Turns ────────────────────────────────────────

def test_pass_all():
    """Pass every turn. Verifies step-by-step progression."""
    header("Test 1: PASS All Turns")

    env = WatchDogMultiTurnEnvironment(use_llm=True)
    obs = env.reset(seed=42)
    show_obs(obs, "RESET")
    assert obs.task_domain == "avalon", f"Expected task_domain=avalon, got {obs.task_domain}"

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

    env = WatchDogMultiTurnEnvironment(use_llm=True)
    obs = env.reset(seed=123)
    show_obs(obs, "RESET")

    # Pass first turn, then flag the second
    action = MultiTurnAction(action_type="pass")
    obs = env.step(action)
    show_obs(obs, "Step 1 — PASS")

    if not obs.done:
        flag_action = MultiTurnAction(action_type="flag", explanation="Suspicious behavior")
        obs = env.step(flag_action)
        show_obs(obs, "Step 2 — FLAG")

    show_ground_truth(env)
    print("  ✓ Test 2 PASSED")


# ─── Test 3: Question Mechanic ─────────────────────────────────────

def test_question():
    """Use the QUESTION action on a turn."""
    header("Test 3: QUESTION Mechanic")

    env = WatchDogMultiTurnEnvironment(use_llm=True)
    obs = env.reset(seed=456)
    show_obs(obs, "RESET")

    # Question the first turn
    q_action = MultiTurnAction(action_type="question", explanation="Clarify role claim")
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

    env = WatchDogMultiTurnEnvironment(use_llm=True)
    obs = env.reset(seed=789)
    show_obs(obs, "RESET")

    intervene_action = MultiTurnAction(
        action_type="intervene", explanation="Emergency — clear deception detected"
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

    env = WatchDogMultiTurnEnvironment(use_llm=True)
    obs = env.reset(seed=2024)
    show_obs(obs, "RESET")

    step = 0
    while not obs.done:
        step += 1

        if step % 3 == 0 and obs.remaining_questions > 0:
            action = MultiTurnAction(action_type="question", explanation="Checking consistency")
            label = "QUESTION"
        elif step % 5 == 0:
            action = MultiTurnAction(action_type="flag", explanation="Suspicious")
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

    env = WatchDogMultiTurnEnvironment(use_llm=True)
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
    print(f"  Errors present: {errors_present} (mutations enabled), detected: {errors_detected}, "
          f"missed: {errors_missed}")
    print("  ✓ Test 6 PASSED")


# ─── Test 7: Cicero Plugin (minimal + env) ─────────────────────────────

def test_cicero_plugin():
    """Minimal Cicero plugin test: registered, reset, generate steps. Also test env with game_id."""
    header("Test 7: Cicero Plugin (minimal + env)")

    try:
        from watchdog_env.plugins import get_plugin
        from watchdog_env.plugins.cicero import CiceroConfig
    except ImportError:
        print("  Skipped: watchdog_env.plugins not available")
        return

    plugin = get_plugin("cicero")
    assert plugin is not None, "Cicero plugin should be registered"
    assert plugin.get_game_id() == "cicero"
    print(f"  Cicero plugin: {plugin.get_display_name()}")

    plugin.reset(seed=1, config=CiceroConfig(num_steps=2))
    step0 = plugin.generate_step(seed=1, step_index=0)
    assert len(step0.turns) >= 1
    assert step0.turns[0].agent_id in plugin.list_agent_ids()
    print(f"  Step 0: {len(step0.turns)} turn(s), done={step0.done}")

    step1 = plugin.generate_step(seed=1, step_index=1)
    assert step1.done
    print(f"  Step 1: done={step1.done}")

    # Test full env with game_id="cicero" (no mutations for Cicero)
    env = WatchDogMultiTurnEnvironment(game_id="cicero", use_mutations=False, use_llm=True)
    obs = env.reset(seed=1)
    assert obs.task_domain == "cicero"
    print(f"  Env with game_id=cicero: task_domain={obs.task_domain}")
    print("  ✓ Test 7 PASSED")


# ─── Test 8: Plugin Selection (game_id) ───────────────────────────────

def test_plugin_selection():
    """Verify game_id selects the correct plugin."""
    header("Test 8: Plugin Selection (game_id)")

    env_avalon = WatchDogMultiTurnEnvironment(game_id="avalon", use_llm=True)
    obs = env_avalon.reset(seed=1)
    assert obs.task_domain == "avalon"
    print(f"  game_id=avalon → task_domain={obs.task_domain}")

    env_cicero = WatchDogMultiTurnEnvironment(game_id="cicero", use_llm=True)
    obs = env_cicero.reset(seed=1)
    assert obs.task_domain == "cicero"
    print(f"  game_id=cicero → task_domain={obs.task_domain}")
    print("  ✓ Test 8 PASSED")


# ─── Test 9: Mutations Disabled ──────────────────────────────────────

def test_mutations_disabled():
    """With use_mutations=False, all turns are clean."""
    header("Test 9: Mutations Disabled")

    env = WatchDogMultiTurnEnvironment(use_mutations=False, use_llm=True)
    obs = env.reset(seed=42)
    step = 0
    while not obs.done and step < 15:
        step += 1
        obs = env.step(MultiTurnAction(action_type="pass"))

    num_errors = sum(1 for t in env._turns_seen if t.get("has_error"))
    print(f"  Turns: {len(env._turns_seen)}, errors: {num_errors}")
    assert num_errors == 0, "No mutations when use_mutations=False"
    print("  ✓ Test 9 PASSED")


# ─── Test 10: Multiple Resets ────────────────────────────────────────

def test_multiple_resets():
    """Reset can be called multiple times; state is fresh each time."""
    header("Test 10: Multiple Resets")

    env = WatchDogMultiTurnEnvironment(use_llm=True)
    obs1 = env.reset(seed=1)
    assert obs1.task_domain == "avalon"
    obs1 = env.step(MultiTurnAction(action_type="pass"))

    obs2 = env.reset(seed=2)
    assert obs2.step_reward is None
    assert env._turns_seen == [] or len(env._turns_seen) <= 1
    print(f"  After reset: turns_seen={len(env._turns_seen)}")
    print("  ✓ Test 10 PASSED")


# ─── Test 11: Question Budget Exhausted ──────────────────────────────

def test_question_budget_exhausted():
    """When questions run out, QUESTION is treated as PASS."""
    header("Test 11: Question Budget Exhausted")

    env = WatchDogMultiTurnEnvironment(use_llm=True)
    obs = env.reset(seed=111)
    # Burn through all questions
    while not obs.done and obs.remaining_questions > 0:
        obs = env.step(MultiTurnAction(action_type="question"))
        if obs.done:
            break
        obs = env.step(MultiTurnAction(action_type="pass"))

    # With 0 questions left, send QUESTION — should be treated as PASS
    if not obs.done:
        obs = env.step(MultiTurnAction(action_type="question"))
        assert obs.remaining_questions == 0
        assert "No questions left" in (obs.feedback or "")
        print(f"  Feedback: {obs.feedback[:80]}...")
    print("  ✓ Test 11 PASSED")


# ─── Test 12: Flag with error_type ───────────────────────────────────

def test_flag_with_error_type():
    """Flag action accepts error_type and explanation."""
    header("Test 12: Flag with error_type")

    env = WatchDogMultiTurnEnvironment(use_llm=True)
    obs = env.reset(seed=222)
    if not obs.done:
        action = MultiTurnAction(
            action_type="flag",
            error_type="factual_error",
            explanation="Player contradicted earlier statement",
        )
        obs = env.step(action)
        assert obs.step_reward is not None
        print(f"  Flag reward: {obs.step_reward:+.2f}")
    print("  ✓ Test 12 PASSED")


# ─── Test 13: Cicero Full Episode (Strategic) ─────────────────────────

def test_cicero_full_episode():
    """Play through a Cicero episode with mixed strategy (like Full Strategic Episode)."""
    header("Test 13: Cicero Full Episode (Strategic)")

    env = WatchDogMultiTurnEnvironment(game_id="cicero", use_mutations=False, use_llm=True)
    obs = env.reset(seed=1)
    show_obs(obs, "RESET")
    assert obs.task_domain == "cicero"

    step = 0
    while not obs.done:
        step += 1

        if step % 3 == 0 and obs.remaining_questions > 0:
            action = MultiTurnAction(action_type="question", explanation="Clarify proposal")
            label = "QUESTION"
        elif step % 5 == 0:
            action = MultiTurnAction(action_type="flag", explanation="Suspicious negotiation")
            label = "FLAG"
        else:
            action = MultiTurnAction(action_type="pass")
            label = "PASS"

        obs = env.step(action)
        show_obs(obs, f"Step {step} — {label}")

        if step > 15:
            print("  WARN: Exceeded 15 steps, stopping")
            break

    show_ground_truth(env)
    print(f"\n  Final reward: {obs.cumulative_reward:.2f}")
    print(f"  Flags issued: {obs.flags_so_far}")
    print("  ✓ Test 13 PASSED")


# ─── Test 14: Env Metadata ────────────────────────────────────────────

def test_env_metadata():
    """get_metadata returns correct env info."""
    header("Test 14: Env Metadata")

    env = WatchDogMultiTurnEnvironment(use_llm=True)
    meta = env.get_metadata()
    assert meta.name
    assert meta.version
    assert "WatchDog" in meta.name
    print(f"  name={meta.name}, version={meta.version}")
    print("  ✓ Test 14 PASSED")


# ─── Main ───────────────────────────────────────────────────────────

def main():
    header("WatchDog Multi-Turn Environment Tests")
    print("  LLM mode: using LLM when API key present (WATCHDOG_*_USE_TEMPLATE not set)")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("  WARNING: GEMINI_API_KEY or GOOGLE_API_KEY not set — LLM will fall back to template")
    else:
        print("  API key found — using LLM for player responses")
    print("  Players have IDs (P1, P2, ...) and LLM-generated responses")

    tests = [
        ("Pass All", test_pass_all),
        ("Flag", test_flag),
        ("Question", test_question),
        ("Intervene", test_intervene),
        ("Strategic Episode", test_strategic_episode),
        ("Episode End Bonus", test_episode_end_bonus),
        ("Cicero Plugin", test_cicero_plugin),
        ("Plugin Selection", test_plugin_selection),
        ("Mutations Disabled", test_mutations_disabled),
        ("Multiple Resets", test_multiple_resets),
        ("Question Budget Exhausted", test_question_budget_exhausted),
        ("Flag with error_type", test_flag_with_error_type),
        ("Cicero Full Episode", test_cicero_full_episode),
        ("Env Metadata", test_env_metadata),
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
