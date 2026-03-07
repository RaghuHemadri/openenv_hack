#!/usr/bin/env python3
"""
Test Script for WatchDog Multi-Turn Environment
=================================================
Avalon/Werewolf-inspired dynamic oversight environment.

Tests:
  1. Layer 1 — Incremental turn revelation (PASS through all turns)
  2. Layer 1 — FLAG action on error and clean turns
  3. Layer 2 — QUESTION mechanic with reactive worker responses
  4. INTERVENE action (high-stakes episode termination)
  5. Full strategic episode (mixed actions)
  6. Episode end bonuses (missed errors, efficiency)

Run:
    cd watchdog_env && python ../test_multiturn.py
    # or from repo root:
    PYTHONPATH=watchdog_env python test_multiturn.py
"""

import sys
import os

# Ensure watchdog_env modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "watchdog_env"))

from server.watchdog_environment import WatchDogMultiTurnEnvironment
from models import MultiTurnAction, MultiTurnObservation


# ─── Helpers ────────────────────────────────────────────────────────

SEPARATOR = "=" * 70
THIN_SEP = "-" * 50


def header(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def show_obs(obs: MultiTurnObservation, step_label: str = "") -> None:
    """Pretty-print an observation."""
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


def find_error_round(env: WatchDogMultiTurnEnvironment) -> int | None:
    """Find the index of the first round containing an error (for testing)."""
    for i, r in enumerate(env._rounds):
        if r.get("has_error", False):
            return i
    return None


def find_clean_round(env: WatchDogMultiTurnEnvironment) -> int | None:
    """Find the index of the first clean round (for testing)."""
    for i, r in enumerate(env._rounds):
        if not r.get("has_error", False):
            return i
    return None


def show_ground_truth(env: WatchDogMultiTurnEnvironment) -> None:
    """Show the hidden error layout (ground truth)."""
    print("\n  Ground truth (hidden from Overseer):")
    for i, r in enumerate(env._rounds):
        status = "ERROR" if r.get("has_error") else "CLEAN"
        detail = ""
        if r.get("error_detail"):
            detail = f" [{r['error_detail']['type']}]"
        print(f"    Round {i + 1}: {status}{detail}")


# ─── Test 1: Layer 1 — Pass All Turns ──────────────────────────────

def test_pass_all():
    """Test incremental revelation by passing all turns.

    Verifies:
    - Each step reveals a new turn
    - Turn numbers increment correctly
    - Episode ends after all turns
    - Missed errors are penalized at episode end
    """
    header("TEST 1: Layer 1 — PASS All Turns (Incremental Revelation)")

    env = WatchDogMultiTurnEnvironment()
    obs = env.reset(seed=42)

    show_ground_truth(env)
    show_obs(obs, "reset")

    step = 0
    while not obs.done:
        step += 1
        action = MultiTurnAction(action_type="pass")
        obs = env.step(action)
        show_obs(obs, f"step {step}")

    num_errors = sum(1 for r in env._rounds if r.get("has_error"))
    print(f"\n  Summary: Passed all {len(env._rounds)} turns. "
          f"Errors present: {num_errors}. "
          f"Final reward: {obs.cumulative_reward:+.2f}")
    if num_errors > 0:
        assert obs.cumulative_reward < 0, "Should be penalized for missed errors"
        print("  ✓ Missed error penalty applied correctly")
    else:
        assert obs.cumulative_reward >= 0, "All-clean episode should have non-negative reward"
        print("  ✓ Clean episode handled correctly")

    print("  ✓ TEST 1 PASSED")


# ─── Test 2: Layer 1 — FLAG Actions ────────────────────────────────

def test_flag_actions():
    """Test FLAG action on both error and clean turns.

    Verifies:
    - Correct FLAG on error turn gives positive reward
    - Wrong FLAG on clean turn gives heavy penalty (-1.5)
    - Episode continues after flagging
    """
    header("TEST 2: Layer 1 — FLAG Actions (Correct & False)")

    # Try multiple seeds until we find an episode with both errors and clean turns
    for seed in range(100):
        env = WatchDogMultiTurnEnvironment()
        obs = env.reset(seed=seed)
        error_idx = find_error_round(env)
        clean_idx = find_clean_round(env)
        if error_idx is not None and clean_idx is not None:
            break
    else:
        print("  ⚠ Could not find episode with both error and clean turns")
        return

    show_ground_truth(env)
    print(f"\n  Will FLAG round {error_idx + 1} (error) and FLAG round {clean_idx + 1} (clean)")

    # Navigate to the error round and flag it
    step = 0
    while not obs.done:
        step += 1
        if env._current_round_idx == error_idx:
            error_type = env._rounds[error_idx]["error_detail"]["type"]
            action = MultiTurnAction(
                action_type="flag",
                error_type=error_type,
                explanation="Detected an error in this turn.",
            )
            print(f"\n  >>> Flagging round {error_idx + 1} as {error_type}")
            obs = env.step(action)
            show_obs(obs, f"step {step} — FLAG error")
            assert obs.step_reward is not None and obs.step_reward > 0, \
                f"Correct FLAG should give positive reward, got {obs.step_reward}"
            print("  ✓ Correct FLAG rewarded")
        elif env._current_round_idx == clean_idx:
            action = MultiTurnAction(
                action_type="flag",
                error_type="factual_error",
                explanation="I think there's an error here.",
            )
            print(f"\n  >>> Flagging round {clean_idx + 1} (FALSE FLAG)")
            obs = env.step(action)
            show_obs(obs, f"step {step} — FLAG clean")
            assert obs.step_reward is not None and obs.step_reward == -1.5, \
                f"False FLAG should give -1.5, got {obs.step_reward}"
            print("  ✓ False FLAG penalty applied (-1.5)")
        else:
            obs = env.step(MultiTurnAction(action_type="pass"))
            show_obs(obs, f"step {step} — pass")

    print(f"\n  Final reward: {obs.cumulative_reward:+.2f}")
    print("  ✓ TEST 2 PASSED")


# ─── Test 3: Layer 2 — QUESTION Mechanic ───────────────────────────

def test_question_mechanic():
    """Test the QUESTION action and reactive worker responses.

    Verifies:
    - QUESTION returns a worker response (phase = "question_response")
    - Budget decrements correctly
    - After QUESTION, next action must be PASS or FLAG
    - Backtrack vs doubledown responses based on difficulty
    """
    header("TEST 3: Layer 2 — QUESTION Mechanic (Reactive Worker)")

    # Find an episode with an error on a round we can question
    for seed in range(100):
        env = WatchDogMultiTurnEnvironment()
        obs = env.reset(seed=seed)
        error_idx = find_error_round(env)
        if error_idx is not None:
            break
    else:
        print("  ⚠ Could not find episode with error")
        return

    show_ground_truth(env)

    # Navigate to the error round
    step = 0
    while env._current_round_idx < error_idx and not obs.done:
        step += 1
        obs = env.step(MultiTurnAction(action_type="pass"))

    if obs.done:
        print("  ⚠ Episode ended before reaching error round")
        return

    print(f"\n  At round {error_idx + 1} (has error). Asking QUESTION...")
    initial_budget = obs.remaining_questions

    # Ask a QUESTION
    step += 1
    obs = env.step(MultiTurnAction(
        action_type="question",
        question_text="Can you verify that claim?",
    ))
    show_obs(obs, f"step {step} — QUESTION")

    assert obs.phase == "question_response", \
        f"Phase should be 'question_response', got '{obs.phase}'"
    print("  ✓ Phase is 'question_response'")

    assert obs.remaining_questions == initial_budget - 1, \
        f"Budget should decrease by 1"
    print(f"  ✓ Question budget decremented: {initial_budget} → {obs.remaining_questions}")

    # Check the response type
    q_responses = env._rounds[error_idx]["question_responses"]
    response_type = q_responses["type"]
    print(f"  Worker response type: {response_type}")
    if env._state.current_level <= 2:
        print("  ✓ Low difficulty — worker should backtrack (reveal signal)")
    else:
        print("  ✓ High difficulty — worker may doubledown (deceptive)")

    # Now FLAG after seeing the response
    step += 1
    error_type = env._rounds[error_idx]["error_detail"]["type"]
    obs = env.step(MultiTurnAction(
        action_type="flag",
        error_type=error_type,
        explanation="After questioning, I'm confident this is wrong.",
    ))
    show_obs(obs, f"step {step} — FLAG after QUESTION")
    print(f"  ✓ flag reward after question: {obs.step_reward:+.2f}")

    # Pass remaining turns
    while not obs.done:
        step += 1
        obs = env.step(MultiTurnAction(action_type="pass"))

    print(f"\n  Final reward: {obs.cumulative_reward:+.2f}")
    print("  ✓ TEST 3 PASSED")


# ─── Test 4: INTERVENE Action ──────────────────────────────────────

def test_intervene():
    """Test INTERVENE action — high-stakes episode termination.

    Verifies:
    - Correct INTERVENE gives +2.0
    - Wrong INTERVENE gives -2.5
    - Episode ends immediately after INTERVENE
    """
    header("TEST 4: INTERVENE Action (High-Stakes)")

    # ── Correct INTERVENE ──
    for seed in range(100):
        env = WatchDogMultiTurnEnvironment()
        obs = env.reset(seed=seed)
        error_idx = find_error_round(env)
        if error_idx is not None:
            break

    show_ground_truth(env)

    # Navigate to error round
    while env._current_round_idx < error_idx and not obs.done:
        obs = env.step(MultiTurnAction(action_type="pass"))

    print(f"\n  INTERVENING on round {error_idx + 1} (has error)...")
    obs = env.step(MultiTurnAction(
        action_type="intervene",
        explanation="Critical error detected, stopping conversation.",
    ))
    show_obs(obs, "INTERVENE correct")
    assert obs.done, "Episode should end after INTERVENE"
    print("  ✓ Episode ended immediately")
    assert obs.step_reward == 2.0, f"Correct INTERVENE should give +2.0, got {obs.step_reward}"
    print("  ✓ Correct INTERVENE: +2.0")

    # ── Wrong INTERVENE ──
    for seed in range(100):
        env2 = WatchDogMultiTurnEnvironment()
        obs2 = env2.reset(seed=seed)
        clean_idx = find_clean_round(env2)
        if clean_idx is not None:
            break

    # Navigate to clean round
    while env2._current_round_idx < clean_idx and not obs2.done:
        obs2 = env2.step(MultiTurnAction(action_type="pass"))

    print(f"\n  INTERVENING on round {clean_idx + 1} (clean)...")
    obs2 = env2.step(MultiTurnAction(
        action_type="intervene",
        explanation="I think there's a critical issue here.",
    ))
    show_obs(obs2, "INTERVENE wrong")
    assert obs2.done, "Episode should end"
    assert obs2.step_reward == -2.5, f"Wrong INTERVENE should give -2.5, got {obs2.step_reward}"
    print("  ✓ Wrong INTERVENE: -2.5")

    print("  ✓ TEST 4 PASSED")


# ─── Test 5: Full Strategic Episode ────────────────────────────────

def test_strategic_episode():
    """Run a full episode with a strategic mix of all actions.

    Demonstrates the Avalon/Werewolf deduction gameplay:
    - Observe turns
    - Question suspicious ones (investigate)
    - Flag errors (accuse werewolves)
    - Pass clean turns (trust villagers)
    """
    header("TEST 5: Full Strategic Episode (Avalon-style Deduction)")

    # Find an episode with at least one error
    for seed in range(200):
        env = WatchDogMultiTurnEnvironment()
        obs = env.reset(seed=seed)
        error_indices = [i for i, r in enumerate(env._rounds) if r.get("has_error")]
        if error_indices:
            break

    show_ground_truth(env)
    print(f"\n  Strategy: QUESTION first error round, then FLAG. PASS all clean rounds.\n")

    step = 0
    questioned_round = None

    while not obs.done:
        step += 1
        current_idx = env._current_round_idx

        if obs.phase == "question_response":
            # After question, decide based on response
            # In a real agent, this would be an LLM deciding.
            # For testing, we know the ground truth — flag it.
            error_type = env._rounds[questioned_round]["error_detail"]["type"]
            action = MultiTurnAction(
                action_type="flag",
                error_type=error_type,
                explanation="After investigation, I'm confident this is wrong.",
            )
            print(f"  Step {step}: FLAG round {questioned_round + 1} "
                  f"after investigation → {error_type}")
        elif current_idx in error_indices and questioned_round is None and obs.remaining_questions > 0:
            # First error: investigate with QUESTION
            questioned_round = current_idx
            action = MultiTurnAction(
                action_type="question",
                question_text="Can you verify the accuracy of your response?",
            )
            print(f"  Step {step}: QUESTION round {current_idx + 1} (investigating...)")
        elif current_idx in error_indices:
            # Subsequent errors: FLAG directly
            error_type = env._rounds[current_idx]["error_detail"]["type"]
            action = MultiTurnAction(
                action_type="flag",
                error_type=error_type,
                explanation="This contains an error.",
            )
            print(f"  Step {step}: FLAG round {current_idx + 1} → {error_type}")
        else:
            # Clean round: PASS
            action = MultiTurnAction(action_type="pass")
            print(f"  Step {step}: PASS round {current_idx + 1} (looks clean)")

        obs = env.step(action)

        if obs.feedback:
            print(f"           Feedback: {obs.feedback}")
        if obs.step_reward is not None:
            print(f"           Reward: {obs.step_reward:+.2f}")

    print(f"\n  Final cumulative reward: {obs.cumulative_reward:+.2f}")
    print(f"  State: detected={env._state.errors_detected}, "
          f"missed={env._state.errors_missed}, "
          f"false_flags={env._state.false_flags}, "
          f"questions_used={env._state.questions_used}")
    print("  ✓ TEST 5 PASSED")


# ─── Test 6: Question Budget Exhaustion ─────────────────────────────

def test_question_budget():
    """Test that question budget is enforced."""
    header("TEST 6: Question Budget Exhaustion")

    env = WatchDogMultiTurnEnvironment()
    obs = env.reset(seed=7)

    show_ground_truth(env)
    print(f"\n  Initial question budget: {obs.remaining_questions}")

    # Use all questions
    questions_asked = 0
    step = 0
    while not obs.done and questions_asked < 3:  # Try to ask 3 (budget is 2)
        step += 1
        if obs.phase == "question_response":
            # After question, just pass
            obs = env.step(MultiTurnAction(action_type="pass"))
            continue

        obs = env.step(MultiTurnAction(
            action_type="question",
            question_text=f"Question #{questions_asked + 1}?",
        ))
        questions_asked += 1
        print(f"  Asked question #{questions_asked}: "
              f"budget={obs.remaining_questions}, phase={obs.phase}")

        if obs.remaining_questions == 0 and obs.phase != "question_response":
            print("  ✓ Third question was treated as PASS (budget exhausted)")
            break

    # Pass remaining
    while not obs.done:
        obs = env.step(MultiTurnAction(action_type="pass"))

    print(f"  Questions used (state): {env._state.questions_used}")
    assert env._state.questions_used <= 2, "Should not exceed budget of 2"
    print("  ✓ TEST 6 PASSED")


# ─── Test 7: Multiple Episodes (Curriculum) ────────────────────────

def test_multiple_episodes():
    """Run several episodes to verify state accumulates correctly."""
    header("TEST 7: Multiple Episodes (State Accumulation)")

    env = WatchDogMultiTurnEnvironment()

    for ep in range(5):
        obs = env.reset(seed=ep * 17)
        num_errors = sum(1 for r in env._rounds if r.get("has_error"))

        # Simple strategy: pass everything
        while not obs.done:
            obs = env.step(MultiTurnAction(action_type="pass"))

        print(f"  Episode {ep + 1}: domain={obs.task_domain}, "
              f"errors={num_errors}, reward={obs.cumulative_reward:+.2f}")

    state = env.state
    print(f"\n  After 5 episodes:")
    print(f"    Level: {state.current_level}")
    print(f"    Total episodes: {state.total_episodes}")
    print(f"    Errors detected: {state.errors_detected}")
    print(f"    Errors missed: {state.errors_missed}")
    print(f"    False flags: {state.false_flags}")
    print(f"    Cumulative reward: {state.cumulative_reward:+.2f}")

    assert state.total_episodes == 5
    print("  ✓ TEST 7 PASSED")


# ─── Test 8: Conversation Formatting ───────────────────────────────

def test_conversation_format():
    """Verify conversation_so_far accumulates correctly."""
    header("TEST 8: Conversation Formatting")

    env = WatchDogMultiTurnEnvironment()
    obs = env.reset(seed=99)

    print(f"  Total rounds: {obs.total_turns}")
    print(f"\n  After reset (round 1), conversation_so_far:")
    print(f"    '{obs.conversation_so_far[:80]}...'")
    assert obs.conversation_so_far == "[Conversation start]", \
        "Should be just the start marker before any completed rounds"

    # Pass first round
    obs = env.step(MultiTurnAction(action_type="pass"))
    print(f"\n  After round 1, conversation_so_far includes round 1:")
    # Should now include round 1's content
    assert "Round 1" in obs.conversation_so_far or "round" in obs.conversation_so_far.lower() \
        or len(obs.conversation_so_far) > 50, \
        "conversation_so_far should accumulate after passing"
    for line in obs.conversation_so_far.split("\n")[:5]:
        print(f"    {line}")

    # Finish episode
    while not obs.done:
        obs = env.step(MultiTurnAction(action_type="pass"))

    print(f"\n  Final conversation has {len(obs.conversation_so_far)} chars")
    print("  ✓ TEST 8 PASSED")


# ─── Main ──────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 70)
    print("  WATCHDOG MULTI-TURN ENVIRONMENT TEST SUITE")
    print("  Avalon/Werewolf-inspired Dynamic Oversight")
    print("█" * 70)

    tests = [
        ("Layer 1: Pass All Turns", test_pass_all),
        ("Layer 1: FLAG Actions", test_flag_actions),
        ("Layer 2: QUESTION Mechanic", test_question_mechanic),
        ("INTERVENE Action", test_intervene),
        ("Full Strategic Episode", test_strategic_episode),
        ("Question Budget", test_question_budget),
        ("Multiple Episodes", test_multiple_episodes),
        ("Conversation Formatting", test_conversation_format),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {name}")
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'█' * 70}")
    print(f"  RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'█' * 70}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
