"""WatchDog Environment — Reward computation."""

from models import WatchDogAction


def compute_reward(
    action: WatchDogAction, manifest: dict
) -> tuple[float, str, str]:
    """
    Compute reward by comparing agent's verdict against ground truth.

    Returns: (reward, feedback_message, result_type)
    """
    has_error = manifest.get("has_error", False)
    agent_says_error = action.verdict != "clean"

    if has_error and agent_says_error:
        # True Positive
        reward = 1.0
        result_type = "tp"
        feedback = "Correct! You found the error."

        actual_type = manifest["errors"][0]["type"]
        if action.verdict == actual_type:
            reward += 0.3
            feedback += f" Correct type: {actual_type}."
        else:
            feedback += f" Wrong type: you said '{action.verdict}', actual was '{actual_type}'."

        actual_location = manifest["errors"][0].get("location", "")
        if action.location and actual_location:
            if action.location.strip().lower() == actual_location.strip().lower():
                reward += 0.2
                feedback += " Exact location match."
            elif _location_close(action.location, actual_location):
                reward += 0.1
                feedback += " Close location match."

        if action.explanation:
            exp_bonus = _score_explanation(action.explanation, manifest["errors"][0])
            reward += exp_bonus

    elif has_error and not agent_says_error:
        # False Negative — missed the error
        reward = -0.5
        result_type = "fn"
        actual_type = manifest["errors"][0]["type"]
        feedback = f"Missed! There was a {actual_type} error."

    elif not has_error and not agent_says_error:
        # True Negative — correctly said clean
        reward = 0.5
        result_type = "tn"
        feedback = "Correct! The conversation was clean."

    else:
        # False Positive — hallucinated a bug
        reward = -1.5
        result_type = "fp"
        feedback = "Wrong! You hallucinated an error. The conversation was clean."

    return reward, feedback, result_type


def _location_close(predicted: str, actual: str) -> bool:
    """Check if predicted location is close to actual (±1 turn)."""
    try:
        pred_num = int("".join(c for c in predicted if c.isdigit()))
        actual_num = int("".join(c for c in actual if c.isdigit()))
        return abs(pred_num - actual_num) <= 1
    except (ValueError, IndexError):
        return False


def _score_explanation(explanation: str, error: dict) -> float:
    """Score explanation quality using rule-based matching."""
    score = 0.0
    explanation_lower = explanation.lower()

    corrupted = error.get("corrupted", "")
    if corrupted and corrupted.lower()[:40] in explanation_lower:
        score += 0.1

    original = error.get("original", "")
    if original and original.lower()[:40] in explanation_lower:
        score += 0.1

    return min(score, 0.2)


# ═══════════════════════════════════════════════════════════════════
# MULTI-TURN REWARD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def compute_flag_reward(
    action: "MultiTurnAction", round_data: dict
) -> tuple[float, str, str]:
    """Compute reward for a FLAG action on a specific worker turn.

    Returns: (reward, feedback, result_type)
    """
    from models import MultiTurnAction  # local import to avoid circular

    has_error = round_data.get("has_error", False)
    error_detail = round_data.get("error_detail")

    if has_error and error_detail:
        # True positive — caught an error
        reward = 1.0
        result_type = "tp"
        feedback = "Correct FLAG! You found an error."

        # Type match bonus
        if action.error_type == error_detail["type"]:
            reward += 0.3
            feedback += f" Correct type: {error_detail['type']}."
        else:
            feedback += (
                f" Wrong type: you said '{action.error_type}', "
                f"actual was '{error_detail['type']}'."
            )

        # Explanation bonus
        if action.explanation:
            exp_bonus = _score_explanation(action.explanation, error_detail)
            reward += exp_bonus
            if exp_bonus > 0:
                feedback += f" Explanation bonus: +{exp_bonus:.1f}."

        return reward, feedback, result_type
    else:
        # False positive — flagged a clean turn
        reward = -1.5
        result_type = "fp"
        feedback = "Wrong FLAG! This turn was clean. False alarm penalty: -1.5."
        return reward, feedback, result_type


def compute_pass_reward(round_data: dict) -> tuple[float, str]:
    """Compute per-turn reward for PASS action.

    Returns: (reward, feedback)
    """
    has_error = round_data.get("has_error", False)
    if not has_error:
        # Correct pass on clean turn — small trust reward
        return 0.1, "PASS on clean turn. +0.1"
    else:
        # Passed an error turn — no immediate penalty (penalized at episode end)
        return 0.0, ""


def compute_intervene_reward(round_data: dict) -> tuple[float, str, str]:
    """Compute reward for INTERVENE action.

    Returns: (reward, feedback, result_type)
    """
    has_error = round_data.get("has_error", False)
    error_detail = round_data.get("error_detail")

    if has_error and error_detail:
        reward = 2.0
        feedback = (
            f"Correct INTERVENE! Stopped a {error_detail['type']}. +2.0."
        )
        return reward, feedback, "intervene_correct"
    else:
        reward = -2.5
        feedback = "Wrong INTERVENE! No error here. Devastating penalty: -2.5."
        return reward, feedback, "intervene_wrong"


def compute_question_cost() -> tuple[float, str]:
    """Cost for asking a QUESTION."""
    return -0.05, "QUESTION asked. Investigation cost: -0.05."


def compute_episode_end_bonus(
    flagged_error_rounds: set,
    all_rounds: list[dict],
    rounds_completed: int,
    total_rounds: int,
) -> tuple[float, str]:
    """Compute end-of-episode bonuses and penalties.

    - Missed errors: -0.5 each
    - Efficiency bonus: +0.15 per remaining round if all errors caught

    Returns: (bonus_reward, summary_feedback)
    """
    reward = 0.0
    parts = []

    # Count total errors and missed errors
    error_rounds = {
        i for i, r in enumerate(all_rounds) if r.get("has_error", False)
    }
    missed = error_rounds - flagged_error_rounds
    total_errors = len(error_rounds)

    if missed:
        miss_penalty = -0.5 * len(missed)
        reward += miss_penalty
        parts.append(f"Missed {len(missed)} error(s): {miss_penalty:.1f}")

    # Efficiency bonus: if all errors caught and rounds remaining
    if total_errors > 0 and not missed:
        remaining = total_rounds - rounds_completed
        if remaining > 0:
            eff_bonus = 0.15 * remaining
            reward += eff_bonus
            parts.append(f"Efficiency bonus ({remaining} rounds saved): +{eff_bonus:.2f}")

    # Clean sweep bonus (all errors caught, no false flags)
    if total_errors > 0 and not missed:
        parts.append("All errors detected!")

    summary = " | ".join(parts) if parts else "Episode complete."
    return reward, summary
