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
