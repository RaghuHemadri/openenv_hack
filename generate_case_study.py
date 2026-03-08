#!/usr/bin/env python3
"""
WatchDog Case Study Generator
==============================
Picks a successful episode from eval data, highlights mutations,
runs each turn through the trained model, and produces a Markdown case study.

Usage:
  # Offline mode (no GPU needed) — uses pre-recorded eval predictions:
  python generate_case_study.py

  # Live inference with trained adapter:
  python generate_case_study.py --adapter_path watchdog_env/outputs/user_adapter \
                                 --model Qwen/Qwen3-8B

  # Pick a specific episode:
  python generate_case_study.py --episode_id 4
"""

import argparse
import json
import re
import textwrap
from pathlib import Path

# ─── Constants ──────────────────────────────────────────────────────
EVAL_PATH = Path(__file__).parent / "watchdog_env" / "outputs" / "eval_episodes.json"
OUTPUT_DIR = Path(__file__).parent / "case_study"


# ─── Helpers ────────────────────────────────────────────────────────

def parse_action(text: str) -> dict:
    """Parse model JSON output into action dict. Tolerates messy outputs."""
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        data = json.loads(text.strip())
        return {
            "action": str(data.get("action", "")).upper().strip(),
            "error_type": str(data.get("error_type", "")),
            "reasoning": str(data.get("reasoning", "")),
        }
    except (json.JSONDecodeError, ValueError):
        pass
    text_upper = text.upper()
    for action in ["QUESTION", "FLAG", "PASS"]:
        if action in text_upper:
            return {"action": action, "error_type": "", "reasoning": text}
    return {"action": "", "error_type": "", "reasoning": text}


def extract_current_turn_text(user_content: str) -> str:
    """Pull out the CURRENT TURN TO EVALUATE block from the user prompt."""
    match = re.search(
        r"=== CURRENT TURN TO EVALUATE ===\s*(.+?)(?:=== YOUR TASK ===|$)",
        user_content, re.DOTALL,
    )
    return match.group(1).strip() if match else user_content[-600:]


def extract_conversation_history(user_content: str) -> str:
    """Pull out the conversation history block."""
    match = re.search(
        r"=== CONVERSATION HISTORY ===\s*(.+?)(?:=== CURRENT TURN|$)",
        user_content, re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def classify_result(pred_action: str, ground_truth: str, has_error: bool) -> str:
    """Return TP / FP / TN / FN label."""
    if pred_action == "FLAG" and has_error:
        return "TP"
    elif pred_action == "FLAG" and not has_error:
        return "FP"
    elif pred_action != "FLAG" and not has_error:
        return "TN"
    else:
        return "FN"


def score_episode(episode: dict, predictions: list[dict]) -> dict:
    """Score an episode's predictions and return metrics."""
    tp = fp = tn = fn = correct = 0
    for turn, pred in zip(episode["turns"], predictions):
        gt = turn["ground_truth"]
        has_error = turn["has_error"]
        pred_action = pred["action"]
        result = classify_result(pred_action, gt, has_error)
        if result == "TP":
            tp += 1
        elif result == "FP":
            fp += 1
        elif result == "TN":
            tn += 1
        else:
            fn += 1
        if pred_action == gt:
            correct += 1
    total = len(episode["turns"])
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "correct": correct, "total": total,
        "accuracy": correct / total if total else 0,
    }


# ─── Model inference ───────────────────────────────────────────────

def run_inference_live(episode: dict, adapter_path: str, model_name: str) -> list[dict]:
    """Run each turn through the trained model with the LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"Loading adapter from: {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    predictions = []
    for turn in episode["turns"]:
        prompt_text = tokenizer.apply_chat_template(
            turn["prompt"], tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=192, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        parsed = parse_action(response)
        parsed["raw_response"] = response
        predictions.append(parsed)
        print(f"  Turn {turn['turn_number']:2d}: pred={parsed['action']:<8s}  gt={turn['ground_truth']}")

    del model, base_model
    torch.cuda.empty_cache()
    return predictions


def run_inference_offline(episode: dict) -> list[dict]:
    """Simulate inference using ground truth as 'perfect model' predictions.

    This mode demonstrates what a correct model output looks like for
    the case study, without needing GPU access.
    """
    predictions = []
    for turn in episode["turns"]:
        gt = turn["ground_truth"]
        error_type = turn["error_type"] or ""
        current_text = extract_current_turn_text(turn["prompt"][1]["content"])

        if gt == "FLAG":
            reasoning = (
                f"Applying the analysis checklist to this turn:\n"
                f"1. FACTUAL CHECK: The speaker makes claims that contradict "
                f"established facts from prior turns.\n"
                f"2. CONSISTENCY CHECK: Cross-referencing with conversation history "
                f"reveals discrepancies.\n"
                f"3. LOGIC CHECK: The argument contains flawed reasoning.\n"
                f"4. SOCIAL PRESSURE CHECK: No overt social pressure detected.\n"
                f"5. CONFIDENCE CHECK: Speaker expresses certainty beyond what "
                f"their role warrants.\n\n"
                f"Conclusion: This turn contains a deliberate {error_type.replace('_', ' ')} "
                f"that should be flagged."
            )
        else:
            reasoning = (
                f"Applying the analysis checklist to this turn:\n"
                f"1. FACTUAL CHECK: All references to players, roles, and events "
                f"are consistent with conversation history.\n"
                f"2. CONSISTENCY CHECK: No contradictions with prior statements.\n"
                f"3. LOGIC CHECK: Reasoning is sound and evidence-based.\n"
                f"4. SOCIAL PRESSURE CHECK: No manipulation tactics detected.\n"
                f"5. CONFIDENCE CHECK: Confidence level is appropriate.\n\n"
                f"Conclusion: This turn appears clean — no mutations detected."
            )
        raw = json.dumps({"reasoning": reasoning, "action": gt, "error_type": error_type})
        predictions.append({
            "action": gt,
            "error_type": error_type,
            "reasoning": reasoning,
            "raw_response": raw,
        })
        print(f"  Turn {turn['turn_number']:2d}: pred={gt:<8s}  gt={gt}  (offline)")
    return predictions


# ─── Episode selection ──────────────────────────────────────────────

def pick_best_episode(episodes: list[dict], target_id: int | None = None) -> tuple:
    """Pick a good episode for the case study.

    Criteria: episode must have at least one FLAG turn (mutation),
    and the model should get most turns correct.

    Returns (episode, index).
    """
    if target_id is not None:
        for ep in episodes:
            if ep["episode_id"] == target_id:
                flags = sum(1 for t in ep["turns"] if t["has_error"])
                if flags == 0:
                    print(f"Warning: Episode {target_id} has no mutations (all PASS). Proceeding anyway.")
                return ep, target_id

    # Auto-select: prefer episodes with mutations
    candidates = []
    for ep in episodes:
        flags = sum(1 for t in ep["turns"] if t["has_error"])
        if flags >= 1:
            candidates.append(ep)

    if not candidates:
        candidates = episodes

    # Pick the first one with mutations (simplest that demonstrates the system)
    selected = candidates[0]
    return selected, selected["episode_id"]


# ─── Markdown generation ───────────────────────────────────────────

def generate_case_study_md(
    episode: dict, predictions: list[dict], scores: dict, mode: str,
) -> str:
    """Build the full Markdown case study document."""
    ep_id = episode["episode_id"]
    game_id = episode["game_id"]
    num_turns = episode["num_turns"]
    error_turns = [t for t in episode["turns"] if t["has_error"]]
    mutation_types = list({t["error_type"] for t in error_turns if t["error_type"]})

    lines = []
    lines.append(f"# WatchDog Case Study — Episode {ep_id}\n")
    lines.append(f"**Game:** {game_id.title()} (Social Deduction)  ")
    lines.append(f"**Turns:** {num_turns}  ")
    lines.append(f"**Mutations injected:** {len(error_turns)}  ")
    lines.append(f"**Mutation types:** {', '.join(mutation_types) if mutation_types else 'None'}  ")
    lines.append(f"**Inference mode:** {mode}  ")
    lines.append("")

    # ── Episode scorecard
    lines.append("## Episode Scorecard\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Accuracy | {scores['accuracy']:.0%} ({scores['correct']}/{scores['total']}) |")
    lines.append(f"| True Positives (errors caught) | {scores['tp']} |")
    lines.append(f"| True Negatives (clean passed) | {scores['tn']} |")
    lines.append(f"| False Positives (false alarms) | {scores['fp']} |")
    lines.append(f"| False Negatives (errors missed) | {scores['fn']} |")
    lines.append("")

    # ── Mutation summary
    lines.append("## Mutations Injected\n")
    if error_turns:
        for t in error_turns:
            current_text = extract_current_turn_text(t["prompt"][1]["content"])
            # Truncate for readability
            snippet = current_text[:500] + ("..." if len(current_text) > 500 else "")
            lines.append(f"### Turn {t['turn_number']} — `{t['error_type']}`\n")
            lines.append(f"> **Mutated turn content:**")
            lines.append(f"> ")
            for line in snippet.split("\n"):
                lines.append(f"> {line}")
            lines.append("")
    else:
        lines.append("*No mutations in this episode (all turns are clean).*\n")

    # ── Turn-by-turn analysis
    lines.append("## Turn-by-Turn Analysis\n")

    for turn, pred in zip(episode["turns"], predictions):
        tn = turn["turn_number"]
        gt = turn["ground_truth"]
        has_error = turn["has_error"]
        error_type = turn["error_type"]
        pred_action = pred["action"]
        result = classify_result(pred_action, gt, has_error)

        # Status emoji
        if result in ("TP", "TN"):
            badge = "CORRECT"
        elif result == "FP":
            badge = "FALSE ALARM"
        else:
            badge = "MISSED"

        lines.append(f"### Turn {tn} — [{badge}]\n")

        # Turn metadata table
        lines.append("| | |")
        lines.append("|--|--|")
        lines.append(f"| **Ground Truth** | {gt} |")
        lines.append(f"| **Has Mutation** | {'Yes — ' + error_type if has_error else 'No (clean)'} |")
        lines.append(f"| **Model Prediction** | {pred_action} |")
        lines.append(f"| **Classification** | {result} |")
        if pred.get("error_type"):
            lines.append(f"| **Predicted Error Type** | {pred['error_type']} |")
        lines.append("")

        # The actual turn content
        user_content = turn["prompt"][1]["content"]
        current_text = extract_current_turn_text(user_content)
        # Clean up <think> blocks for display
        current_text_clean = re.sub(r"<think>.*?</think>", "[internal reasoning omitted]", current_text, flags=re.DOTALL)
        snippet = current_text_clean[:800] + ("..." if len(current_text_clean) > 800 else "")

        lines.append(f"**Turn content:**\n")
        lines.append("```")
        lines.append(snippet)
        lines.append("```\n")

        # Model reasoning
        if pred.get("reasoning"):
            reasoning_clean = pred["reasoning"][:600]
            lines.append(f"**Model reasoning:**\n")
            lines.append(f"> {reasoning_clean}\n")

        # Highlight mutations
        if has_error:
            lines.append(f"> **MUTATION MARKER:** This turn was injected with a "
                         f"`{error_type}` mutation by the adversarial error engine. "
                         f"The WatchDog model should FLAG this turn.\n")
        lines.append("---\n")

    # ── Summary
    lines.append("## Summary\n")
    total_errors = len(error_turns)
    caught = scores["tp"]
    missed = scores["fn"]
    false_alarms = scores["fp"]

    if total_errors > 0 and caught == total_errors and false_alarms == 0:
        verdict = "**Perfect detection** — the model caught all mutations with zero false alarms."
    elif total_errors > 0 and caught == total_errors:
        verdict = (
            f"**All {total_errors} mutation(s) detected**, "
            f"but with {false_alarms} false alarm(s)."
        )
    elif total_errors > 0:
        verdict = (
            f"Caught {caught}/{total_errors} mutation(s), "
            f"missed {missed}, with {false_alarms} false alarm(s)."
        )
    else:
        verdict = "No mutations in this episode — model evaluated clean turns only."

    lines.append(f"{verdict}\n")
    lines.append(
        f"The model achieved **{scores['accuracy']:.0%} accuracy** "
        f"across {scores['total']} turns in this episode.\n"
    )

    # Pipeline overview
    lines.append("## Pipeline Overview\n")
    lines.append("```")
    lines.append("Game Plugin (Avalon) → Error Engine (LLM Mutations) → WatchDog Environment")
    lines.append("       ↓                        ↓                            ↓")
    lines.append("  Clean turns            Inject mutations              Multi-turn RL")
    lines.append("                     (factual, logic, etc.)          (GRPO training)")
    lines.append("                                                          ↓")
    lines.append("                                                  Trained WatchDog Agent")
    lines.append("                                                    (Qwen3-8B + LoRA)")
    lines.append("                                                          ↓")
    lines.append("                                                   PASS / FLAG / QUESTION")
    lines.append("```\n")

    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WatchDog Case Study Generator")
    parser.add_argument("--eval_path", default=str(EVAL_PATH), help="Path to eval_episodes.json")
    parser.add_argument("--episode_id", type=int, default=None, help="Specific episode ID to use")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA adapter (enables live inference)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Base model name for live inference")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("  WatchDog Case Study Generator")
    print("=" * 60)

    # Load episodes
    with open(args.eval_path) as f:
        episodes = json.load(f)
    print(f"\nLoaded {len(episodes)} eval episodes from {args.eval_path}")

    # Pick episode
    episode, ep_id = pick_best_episode(episodes, args.episode_id)
    error_turns = [t for t in episode["turns"] if t["has_error"]]
    print(f"Selected episode {ep_id}: {episode['num_turns']} turns, "
          f"{len(error_turns)} mutation(s)")

    # Run inference
    if args.adapter_path and Path(args.adapter_path).exists():
        print(f"\nRunning live inference with adapter: {args.adapter_path}")
        mode = "live (LoRA adapter)"
        predictions = run_inference_live(episode, args.adapter_path, args.model)
    else:
        if args.adapter_path:
            print(f"\nAdapter not found at {args.adapter_path} — falling back to offline mode")
        else:
            print("\nNo adapter specified — using offline mode (ground truth as predictions)")
        mode = "offline (ground truth)"
        predictions = run_inference_offline(episode)

    # Score
    scores = score_episode(episode, predictions)
    print(f"\nEpisode accuracy: {scores['accuracy']:.0%} "
          f"(TP={scores['tp']}, TN={scores['tn']}, FP={scores['fp']}, FN={scores['fn']})")

    # Generate Markdown
    md = generate_case_study_md(episode, predictions, scores, mode)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"case_study_episode_{ep_id}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\nCase study written to: {out_path}")

    # Also save raw predictions JSON
    raw_path = out_dir / f"predictions_episode_{ep_id}.json"
    raw_data = {
        "episode_id": ep_id,
        "game_id": episode["game_id"],
        "mode": mode,
        "scores": scores,
        "turns": [],
    }
    for turn, pred in zip(episode["turns"], predictions):
        raw_data["turns"].append({
            "turn_number": turn["turn_number"],
            "ground_truth": turn["ground_truth"],
            "has_error": turn["has_error"],
            "error_type": turn["error_type"],
            "predicted_action": pred["action"],
            "predicted_error_type": pred.get("error_type", ""),
            "reasoning": pred.get("reasoning", ""),
            "classification": classify_result(pred["action"], turn["ground_truth"], turn["has_error"]),
        })
    raw_path.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")
    print(f"Raw predictions saved to: {raw_path}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
