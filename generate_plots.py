#!/usr/bin/env python3
"""Parse output1.txt training logs and generate evaluation plots."""

import re
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__), "eval_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Parse training step logs from output1.txt ──────────────────────────────
log_path = os.path.join(os.path.dirname(__file__), "output1.txt")
with open(log_path, "r") as f:
    raw = f.read()

# Lines are wrapped by the terminal. Join them back: find every { that starts
# a training log dict (contains 'loss') and grab until the matching }.
# Strategy: collapse all newlines that break inside a dict.
steps_data = []
i = 0
while i < len(raw):
    # Find next opening brace followed (eventually) by 'loss'
    start = raw.find("{'loss'", i)
    if start == -1:
        break
    # Find closing brace — count nesting (only 1 level for these dicts)
    depth = 0
    end = start
    for end in range(start, len(raw)):
        if raw[end] == '{':
            depth += 1
        elif raw[end] == '}':
            depth -= 1
            if depth == 0:
                break
    chunk = raw[start:end+1]
    # Remove newlines inserted by terminal wrapping
    chunk = chunk.replace('\n', '')
    # Convert Python dict syntax to JSON
    chunk = chunk.replace("'", '"')
    try:
        d = json.loads(chunk)
        steps_data.append(d)
    except json.JSONDecodeError:
        pass
    i = end + 1

print(f"Parsed {len(steps_data)} training steps")

# Extract per-step arrays
steps = list(range(1, len(steps_data) + 1))
rewards       = [d["reward"] for d in steps_data]
action_rews   = [d["rewards/_reward_action/mean"] for d in steps_data]
format_rews   = [d["rewards/reward_format/mean"] for d in steps_data]
entropy       = [d["entropy"] for d in steps_data]
lr            = [d["learning_rate"] for d in steps_data]
grad_norm     = [d["grad_norm"] for d in steps_data]
loss          = [d["loss"] for d in steps_data]

# Smoothed reward (exponential moving average)
def ema(arr, alpha=0.15):
    out = [arr[0]]
    for v in arr[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out

reward_smooth = ema(rewards)
action_smooth = ema(action_rews)

# ── 2. Before / After evaluation metrics ──────────────────────────────────────
before = {"accuracy": 0.033, "precision": 0.333, "recall": 0.103, "f1": 0.158}
after  = {"accuracy": 0.540, "precision": 0.215, "recall": 0.586, "f1": 0.315}

before_actions = {"PASS": 8, "FLAG": 9, "QUESTION": 130, "UNKNOWN": 153}
after_actions  = {"PASS": 154, "FLAG": 79, "QUESTION": 5, "UNKNOWN": 62}

confusion = {"tp": 17, "fp": 62, "tn": 209, "fn": 12}

# ── Color palette ─────────────────────────────────────────────────────────────
C_PRIMARY   = "#4A90D9"
C_SECONDARY = "#E8744F"
C_ACCENT    = "#50C878"
C_DARK      = "#2C3E50"
C_LIGHT     = "#BDC3C7"
C_BG        = "#FAFAFA"

plt.rcParams.update({
    "figure.facecolor": C_BG,
    "axes.facecolor": "#FFFFFF",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 1: Total Reward Curve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(steps, rewards, alpha=0.25, color=C_PRIMARY, linewidth=0.8, label="Raw reward")
ax.plot(steps, reward_smooth, color=C_PRIMARY, linewidth=2.2, label="EMA (α=0.15)")
ax.axhline(0, color=C_DARK, linewidth=0.6, linestyle="--", alpha=0.5)
ax.set_xlabel("Training Step")
ax.set_ylabel("Total Reward")
ax.set_title("GRPO Training — Total Reward per Step")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_reward_curve.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 01_reward_curve.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 2: Action Reward vs Format Reward
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(steps, ema(action_rews), color=C_PRIMARY, linewidth=2, label="Action Reward (EMA)")
ax.plot(steps, ema(format_rews), color=C_SECONDARY, linewidth=2, label="Format Reward (EMA)")
ax.axhline(0, color=C_DARK, linewidth=0.6, linestyle="--", alpha=0.5)
ax.set_xlabel("Training Step")
ax.set_ylabel("Reward Component")
ax.set_title("Reward Decomposition — Action vs Format")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_reward_decomposition.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 02_reward_decomposition.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 3: Entropy Curve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(steps, entropy, alpha=0.3, color=C_ACCENT, linewidth=0.8)
ax.plot(steps, ema(entropy, 0.1), color=C_ACCENT, linewidth=2, label="Entropy (EMA)")
ax.set_xlabel("Training Step")
ax.set_ylabel("Entropy")
ax.set_title("Policy Entropy Over Training")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_entropy_curve.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 03_entropy_curve.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 4: Learning Rate Schedule
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(steps, [x * 1e5 for x in lr], color=C_DARK, linewidth=1.8)
ax.set_xlabel("Training Step")
ax.set_ylabel("Learning Rate (×10⁻⁵)")
ax.set_title("Cosine Learning Rate Schedule")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_learning_rate.png"), dpi=150)
plt.close(fig)
print("  ✓ 04_learning_rate.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 5: Before vs After — Metric Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
metrics = ["Accuracy", "Precision", "Recall", "F1"]
before_vals = [before["accuracy"], before["precision"], before["recall"], before["f1"]]
after_vals  = [after["accuracy"],  after["precision"],  after["recall"],  after["f1"]]

x = np.arange(len(metrics))
w = 0.32

fig, ax = plt.subplots(figsize=(16, 9))
bars1 = ax.bar(x - w/2, before_vals, w, label="Before Training", color=C_LIGHT, edgecolor=C_DARK, linewidth=0.8)
bars2 = ax.bar(x + w/2, after_vals, w, label="After Training", color=C_PRIMARY, edgecolor=C_DARK, linewidth=0.8)

for bar, val in zip(bars1, before_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{val:.3f}",
            ha="center", va="bottom", fontsize=16, color=C_DARK)
for bar, val in zip(bars2, after_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{val:.3f}",
            ha="center", va="bottom", fontsize=16, color=C_PRIMARY, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 0.75)
ax.set_ylabel("Score")
ax.set_title("Evaluation Metrics — Before vs After GRPO Training")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_before_after_metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 05_before_after_metrics.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 6: Action Distribution — Before vs After
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
action_labels = ["PASS", "FLAG", "QUESTION", "UNKNOWN"]
before_counts = [before_actions[a] for a in action_labels]
after_counts  = [after_actions[a]  for a in action_labels]

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
colors = [C_ACCENT, C_SECONDARY, C_PRIMARY, C_LIGHT]

for ax, counts, title in zip(axes, [before_counts, after_counts], ["Before Training", "After Training"]):
    wedges, texts, autotexts = ax.pie(
        counts, labels=action_labels, autopct="%1.0f%%",
        colors=colors, startangle=90, textprops={"fontsize": 20}
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title(title, fontsize=20, fontweight="bold")

fig.suptitle("Action Distribution — Before vs After Training", fontsize=20, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "06_action_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 06_action_distribution.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 7: Confusion Matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cm = np.array([[confusion["tp"], confusion["fn"]],
               [confusion["fp"], confusion["tn"]]])

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(cm, cmap="Blues", vmin=0)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["FLAG (Pred)", "PASS (Pred)"])
ax.set_yticklabels(["FLAG (True)", "PASS (True)"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (Trained Adapter, n=300)")

for i in range(2):
    for j in range(2):
        label = ["TP", "FN", "FP", "TN"][i * 2 + j]
        color = "white" if cm[i, j] > 100 else C_DARK
        ax.text(j, i, f"{label}\n{cm[i,j]}", ha="center", va="center",
                fontsize=20, fontweight="bold", color=color)

fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(pad=2.0)
fig.savefig(os.path.join(OUT_DIR, "07_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 07_confusion_matrix.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 8: Grad Norm over Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(steps, grad_norm, alpha=0.3, color="#8E44AD", linewidth=0.8)
ax.plot(steps, ema(grad_norm, 0.1), color="#8E44AD", linewidth=2, label="Grad Norm (EMA)")
ax.set_xlabel("Training Step")
ax.set_ylabel("Gradient Norm")
ax.set_title("Gradient Norm Over Training")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "08_grad_norm.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 08_grad_norm.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 9: Summary Dashboard (combined)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig = plt.figure(figsize=(30, 26))
gs  = gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.5)

# Reward curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(steps, rewards, alpha=0.2, color=C_PRIMARY, linewidth=0.7)
ax1.plot(steps, reward_smooth, color=C_PRIMARY, linewidth=2)
ax1.axhline(0, color=C_DARK, linewidth=0.5, linestyle="--", alpha=0.5)
ax1.set_title("Total Reward")
ax1.set_xlabel("Step")

# Reward decomposition
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(steps, ema(action_rews), color=C_PRIMARY, linewidth=2, label="Action")
ax2.plot(steps, ema(format_rews), color=C_SECONDARY, linewidth=2, label="Format")
ax2.axhline(0, color=C_DARK, linewidth=0.5, linestyle="--", alpha=0.5)
ax2.set_title("Reward Decomposition")
ax2.set_xlabel("Step")
ax2.legend(fontsize=20, loc="best")

# Entropy
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(steps, ema(entropy, 0.1), color=C_ACCENT, linewidth=2)
ax3.set_title("Policy Entropy")
ax3.set_xlabel("Step")

# Before/After bars
ax4 = fig.add_subplot(gs[1, 1])
x = np.arange(len(metrics))
ax4.bar(x - w/2, before_vals, w, label="Before", color=C_LIGHT, edgecolor=C_DARK, linewidth=0.6)
ax4.bar(x + w/2, after_vals, w, label="After", color=C_PRIMARY, edgecolor=C_DARK, linewidth=0.6)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, fontsize=20)
ax4.set_ylim(0, 0.85)
ax4.set_title("Eval Metrics: Before vs After")
ax4.legend(fontsize=20, loc="upper left")

# Confusion matrix
ax5 = fig.add_subplot(gs[2, 0])
im = ax5.imshow(cm, cmap="Blues", vmin=0)
ax5.set_xticks([0, 1]); ax5.set_yticks([0, 1])
ax5.set_xticklabels(["FLAG", "PASS"]); ax5.set_yticklabels(["FLAG", "PASS"])
for i in range(2):
    for j in range(2):
        label = ["TP","FN","FP","TN"][i*2+j]
        color = "white" if cm[i,j] > 100 else C_DARK
        ax5.text(j, i, f"{label}\n{cm[i,j]}", ha="center", va="center", fontsize=20, fontweight="bold", color=color)
ax5.set_title("Confusion Matrix (After)")

# Action distribution after
ax6 = fig.add_subplot(gs[2, 1])
ax6.pie(after_counts, labels=action_labels, autopct="%1.0f%%", colors=colors, startangle=90)
ax6.set_title("Action Distribution (After)")

fig.suptitle("WatchDog — GRPO Training & Evaluation Dashboard\nQwen3-8B + LoRA r=16  |  200 steps  |  300 eval samples",
             fontsize=20, fontweight="bold", y=0.99)
fig.savefig(os.path.join(OUT_DIR, "09_dashboard.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 09_dashboard.png")

print(f"\nAll plots saved to {OUT_DIR}/")
