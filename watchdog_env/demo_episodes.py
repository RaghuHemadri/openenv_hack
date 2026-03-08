import os
import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ============ GLOBAL CONFIG ============
OUTPUT_FILE = "demo_output.txt"
GAME_ID = "codenames"  # "avalon", "cicero", or "codenames"
NUM_EPISODES = 1
DIFFICULTY = 2
TURNS_PER_EPISODE = 6
# =======================================

from server.watchdog_environment import WatchDogMultiTurnEnvironment
from models import MultiTurnAction

env = WatchDogMultiTurnEnvironment(game_id=GAME_ID, use_mutations=True, use_llm=True)

with open(OUTPUT_FILE, "w") as f:
    f.write(f"Game: {GAME_ID} | Episodes: {NUM_EPISODES} | Difficulty: {DIFFICULTY}\n\n")

    for ep in range(NUM_EPISODES):
        env._state.current_level = DIFFICULTY
        obs = env.reset(seed=ep + 42)
        f.write(f"EPISODE {ep + 1}\n")

        for turn in range(TURNS_PER_EPISODE):
            if obs.phase == "done":
                break

            has_error = getattr(env, '_current_has_error', False)
            error_detail = getattr(env, '_current_error_detail', None)

            f.write(f"\n  TURN {turn + 1}\n")
            f.write(f"  reward: {obs.reward}\n")
            f.write(f"  state: {obs.current_turn}\n")

            f.write(f"  has_mutation: {has_error}\n")
            if has_error and error_detail:
                f.write(f"  mutation_type: {error_detail.get('type')}\n")
                f.write(f"  mutation_description: {error_detail.get('description')}\n")
            f.write(f"  ground_truth: {'FLAG' if has_error else 'PASS'}\n")

            action = random.choice(["pass", "flag", "question"])
            # action = "pass"
            obs = env.step(MultiTurnAction(action_type=action))
            f.write(f"  action: {action}\n")

        f.write(f"\n{'='*40}\n\n")
        print(f"Episode {ep + 1} done")

print(f"Saved to {OUTPUT_FILE}")
