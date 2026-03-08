import json
import os
import random
from websocket import create_connection

# ============ GLOBAL CONFIG ============
SERVER_URL = "ws://localhost:8000/ws"
OUTPUT_FILE = "demo_http_output.txt"
GAME_ID = "avalon"
NUM_EPISODES = 1
DIFFICULTY = 2
TURNS_PER_EPISODE = 5
# =======================================


def send_and_receive(ws, message: dict) -> dict:
    """Send a message and receive response."""
    ws.send(json.dumps(message))
    response = ws.recv()
    return json.loads(response)


with open(OUTPUT_FILE, "w") as f:
    f.write(f"Server: {SERVER_URL}\n")
    f.write(f"Game: {GAME_ID} | Episodes: {NUM_EPISODES} | Difficulty: {DIFFICULTY}\n\n")
    os.environ["WATCHDOG_GAME_ID"] = GAME_ID
    print(os.environ.get("WATCHDOG_GAME_ID"))

    for ep in range(NUM_EPISODES):
        # Create new WebSocket connection for each episode (maintains session state)
        ws = create_connection(SERVER_URL)
        
        try:
            # Reset environment - data contains reset params
            result = send_and_receive(ws, {
                "type": "reset",
                "data": {"seed": ep + 42}
            })
            # Response has type="observation" with data containing the actual observation
            obs = result.get("data", {}).get("observation", {})
            done = result.get("data", {}).get("done", False)
            
            f.write(f"EPISODE {ep + 1}\n")

            for turn in range(TURNS_PER_EPISODE):
                if done or obs.get("phase") == "done":
                    break

                f.write(f"\n  TURN {turn + 1}\n")
                # f.write(f"  phase: {obs.get('phase')}\n")
                f.write(f"  reward: {result.get('data', {}).get('reward')}\n")
                f.write(f"  state: {obs.get('current_turn')}\n")

                # Take step - data contains the action
                action = random.choice(["pass", "flag", "question"])
                result = send_and_receive(ws, {
                    "type": "step",
                    "data": {"action_type": action}
                })
                obs = result.get("data", {}).get("observation", {})
                done = result.get("data", {}).get("done", False)
                f.write(f"  action: {action}\n")

            f.write(f"\n{'='*40}\n\n")
            print(f"Episode {ep + 1} done")
            
        finally:
            ws.close()

print(f"Saved to {OUTPUT_FILE}")
