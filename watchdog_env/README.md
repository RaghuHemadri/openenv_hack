# WatchDog 🐕 — Train the AI That Watches the AI

**An RL environment for training AI oversight agents using [OpenEnv](https://github.com/meta-pytorch/OpenEnv) (v0.2.0+)**

> AI agents are everywhere — writing code, giving medical advice, managing finances. But they hallucinate, make logic errors, and sometimes cross safety boundaries. WatchDog trains dedicated AI oversight agents to catch these mistakes in real time.

## What is WatchDog?

WatchDog is a reinforcement learning environment where an **Overseer agent** reviews conversations between a User and Worker AIs in multi-agent games, detecting:

| Error Type | Example |
|-----------|---------|
| **Factual Error** | "The capital of Australia is Sydney" |
| **Logic Error** | Post hoc fallacy, false dichotomy |
| **Code Bug** | Off-by-one, infinite recursion |
| **Safety Violation** | Dangerous health/financial advice |
| **Sycophancy** | Agreeing with user's wrong claims |

The Overseer chooses **PASS** (no error), **FLAG** (error found), or **QUESTION** (ask the Worker for clarification). False alarms are heavily penalized; catching real errors is rewarded.

## Games (Plugins)

| Game | Description |
|------|-------------|
| **Avalon** 🐺 | Werewolf/Mafia — Detect lies and misdirection in social deduction |
| **Cicero** ⚔️ | Diplomacy — Seven powers negotiate in 1914 Europe; spot bluffs and fabricated claims |

Each plugin generates multi-turn conversations; the error engine optionally **mutates** turns to inject realistic errors for training.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                            │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  GRPOTrainer │───▶│  Environment │───▶│   Reward     │   │
│  │  (TRL/       │    │  reset/step  │    │  (F1 + type  │   │
│  │   PEFT)      │◀───│  HTTP/WS     │◀───│  + bonus)    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
│  Curriculum: Level 1 (easy) → Level 4 (adversarial)          │
│  Auto-advances when rolling F1 > threshold                    │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install

```bash
pip install openenv-core[core]>=0.2.0
# Optional: Gemini backend for live LLM plugins
pip install langchain-google-genai langchain-core
```

### 2. Configure

```bash
cp ../.env.example ../.env
# Edit .env: set GEMINI_API_KEY for Avalon/Cicero
# Set WATCHDOG_USE_LLM=0 to use template fallback (no API key)
```

### 3. Run the Server

```bash
cd watchdog_env
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 4. Play with the Gradio UI (optional)

```bash
cd watchdog_env
pip install gradio>=4.0.0
PYTHONPATH=. python -c "from server.ui import build_ui; build_ui().launch()"
```

### 5. Use the Client

```python
from watchdog_env.client import WatchDogMultiTurnEnv
from watchdog_env.models import MultiTurnAction

with WatchDogMultiTurnEnv(base_url="http://localhost:8000") as env:
    result = env.reset(game_id="avalon")  # or "cicero"
    print(result.observation.current_turn)

    # PASS = no error, FLAG = error found, QUESTION = ask Worker
    action = MultiTurnAction(action_type="pass")
    step_result = env.step(action)
    print(f"Reward: {step_result.reward}")
    print(f"Feedback: {step_result.observation.feedback}")
```

### 6. Train with GRPO

```bash
# Train the oversight model (4-bit Qwen3 8B + LoRA)
python -m watchdog_env.train_user \
    --model Qwen/Qwen3-8B \
    --episodes 100 \
    --train_steps 200
```

Or use the **Colab notebook** (`watchdog_train_colab.ipynb`) for GPU training in the cloud.

## Reward Function

```
R_total = R_detection + R_classification + R_location + R_explanation

Detection:
  True Positive  (found real error):      +1.0
  True Negative  (clean = clean):         +0.5
  False Positive (hallucinated error):   -1.5  ← Heavy penalty
  False Negative (missed error):         -0.5

Bonuses (on TP only):
  Correct error type:     +0.3
  Exact location match:   +0.2
  Good explanation:       +0.2
```

## Curriculum

| Level | Difficulty | Error Types | F1 Threshold |
|-------|-----------|-------------|--------------|
| 1 | Easy | Factual only | > 0.60 |
| 2 | Medium | + Logic + Code | > 0.65 |
| 3 | Hard | + Safety + Sycophancy | > 0.70 |
| 4 | Adversarial | All types, subtle | — |

## File Structure

```
watchdog_env/
├── __init__.py                  # Package exports
├── models.py                    # MultiTurnAction (PASS/FLAG/QUESTION)
├── client.py                    # WatchDogMultiTurnEnv (OpenEnv client)
├── error_engine.py              # Mutation layer (injects errors into turns)
├── rewards.py                   # Reward computation (F1, type bonuses)
├── train_user.py                # GRPO training for oversight model
├── train_adversarial.py         # Adversarial min-max (user vs mutator)
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Dependencies
├── mutations/
│   ├── registry.py              # MutationScenario, MutationCategory
│   └── llm_backend.py           # TrainableMutationModel (Qwen3 8B + LoRA)
├── plugins/
│   ├── base.py                  # BasePlugin interface
│   ├── registry.py              # Plugin registry
│   ├── avalon/                  # Werewolf/Mafia game
│   ├── cicero/                  # Diplomacy negotiation
│   └── README.md                # Guide for adding plugins
└── server/
    ├── watchdog_environment.py  # WatchDogMultiTurnEnvironment
    ├── app.py                   # FastAPI server
    └── ui.py                    # Gradio play UI
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|--------------|
| `/health` | GET | Health check |
| `/schema` | GET | Action/Observation JSON schemas |
| `/reset` | POST | Start new episode (`game_id`: avalon, cicero) |
| `/step` | POST | Submit action (pass/flag/question) |
| `/state` | GET | Environment state |
| `/ws` | WS | WebSocket for persistent sessions |

## Deploy to Hugging Face Spaces

```bash
openenv push --repo-id YOUR_USERNAME/watchdog_env
```

## References

- [CriticGPT (OpenAI, 2024)](https://arxiv.org/abs/2407.00215) — RL-trained critics catch 63% more bugs
- [Weak-to-Strong Generalization (OpenAI, 2023)](https://arxiv.org/abs/2312.09390) — Small models can oversee large ones
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) — GRPO produces emergent self-verification
- [Prover-Verifier Games (OpenAI, 2024)](https://arxiv.org/abs/2407.13692) — 1000x smaller verifiers work

## License

MIT
