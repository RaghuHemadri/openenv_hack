# OpenEnv Hack — WatchDog

**AI oversight training using [OpenEnv](https://github.com/meta-pytorch/OpenEnv) and multi-agent game plugins.**

This repository contains the **WatchDog** RL environment for training AI agents to detect errors, lies, and misdirection in multi-agent conversations. Built for the OpenEnv hackathon.

## What's Inside

| Component | Description |
|-----------|-------------|
| **watchdog_env** | Core RL environment — server, client, plugins, training scripts |
| **watchdog_train_colab.ipynb** | Google Colab notebook for GRPO training (Qwen3-8B + LoRA) |
| **generate_case_study.py** | Produce markdown case studies from eval episodes |
| **generate_plots.py** | Parse training logs and generate evaluation plots |

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/RaghuHemadri/openenv_hack.git
cd openenv_hack
cp .env.example .env   # Edit with your GEMINI_API_KEY
pip install -e watchdog_env
pip install watchdog_env/requirements.txt
```

### 2. Run the server

```bash
cd watchdog_env
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Try the Gradio UI (optional)

```bash
cd watchdog_env
pip install gradio>=4.0.0
PYTHONPATH=. python -c "from server.ui import build_ui; build_ui().launch()"
```

### 4. Train in Colab

Open `watchdog_train_colab.ipynb` in Google Colab, enable GPU, and run all cells. The notebook clones the repo, installs dependencies, and runs GRPO training.

## Project Structure

```
openenv_hack/
├── README.md                    # This file
├── .env.example                 # Config template (GEMINI_API_KEY, etc.)
├── watchdog_train_colab.ipynb   # Colab training notebook
├── generate_case_study.py       # Case study generator
├── generate_plots.py            # Training log plots
└── watchdog_env/                # Main environment package
    ├── README.md                # Detailed WatchDog docs
    ├── server/                  # FastAPI server + Gradio UI
    ├── plugins/                 # Avalon, Cicero (multi-agent games)
    ├── client.py                # OpenEnv client
    ├── train_user.py            # GRPO training
    └── ...
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

- **GEMINI_API_KEY** — Required for LLM-backed plugins (Avalon, Cicero)
- **WATCHDOG_LLM_BACKEND** — `gemini` or `local`
- **WATCHDOG_USE_LLM** — Set to `0` to use template fallback (no API key needed)

## License

MIT
