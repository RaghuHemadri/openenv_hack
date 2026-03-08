# Multi-Agent System Plugins

This package provides an extensible **multi-agent system plugin** layer. Each plugin simulates a game or scenario with multiple agents; the primitive is a **step** (each step can have multiple agent turns). Plugins are used to generate steps based on **state history** and optional **config**.

## Quick start

- **List games**: `from watchdog_env.plugins import list_game_ids; list_game_ids()` → e.g. `['cicero']`
- **Get a plugin**: `from watchdog_env.plugins import get_plugin; plugin = get_plugin('cicero')`
- **Run a scenario**: `plugin.reset(seed=42, config=CiceroConfig(...))` then `plugin.generate_step(seed, 0)`, `generate_step(seed, 1)`, … until `step.done` is True.

## API

- **MultiAgentConfig** — Base config class; subclass for your game (e.g. `CiceroConfig`).
- **MultiAgentState** — Tracks system behaviour (step_index, turns_so_far, done); used when generating each step.
- **MultiAgentStep** — One step: list of **AgentTurn** (agent_id, action_text), plus `done`, optional `state` snapshot.
- **MultiAgentSystemPlugin** — Abstract base. Implement: `get_game_id`, `reset(seed, config)`, `generate_step(seed, step_index)`, `get_state`, `get_display_name`, `list_agent_ids`.

## Cicero plugin

- **Game ID**: `cicero`
- **Config**: `CiceroConfig(num_steps=3, powers=None, model_name="gemini-2.0-flash", temperature=0.85)`
- **API key**: Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` for live Gemini calls. No template fallback; LLM is required.
- **Optional deps**: `pip install langchain-google-genai langchain-core` (or `pip install -e ".[plugins]"` from `watchdog_env`).

## Tests

- **Base and registry** (no API key):  
  `pytest watchdog_env/plugins/tests/test_base_and_registry.py -v`
- **Cicero** (requires API key; skipped if unset):  
  Set `GEMINI_API_KEY` or `GOOGLE_API_KEY`, then:  
  `pytest watchdog_env/plugins/tests/test_cicero_plugin.py -v`  
  Run from repo root with `PYTHONPATH=<repo_root>`.

---

# Guide: Adding Additional Plugins

Follow these steps to add a new multi-agent system plugin (e.g. a new game or scenario).

## 1. Create a plugin folder

Create a new directory under `watchdog_env/plugins/`, for example:

```
watchdog_env/plugins/
  my_game/
    __init__.py
    my_game_config.py   # optional: your config class
    my_game_plugin.py   # your plugin implementation
```

## 2. Define your config (optional but recommended)

Subclass **MultiAgentConfig** with your game-specific fields:

```python
# my_game_config.py
from dataclasses import dataclass
from watchdog_env.plugins.base import MultiAgentConfig

@dataclass
class MyGameConfig(MultiAgentConfig):
    num_rounds: int = 5
    agent_names: list[str] | None = None
    difficulty: str = "medium"
```

## 3. Implement the plugin

Create a class that implements **all** methods of **MultiAgentSystemPlugin**:

```python
# my_game_plugin.py
from watchdog_env.plugins.base import (
    AgentTurn,
    MultiAgentConfig,
    MultiAgentState,
    MultiAgentStep,
    MultiAgentSystemPlugin,
)

class MyGamePlugin(MultiAgentSystemPlugin):
    def __init__(self) -> None:
        self._state = MultiAgentState()

    def get_game_id(self) -> str:
        return "my_game"

    def reset(self, seed: int | None = None, config: MultiAgentConfig | None = None) -> None:
        # Initialize self._state (step_index=0, turns_so_far=[], config, done=False).
        ...

    def generate_step(self, seed: int | None, step_index: int) -> MultiAgentStep:
        # 1. Use self._state (e.g. turns_so_far) to build context for this step.
        # 2. Produce one or more AgentTurn(s); append to state.turns_so_far.
        # 3. Update state (step_index, done if last step).
        # 4. Return MultiAgentStep(turns=..., done=..., state=snapshot of state).
        ...

    def get_state(self) -> MultiAgentState:
        return self._state

    def get_display_name(self) -> str:
        return "My Game"

    def list_agent_ids(self) -> list[str]:
        return ["agent_a", "agent_b"]
```

Important:

- **generate_step must be based on state history**: use `self._state.turns_so_far` (and other fields) when producing the next step (e.g. for LLM context or game logic), then update `self._state` after the step.
- Use **MultiAgentConfig** (or your subclass) in **reset(seed, config=...)**; do not rely on kwargs for config.
- Set **step.done = True** on the last step so consumers know the scenario is finished.

## 4. Export and register

In `my_game/__init__.py`:

```python
from watchdog_env.plugins.my_game.my_game_plugin import MyGamePlugin
from watchdog_env.plugins.my_game.my_game_config import MyGameConfig  # if you have one

__all__ = ["MyGamePlugin", "MyGameConfig"]
```

In **watchdog_env/plugins/__init__.py**, register your plugin so it is available by game_id:

```python
try:
    from watchdog_env.plugins.my_game import MyGamePlugin
    register(MyGamePlugin())
except Exception:
    MyGamePlugin = None  # optional dependency
```

Add `"watchdog_env.plugins.my_game"` to **packages** and **package_dir** in `watchdog_env/pyproject.toml` if you added a new top-level plugin package.

## 5. Add tests

Create tests that:

- Call **get_game_id**, **reset(seed, config)**, **generate_step(seed, 0)**, … **get_state**, **get_display_name**, **list_agent_ids**.
- Assert step content (turns, done) and state updates.
- If your plugin uses an API (e.g. Gemini), require the API key and skip tests when it is unset (see `test_cicero_plugin.py`).

Example:

```python
# plugins/tests/test_my_game_plugin.py
import pytest
from watchdog_env.plugins.my_game import MyGamePlugin, MyGameConfig
from watchdog_env.plugins.registry import get_plugin

def test_get_game_id():
    plugin = MyGamePlugin()
    assert plugin.get_game_id() == "my_game"

def test_reset_and_generate_step():
    plugin = MyGamePlugin()
    plugin.reset(seed=1, config=MyGameConfig(num_rounds=2))
    step0 = plugin.generate_step(1, 0)
    assert len(step0.turns) >= 1
    assert step0.done is False
    step1 = plugin.generate_step(1, 1)
    assert step1.done is True

def test_registered():
    assert get_plugin("my_game") is not None
```

## 6. Document config and usage

In your plugin module or in this README, document:

- Supported **config** fields (your MultiAgentConfig subclass).
- Any **env vars** (e.g. API keys) and optional dependencies.
- How to run your plugin’s tests (e.g. “Set MY_API_KEY and run pytest …”).

---

## Summary checklist

- [ ] New folder under `watchdog_env/plugins/<your_game>/`
- [ ] Config class (subclass of MultiAgentConfig) if needed
- [ ] Plugin class implementing all 6 methods of MultiAgentSystemPlugin
- [ ] generate_step uses state history (e.g. turns_so_far) and updates state
- [ ] Export in `<your_game>/__init__.py` and register in `plugins/__init__.py`
- [ ] Update pyproject.toml packages if adding a new plugin package
- [ ] Tests for all methods; skip or require API key as appropriate
- [ ] Short doc for config and how to run tests
