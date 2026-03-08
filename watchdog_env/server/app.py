"""
FastAPI application for the WatchDog Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app

from models import MultiTurnAction, MultiTurnObservation
from .watchdog_environment import WatchDogMultiTurnEnvironment

# Ensure plugins are registered (Avalon, Cicero)
try:
    import plugins  # noqa: F401
except ImportError:
    import watchdog_env.plugins  # noqa: F401

app = create_app(
    WatchDogMultiTurnEnvironment,
    MultiTurnAction,
    MultiTurnObservation,
    env_name="watchdog_env",
    max_concurrent_envs=4,
)


@app.get("/")
def root():
    """Root endpoint with available API paths."""
    return {
        "message": "WatchDog OpenEnv Environment API",
        "endpoints": {
            "health": "GET /health",
            "schema": "GET /schema",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "docs": "GET /docs",
            "ws": "WS /ws",
        },
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
