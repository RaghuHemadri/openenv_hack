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

from openenv.core.env_server.http_server import create_app

from models import WatchDogAction, WatchDogObservation
from .watchdog_environment import WatchDogEnvironment

app = create_app(
    WatchDogEnvironment,
    WatchDogAction,
    WatchDogObservation,
    env_name="watchdog_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
