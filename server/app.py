# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Chip Flooring Env Environment.

This module creates an HTTP server that exposes the ChipFlooringEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os

try:
    from openenv.core.env_server.http_server import create_fastapi_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ChipFlooringAction, ChipFlooringObservation
    from .chip_flooring_env_environment import ChipFlooringEnvironment
    from .graders import GRADERS
except ImportError:
    from models import ChipFlooringAction, ChipFlooringObservation
    from server.chip_flooring_env_environment import ChipFlooringEnvironment
    from server.graders import GRADERS
from fastapi import Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pathlib


# Create the standard API app
app = create_fastapi_app(
    ChipFlooringEnvironment,
    ChipFlooringAction,
    ChipFlooringObservation,
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
@app.get("/web/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>UI is being built...</h1>"

_TASK_GRADER_SPECS: dict[str, str] = {
    "easy": "server.graders.grade_easy",
    "medium": "server.graders.grade_medium",
    "hard": "server.graders.grade_hard",
    "heterogeneous": "server.graders.grade_heterogeneous",
    "fixed_obstacles": "server.graders.grade_fixed_obstacles",
    "long_horizon": "server.graders.grade_long_horizon",
}


def _parse_task_name(task_name: str) -> dict[str, str]:
    normalized = str(task_name or "").strip().lower().replace("-", "_")
    if normalized == "long_horizon":
        return {"difficulty": "hard", "scenario": "heterogeneous", "horizon": "long_horizon"}
    if normalized.endswith("_long_horizon"):
        base = normalized[: -len("_long_horizon")]
        parts = base.split("_", 1)
        if len(parts) == 2:
            return {
                "difficulty": parts[0] or "hard",
                "scenario": parts[1] or "standard",
                "horizon": "long_horizon",
            }
    return {"difficulty": normalized, "scenario": "standard", "horizon": "short_horizon"}


def _normalize_score(score: float) -> float:
    return round(max(0.01, min(0.99, score)), 2)


def _task_summary() -> list[dict[str, object]]:
    env = ChipFlooringEnvironment()
    tasks: list[dict[str, object]] = []
    for task_name, config in env.task_configs.items():
        if task_name == "long_horizon":
            continue
        task_meta = _parse_task_name(task_name)
        total_blocks = len(config["nodes"])
        fixed_blocks = sum(1 for node in config["nodes"] if node.get("fixed"))
        movable_blocks = total_blocks - fixed_blocks
        max_steps = int(config.get("phase_finalize_step", movable_blocks))
        if task_meta["horizon"] == "long_horizon":
            description = (
                f"Plan, reveal, and repair a {task_meta['difficulty']} {task_meta['scenario']} layout on a {config['grid_size']}x{config['grid_size']} grid "
                f"with hidden constraints, delayed reward, and a later repair phase."
            )
        else:
            description = (
                (
                    f"Place {movable_blocks} movable blocks with {fixed_blocks} pre-placed fixed blocks "
                    f"on a {config['grid_size']}x{config['grid_size']} grid while minimizing criticality-weighted wirelength and keeping the placement legal."
                    if fixed_blocks > 0
                    else f"Place {total_blocks} connected blocks on a {config['grid_size']}x{config['grid_size']} grid while minimizing criticality-weighted wirelength and keeping the placement legal."
                )
            )
        tasks.append(
            {
                "id": task_name,
                "difficulty": task_meta["difficulty"],
                "scenario": task_meta["scenario"],
                "horizon": task_meta["horizon"],
                "description": description,
                "grid_size": config["grid_size"],
                "block_count": movable_blocks,
                "max_steps": max_steps,
                "total_block_count": total_blocks,
                "fixed_block_count": fixed_blocks,
                "score_range": [0.01, 0.99],
                "grader": task_name in GRADERS,
                "grader_ref": _TASK_GRADER_SPECS.get(task_name, _TASK_GRADER_SPECS.get(task_meta["horizon"], "")),
            }
        )
    return tasks


@app.get("/tasks")
def list_tasks():
    return {"tasks": _task_summary()}


@app.post("/grader")
def grader(payload: dict | None = Body(default=None)):
    payload = payload or {}
    task_name = str(
        payload.get("task_name")
        or payload.get("task_id")
        or os.getenv("TASK_NAME", "hard_standard_long_horizon")
    ).strip().lower() or "hard_standard_long_horizon"
    grader_fn = GRADERS.get(task_name)
    if grader_fn is not None:
        result = dict(grader_fn(payload))
        result["task_name"] = task_name
        return result

    raw_score = payload.get("score", payload.get("reward", 0.0))
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "task_name": task_name,
        "score": _normalize_score(score),
        "grader_type": "deterministic",
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m chip_flooring_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn chip_flooring_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port == 8000:
        main()
    else:
        main(port=args.port)
