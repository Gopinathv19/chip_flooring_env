from __future__ import annotations

import json
from typing import Any, Dict, List

import gradio as gr


def _format_board(board_ascii: str) -> str:
    board_ascii = (board_ascii or "").rstrip()
    if not board_ascii:
        return "```text\nReset the environment to view the board.\n```"
    return f"```text\n{board_ascii}\n```"


def _format_status(data: Dict[str, Any]) -> str:
    reward = data.get("reward")
    done = data.get("done")
    observation = data.get("observation", {})
    task_name = observation.get("task_name", "unknown") if isinstance(observation, dict) else "unknown"
    invalid = observation.get("invalid_reasons") if isinstance(observation, dict) else None
    parts = [f"Reward: `{reward}`", f"Done: `{done}`", f"Task: `{task_name}`"]
    if invalid:
        parts.append(f"Invalid: `{invalid}`")
    return " | ".join(parts)


def build_clean_gradio_ui(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """Build a compact board-first UI for the chip flooring environment."""

    async def reset_env():
        data = await web_manager.reset_environment()
        obs = data.get("observation", {}) if isinstance(data, dict) else {}
        board = obs.get("board_ascii", "") if isinstance(obs, dict) else ""
        return (
            _format_board(board),
            _format_status(data if isinstance(data, dict) else {}),
            json.dumps(data, indent=2),
        )

    async def step_env(x: int, y: int, choosen_block_index: int):
        action_data = {
            "x": int(x or 0),
            "y": int(y or 0),
            "choosen_block_index": int(choosen_block_index or 0),
        }
        data = await web_manager.step_environment(action_data)
        obs = data.get("observation", {}) if isinstance(data, dict) else {}
        board = obs.get("board_ascii", "") if isinstance(obs, dict) else ""
        return (
            _format_board(board),
            _format_status(data if isinstance(data, dict) else {}),
            json.dumps(data, indent=2),
        )

    with gr.Blocks(title=f"{title} - Clean Board") as demo:
        gr.Markdown(
            f"# {title}\n"
            "A compact board view with positive-only coordinates and ASCII placement preview."
        )
        with gr.Row():
            with gr.Column(scale=2):
                board = gr.Markdown(value=_format_board(""))
                status = gr.Markdown(value="Reset the environment to begin.")
                raw_json = gr.Code(label="Raw JSON response", language="json")
            with gr.Column(scale=1):
                gr.Markdown("## Action")
                x = gr.Number(label="X", value=0, precision=0, minimum=0)
                y = gr.Number(label="Y", value=0, precision=0, minimum=0)
                choosen_block_index = gr.Number(
                    label="Choosen Block Index",
                    value=0,
                    precision=0,
                    minimum=0,
                )
                with gr.Row():
                    step_btn = gr.Button("Step", variant="primary")
                    reset_btn = gr.Button("Reset", variant="secondary")
        with gr.Accordion("Quick Start", open=False):
            gr.Markdown(quick_start_md)
        with gr.Accordion("README", open=False):
            gr.Markdown(getattr(metadata, "readme_content", "") or "*No README available.*")

        step_btn.click(
            step_env,
            inputs=[x, y, choosen_block_index],
            outputs=[board, status, raw_json],
        )
        reset_btn.click(
            reset_env,
            inputs=[],
            outputs=[board, status, raw_json],
        )

    return demo
