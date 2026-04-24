from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image, ImageDraw

_CSS = """
:root{
  --bg:#06101a;
  --panel:#0b1624;
  --panel2:#09111d;
  --stroke:#1a2a40;
  --muted:#8091aa;
  --text:#e8f0fb;
  --cyan:#00d8ff;
  --cyan2:#14a7d8;
  --good:#30d18a;
  --bad:#ff6d6d;
}

footer{display:none !important;}
.gradio-container{background:var(--bg) !important;}
.wrap{
  background:
    radial-gradient(900px 620px at 20% 10%, rgba(0,216,255,.08), transparent 56%),
    radial-gradient(720px 560px at 90% 20%, rgba(124,92,252,.08), transparent 60%),
    var(--bg);
  border:1px solid var(--stroke);
  border-radius:16px;
  padding:12px;
}
.topbar{
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:12px;
  margin-bottom:10px;
}
.title{color:var(--text); font-weight:800; font-size:1.03rem; letter-spacing:.02em;}
.subtitle{color:var(--muted); font-size:.78rem; margin-top:2px;}
.panel,.boardPanel{
  border:1px solid var(--stroke);
  border-radius:14px;
  background:linear-gradient(180deg, rgba(255,255,255,.03), transparent), var(--panel);
  padding:10px;
}
.boardPanel{background:linear-gradient(180deg, rgba(255,255,255,.03), transparent), var(--panel2);}
.stats{display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end;}
.stat{
  min-width:120px;
  border:1px solid var(--stroke);
  border-radius:12px;
  padding:8px 10px;
  background:rgba(255,255,255,.03);
}
.k{color:var(--muted); font-size:.68rem; text-transform:uppercase; letter-spacing:.1em;}
.v{color:var(--text); font-size:1.18rem; font-weight:800; margin-top:2px;}
.vGood{color:var(--good);}
.vBad{color:var(--bad);}
.hint{color:var(--muted); font-size:.78rem; line-height:1.45; margin-top:8px;}
.chips{display:flex; flex-wrap:wrap; gap:6px; margin-top:6px;}
.chip{
  border:1px solid rgba(0,216,255,.22);
  background:rgba(0,216,255,.08);
  color:var(--cyan);
  border-radius:999px;
  padding:4px 8px;
  font-size:.72rem;
  line-height:1;
}
.section{margin-top:8px;}
"""


def _safe_obs(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    obs = data.get("observation", {})
    return obs if isinstance(obs, dict) else {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_grid(obs: Dict[str, Any]) -> Optional[List[List[int]]]:
    grid = obs.get("canva_space")
    if isinstance(grid, list) and grid and isinstance(grid[0], list):
        return grid
    return None


def _grid_shape(grid: Optional[List[List[int]]]) -> Tuple[int, int]:
    if not grid:
        return 0, 0
    return len(grid), len(grid[0]) if grid and grid[0] else 0


def _cell_px(n: int, m: int, target: int = 660) -> int:
    side = max(n, m, 1)
    return max(14, min(30, int((target - 44) / side)))


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    i = int(h * 6)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return int(r * 255), int(g * 255), int(b * 255)


def _block_color(cell: int) -> Tuple[int, int, int]:
    hues = (190, 265, 145, 35, 310, 175, 55, 330, 200, 85, 240, 120)
    h = hues[abs(cell) % len(hues)] / 360.0
    return _hsv_to_rgb(h, 0.76, 0.84)


def _render_block_chips(blocks: Optional[List[Any]]) -> str:
    if not isinstance(blocks, list) or not blocks:
        return "<div class='hint'>No remaining blocks.</div>"

    chips = []
    for block in blocks:
        if isinstance(block, dict):
            bid = block.get("id", "?")
            h = block.get("height", "?")
            w = block.get("width", "?")
            chips.append(f"<span class='chip'>{bid} · {w}x{h}</span>")
        else:
            chips.append("<span class='chip'>block</span>")
    return "<div class='chips'>" + "".join(chips) + "</div>"


def _render_status(data: Dict[str, Any], selected_task: str) -> str:
    obs = _safe_obs(data)
    reward = _safe_float(data.get("reward")) or 0.0
    step = int(_safe_float(obs.get("phase_step")) or 0)
    hpwl = _safe_float(obs.get("current_hpwl"))
    delta_hpwl = _safe_float(obs.get("delta_hpwl"))
    done = bool(obs.get("done", data.get("done", False)))
    task_name = str(obs.get("task_name") or selected_task or "unknown")
    phase = str(obs.get("phase") or "placement")
    reward_class = "vGood" if reward >= 0 else "vBad"
    reward_sign = "+" if reward >= 0 else ""

    hpwl_html = ""
    if hpwl is not None:
        if delta_hpwl is not None:
            arrow = "↓" if delta_hpwl < 0 else "↑" if delta_hpwl > 0 else "→"
            hpwl_html = f"<div class='k'>HPWL</div><div class='v'>{hpwl:.2f} <span style='color:var(--muted);font-size:.9rem;font-weight:700;'>({delta_hpwl:+.2f} {arrow})</span></div>"
        else:
            hpwl_html = f"<div class='k'>HPWL</div><div class='v'>{hpwl:.2f}</div>"

    return (
        "<div class='stats'>"
        "<div class='stat'>"
        "<div class='k'>Reward</div>"
        f"<div class='v {reward_class}'>{reward_sign}{reward:.4f}</div>"
        "</div>"
        "<div class='stat'>"
        "<div class='k'>Step</div>"
        f"<div class='v'>{step}</div>"
        "</div>"
        "<div class='stat'>"
        "<div class='k'>Task</div>"
        f"<div class='v' style='font-size:1rem;'>{task_name}</div>"
        "</div>"
        "<div class='stat'>"
        "<div class='k'>Phase</div>"
        f"<div class='v' style='font-size:1rem;'>{phase}</div>"
        "</div>"
        "<div class='stat'>"
        f"{hpwl_html or '<div class=\"k\">HPWL</div><div class=\"v\">-</div>'}"
        "</div>"
        "<div class='stat'>"
        "<div class='k'>Done</div>"
        f"<div class='v'>{'Yes' if done else 'No'}</div>"
        "</div>"
        "</div>"
    )


def _render_board(
    grid: Optional[List[List[int]]],
    prev_grid: Optional[List[List[int]]],
    preview_xy: Optional[Tuple[int, int]],
    placed_xy: Optional[Tuple[int, int]],
    reward: float,
    blend: float,
    reset_fade: float,
    cell_px: int,
    pad: int,
) -> Image.Image:
    if not grid:
        img = Image.new("RGB", (960, 620), (6, 16, 27))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((18, 18, 942, 602), radius=18, outline=(26, 38, 60), width=2, fill=(10, 18, 30))
        d.text((36, 40), "Reset the environment to load the board.", fill=(128, 145, 170))
        d.text((36, 66), "Pick a task, then place blocks by choosing coordinates and a block id.", fill=(90, 110, 130))
        return img

    n, m = _grid_shape(grid)
    w = pad * 2 + m * cell_px
    h = pad * 2 + n * cell_px
    img = Image.new("RGB", (w, h), (6, 16, 27))
    d = ImageDraw.Draw(img, "RGBA")

    d.rounded_rectangle((4, 4, w - 4, h - 4), radius=16, fill=(10, 18, 30), outline=(26, 38, 60), width=1)

    if reward:
        strength = max(0.0, min(1.0, (1.0 - blend) * 0.55))
        if strength > 0:
            color = (28, 150, 100) if reward >= 0 else (160, 55, 60)
            overlay = Image.new("RGBA", (w, h), (*color, int(42 * strength)))
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            d = ImageDraw.Draw(img, "RGBA")

    line = (24, 36, 56)
    for x in range(m + 1):
        xx = pad + x * cell_px
        d.line((xx, pad, xx, pad + n * cell_px), fill=line, width=1)
    for y in range(n + 1):
        yy = pad + y * cell_px
        d.line((pad, yy, pad + m * cell_px, yy), fill=line, width=1)

    def cell_at(g: Optional[List[List[int]]], x: int, y: int) -> int:
        if not g:
            return 0
        return g[y][x] if 0 <= y < len(g) and 0 <= x < len(g[y]) else 0

    for y in range(n):
        for x in range(m):
            cur = cell_at(grid, x, y)
            prev = cell_at(prev_grid, x, y)
            x0 = pad + x * cell_px + 1
            y0 = pad + (n - 1 - y) * cell_px + 1
            x1 = x0 + cell_px - 2
            y1 = y0 + cell_px - 2

            if cur == 0:
                base = (10, 18, 30)
                if reset_fade > 0:
                    f = max(0.0, min(1.0, reset_fade))
                    base = (int(base[0] * (1 - f)), int(base[1] * (1 - f)), int(base[2] * (1 - f)))
                d.rectangle((x0, y0, x1, y1), fill=base)
                continue

            is_new = prev == 0 and cur != 0
            if is_new and 0.0 <= blend <= 1.0:
                scale = 0.60 + 0.40 * blend
                lift = int((1.0 - blend) * (cell_px * 0.18))
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                hw = (x1 - x0) * scale / 2
                hh = (y1 - y0) * scale / 2
                x0r = int(cx - hw)
                y0r = int(cy - hh - lift)
                x1r = int(cx + hw)
                y1r = int(cy + hh - lift)
            else:
                x0r, y0r, x1r, y1r = x0, y0, x1, y1

            base = _block_color(cur)
            shell = (base[0] // 4, base[1] // 4, base[2] // 4)
            core = (
                min(255, int(base[0] * (0.74 + 0.26 * blend))),
                min(255, int(base[1] * (0.74 + 0.26 * blend))),
                min(255, int(base[2] * (0.74 + 0.26 * blend))),
            )
            d.rounded_rectangle((x0r, y0r, x1r, y1r), radius=6, fill=shell)
            d.rounded_rectangle((x0r + 3, y0r + 3, x1r - 3, y1r - 3), radius=5, fill=core)
            d.rounded_rectangle((x0r, y0r, x1r, y1r), radius=6, outline=(0, 216, 255), width=1)

    if preview_xy is not None:
        px, py = preview_xy
        if 0 <= px < m and 0 <= py < n:
            x0 = pad + px * cell_px + 1
            y0 = pad + (n - 1 - py) * cell_px + 1
            x1 = x0 + cell_px - 2
            y1 = y0 + cell_px - 2
            d.rounded_rectangle((x0, y0, x1, y1), radius=6, outline=(0, 216, 255), width=2)
            d.rounded_rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), radius=5, outline=(0, 216, 255), width=1)

    if placed_xy is not None:
        ax, ay = placed_xy
        if 0 <= ax < m and 0 <= ay < n:
            x0 = pad + ax * cell_px + 1
            y0 = pad + (n - 1 - ay) * cell_px + 1
            x1 = x0 + cell_px - 2
            y1 = y0 + cell_px - 2
            for k, a in [(0, 180), (3, 90), (6, 40)]:
                d.rounded_rectangle((x0 - k, y0 - k, x1 + k, y1 + k), radius=6 + k, outline=(0, 216, 255, a), width=2)

    return img


def _task_choices() -> List[Tuple[str, str]]:
    try:
        try:
            from .chip_flooring_env_environment import ChipFlooringEnvironment
        except ImportError:
            from server.chip_flooring_env_environment import ChipFlooringEnvironment

        env = ChipFlooringEnvironment()
        preferred = ["easy", "medium", "hard", "heterogeneous", "fixed_obstacles", "long_horizon"]
        task_names = [name for name in preferred if name in env.task_configs]
        task_names.extend(name for name in env.task_configs.keys() if name not in task_names)
        return [(name.replace("_", " ").title(), name) for name in task_names]
    except Exception:
        fallback = os.getenv("TASK_NAME", "hard_standard_long_horizon")
        return [(fallback.replace("_", " ").title(), fallback)]


async def _reset_environment(web_manager: Any, task_name: Optional[str]) -> Dict[str, Any]:
    payload = {"task_name": task_name} if task_name else None
    if payload is not None:
        try:
            return await web_manager.reset_environment(payload)
        except TypeError:
            try:
                return await web_manager.reset_environment(task_name=task_name)
            except TypeError:
                pass
        except Exception:
            pass
    return await web_manager.reset_environment()


def _parse_block_index(choice: Optional[str]) -> int:
    try:
        return int(str(choice or "0").split("—", 1)[0].strip())
    except ValueError:
        return 0


def _block_choices(obs: Dict[str, Any]) -> List[str]:
    blocks = obs.get("remaining_blocks")
    if not isinstance(blocks, list) or not blocks:
        return ["0 — reset"]

    out: List[str] = []
    for i, block in enumerate(blocks):
        if isinstance(block, dict):
            bid = block.get("id", "?")
            h = block.get("height", "?")
            w = block.get("width", "?")
            out.append(f"{i} — {bid} ({w}x{h})")
        else:
            out.append(f"{i} — block")
    return out


def build_clean_gradio_ui(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    task_options = _task_choices()
    default_task = os.getenv("TASK_NAME", task_options[0][1] if task_options else "hard_standard_long_horizon")
    if default_task not in [value for _, value in task_options]:
        task_options = task_options + [(default_task.replace("_", " ").title(), default_task)]

    state0: Dict[str, Any] = {
        "grid": None,
        "prev_grid": None,
        "step": 0,
        "reward": 0.0,
        "hpwl": None,
        "delta_hpwl": None,
        "preview_xy": (0, 0),
        "last_action_xy": None,
        "task_name": default_task,
        "cell_px": 22,
        "pad": 18,
    }

    def _board_for_state(s: Dict[str, Any], blend: float = 1.0, reset_fade: float = 0.0) -> Image.Image:
        return _render_board(
            grid=s.get("grid"),
            prev_grid=s.get("prev_grid"),
            preview_xy=s.get("preview_xy"),
            placed_xy=s.get("last_action_xy"),
            reward=float(s.get("reward", 0.0) or 0.0),
            blend=blend,
            reset_fade=reset_fade,
            cell_px=int(s.get("cell_px", 22)),
            pad=int(s.get("pad", 18)),
        )

    def _task_banner(task_name: str) -> str:
        return (
            "<div class='hint'>"
            f"<strong>Selected task:</strong> {task_name}. "
            "Choose a task, press Reset, then place the next block on the board."
            "</div>"
        )

    with gr.Blocks(title=f"{title} — Clean UI", css=_CSS) as demo:
        s_state = gr.State(dict(state0))

        with gr.Group(elem_classes=["wrap"]):
            with gr.Row(elem_classes=["topbar"]):
                with gr.Column(scale=6):
                    gr.HTML(
                        f"<div class='title'>{title}</div>"
                        "<div class='subtitle'>Human-facing placement interface. No raw observation JSON is shown.</div>"
                    )
                with gr.Column(scale=5):
                    status_panel = gr.HTML(value=_render_status({}, default_task))

            with gr.Row(equal_height=False):
                with gr.Column(scale=9):
                    with gr.Group(elem_classes=["boardPanel"]):
                        board = gr.Image(
                            value=_board_for_state(state0),
                            show_label=False,
                            interactive=True,
                            type="pil",
                            height=680,
                        )
                        gr.HTML("<div class='hint'>Click a cell to set coordinates. The cyan outline is the current preview.</div>")
                with gr.Column(scale=3):
                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML("<h3 style='margin:0 0 8px 0;'>Controls</h3>")
                        task_dd = gr.Dropdown(
                            label="Task",
                            choices=task_options,
                            value=default_task,
                        )
                        x_in = gr.Number(label="X", value=0, precision=0, minimum=0)
                        y_in = gr.Number(label="Y", value=0, precision=0, minimum=0)
                        block_dd = gr.Dropdown(label="Remaining block", choices=["0 — reset"], value="0 — reset")
                        animate = gr.Checkbox(value=True, label="Animate placements")
                        with gr.Row():
                            step_btn = gr.Button("Step", variant="primary")
                            reset_btn = gr.Button("Reset", variant="secondary")
                        gr.HTML("<div class='section'>Remaining blocks</div>")
                        remaining_html = gr.HTML(value=_render_block_chips(None))
                        task_hint = gr.HTML(value=_task_banner(default_task))

        async def do_reset(task_name: str, s: Dict[str, Any]):
            s = dict(s or state0)
            old = dict(s)
            if old.get("grid") is not None:
                for rf in (0.0, 0.35, 0.65):
                    yield (
                        _board_for_state(old, blend=1.0, reset_fade=rf),
                        gr.update(),
                        _render_status({}, task_name),
                        _render_block_chips(None),
                        _task_banner(task_name),
                        gr.update(value=task_name),
                        s,
                    )
                    await asyncio.sleep(0.03)

            data = await _reset_environment(web_manager, task_name)
            obs = _safe_obs(data)
            grid = _extract_grid(obs)
            n, m = _grid_shape(grid)
            remaining = obs.get("remaining_blocks")
            dd_choices = _block_choices(obs)
            actual_task = str(obs.get("task_name") or task_name or default_task)

            s["grid"] = grid
            s["prev_grid"] = None
            s["step"] = 0
            s["reward"] = float(_safe_float(data.get("reward")) or 0.0)
            s["hpwl"] = _safe_float(obs.get("current_hpwl"))
            s["delta_hpwl"] = _safe_float(obs.get("delta_hpwl"))
            s["preview_xy"] = (0, 0)
            s["last_action_xy"] = None
            s["task_name"] = actual_task
            s["cell_px"] = _cell_px(n, m, target=680)

            status = _render_status(data if isinstance(data, dict) else {}, actual_task)
            yield (
                _board_for_state(s),
                gr.update(choices=dd_choices, value=dd_choices[0] if dd_choices else None),
                status,
                _render_block_chips(remaining),
                _task_banner(actual_task),
                gr.update(value=actual_task),
                s,
            )

        async def do_step(x: float, y: float, choice: str, animate_on: bool, s: Dict[str, Any]):
            s = dict(s or state0)
            xi, yi = int(x or 0), int(y or 0)
            s["preview_xy"] = (xi, yi)
            idx = _parse_block_index(choice)
            action_data = {"x": xi, "y": yi, "choosen_block_index": idx}

            data = await web_manager.step_environment(action_data)
            obs = _safe_obs(data)
            grid = _extract_grid(obs)
            remaining = obs.get("remaining_blocks")

            s["prev_grid"] = s.get("grid")
            s["grid"] = grid
            s["reward"] = float(_safe_float(data.get("reward")) or 0.0)
            s["hpwl"] = _safe_float(obs.get("current_hpwl"))
            s["delta_hpwl"] = _safe_float(obs.get("delta_hpwl"))
            s["step"] = int(s.get("step", 0)) + 1
            s["last_action_xy"] = (xi, yi)
            s["task_name"] = str(obs.get("task_name") or s.get("task_name") or default_task)

            dd_choices = _block_choices(obs)
            status = _render_status(data if isinstance(data, dict) else {}, str(s["task_name"]))
            remaining_html = _render_block_chips(remaining)
            task_banner = _task_banner(str(s["task_name"]))

            if animate_on and s.get("prev_grid") is not None and s.get("grid") is not None:
                for blend in (0.15, 0.45, 0.75, 1.0):
                    yield (
                        _board_for_state(s, blend=blend),
                        gr.update(choices=dd_choices, value=choice if choice in dd_choices else (dd_choices[0] if dd_choices else None)),
                        status,
                        remaining_html,
                        task_banner,
                        gr.update(value=str(s["task_name"])),
                        s,
                    )
                    await asyncio.sleep(0.05)
            else:
                yield (
                    _board_for_state(s),
                    gr.update(choices=dd_choices, value=choice if choice in dd_choices else (dd_choices[0] if dd_choices else None)),
                    status,
                    remaining_html,
                    task_banner,
                    gr.update(value=str(s["task_name"])),
                    s,
                )

        def do_preview_from_click(evt: gr.SelectData, s: Dict[str, Any]):
            s = dict(s or state0)
            grid = s.get("grid")
            n, m = _grid_shape(grid)
            if n == 0 or m == 0:
                return gr.update(), gr.update(), s, _board_for_state(s)

            try:
                px, py = evt.index  # type: ignore[misc]
            except Exception:
                return gr.update(), gr.update(), s, _board_for_state(s)

            cell = int(s.get("cell_px", 22))
            pad = int(s.get("pad", 18))
            x = int((px - pad) // cell)
            y = n - 1 - int((py - pad) // cell)
            if x < 0 or y < 0 or x >= m or y >= n:
                return gr.update(), gr.update(), s, _board_for_state(s)

            s["preview_xy"] = (x, y)
            return gr.update(value=x), gr.update(value=y), s, _board_for_state(s)

        def do_task_change(task_name: str, s: Dict[str, Any]):
            s = dict(s or state0)
            s["task_name"] = str(task_name or default_task)
            return _task_banner(s["task_name"]), s

        task_dd.change(fn=do_task_change, inputs=[task_dd, s_state], outputs=[task_hint, s_state])
        reset_btn.click(fn=do_reset, inputs=[task_dd, s_state], outputs=[board, block_dd, status_panel, remaining_html, task_hint, task_dd, s_state])
        step_btn.click(fn=do_step, inputs=[x_in, y_in, block_dd, animate, s_state], outputs=[board, block_dd, status_panel, remaining_html, task_hint, task_dd, s_state])
        board.select(fn=do_preview_from_click, inputs=[s_state], outputs=[x_in, y_in, s_state, board])

        x_in.change(
            fn=lambda x, y, s: (dict({**(s or state0), "preview_xy": (int(x or 0), int(y or 0))}), _board_for_state({**(s or state0), "preview_xy": (int(x or 0), int(y or 0))})),
            inputs=[x_in, y_in, s_state],
            outputs=[s_state, board],
        )
        y_in.change(
            fn=lambda x, y, s: (dict({**(s or state0), "preview_xy": (int(x or 0), int(y or 0))}), _board_for_state({**(s or state0), "preview_xy": (int(x or 0), int(y or 0))})),
            inputs=[x_in, y_in, s_state],
            outputs=[s_state, board],
        )

        demo.load(fn=lambda: _board_for_state(state0), inputs=[], outputs=[board])

    return demo
