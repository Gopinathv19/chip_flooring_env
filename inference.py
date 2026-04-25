import ast
import json
import os
import re
import sys
from time import sleep
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
ENV_DIR = ROOT
SERVER_DIR = ROOT / "server"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(ROOT / ".env")

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = (
    os.getenv("API_BASE_URL")
    or os.getenv("LOCAL_API_BASE_URL")
    or "https://router.huggingface.co/v1"
)
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
TASK_NAME = (
    os.getenv("OPENENV_CHIP_FLOORING_TASK")
    or os.getenv("TASK_NAME")
    or "hard_standard_long_horizon"
).strip().lower() or "hard_standard_long_horizon"
TASKS_TO_RUN_ENV = os.getenv("OPENENV_CHIP_FLOORING_TASKS")
if TASKS_TO_RUN_ENV:
    TASKS_TO_RUN = [task.strip().lower() for task in TASKS_TO_RUN_ENV.split(",") if task.strip()]
else:
    TASKS_TO_RUN = [TASK_NAME]
BENCHMARK = (
    os.getenv("OPENENV_CHIP_FLOORING_BENCHMARK")
    or os.getenv("BENCHMARK")
    or f"chip_flooring_{TASK_NAME}"
)
MAX_STEPS = int(os.getenv("MAX_STEPS") or "100")
TEMPERATURE = float(os.getenv("TEMPERATURE") or "0.2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS") or "120")
MODEL_CONNECTION_RETRY_SLEEP_MS = int(os.getenv("MODEL_CONNECTION_RETRY_SLEEP_MS") or "250")


def load_module(module_name: str, path: Path):
    spec = importlib_util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib_util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


models_mod = load_module("chip_flooring_env.models", ENV_DIR / "models.py")
env_mod = load_module(
    "chip_flooring_env.server.chip_flooring_env_environment",
    SERVER_DIR / "chip_flooring_env_environment.py",
)

ChipFlooringAction = models_mod.ChipFlooringAction
ChipFlooringEnvironment = env_mod.ChipFlooringEnvironment


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def is_long_horizon_task_name(task_name: str) -> bool:
    normalized = str(task_name or "").strip().lower()
    return normalized.endswith("_long_horizon") or normalized == "long_horizon"


def summarize_history(history: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    return history[-limit:]


def compact_block_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": summary.get("id"),
        "priority": round(float(summary.get("priority_score", 0.0)), 3),
        "committed": bool(summary.get("committed", False)),
        "move_count": int(summary.get("move_count", 0) or 0),
        "placed_neighbors": [
            {
                "id": neighbor.get("id"),
                "weight": round(float(neighbor.get("weight", 0.0)), 3),
                "pos": neighbor.get("position"),
            }
            for neighbor in (summary.get("placed_neighbors") or [])[:3]
        ],
        "strongest": [
            {
                "id": neighbor.get("id"),
                "weight": round(float(neighbor.get("weight", 0.0)), 3),
                "placed": bool(neighbor.get("placed")),
            }
            for neighbor in (summary.get("strongest_neighbors") or [])[:3]
        ],
    }


def generate_candidate_actions(
    env: ChipFlooringEnvironment,
    remaining_blocks: List[Dict[str, Any]],
    placed_blocks: Optional[List[Dict[str, Any]]] = None,
    phase: str = "placement",
    per_block_limit: int = 1,
) -> List[Dict[str, Any]]:
    if env.canvas is None:
        return []

    candidates: List[Dict[str, Any]] = []
    sorted_blocks = sorted(
        remaining_blocks,
        key=lambda block: (-int(block["height"]) * int(block["width"]), str(block["id"])),
    )

    for block in sorted_blocks:
        height = int(block["height"])
        width = int(block["width"])
        found = 0
        for row in range(env.grid_size):
            for col in range(env.grid_size):
                if env.canvas.can_occupy((row, col), width, height):
                    candidates.append(
                        {
                            "block_id": block["id"],
                            "x": row,
                            "y": col,
                            "height": height,
                            "width": width,
                            "area": height * width,
                        }
                    )
                    found += 1
                    if found >= per_block_limit:
                        break
            if found >= per_block_limit:
                break

    if phase in {"repair", "finalize"} and placed_blocks:
        movable_blocks = sorted(
            [block for block in placed_blocks if not bool(block.get("fixed")) and block.get("position") is not None],
            key=lambda block: (-float(block.get("power", 1.0)), str(block["id"])),
        )
        for block in movable_blocks[:4]:
            row0, col0 = block["position"]
            height = int(block["height"])
            width = int(block["width"])
            block_id = str(block["id"])
            block_num = next(
                (idx + 1 for idx, existing in enumerate(env.state.blocks) if existing.id == block_id),
                None,
            )
            if block_num is None:
                continue
            candidates.append(
                {
                    "block_id": block_id,
                    "x": int(row0),
                    "y": int(col0),
                    "height": height,
                    "width": width,
                    "area": height * width,
                    "action_type": "commit",
                }
            )
            env.canvas.remove_region((int(row0), int(col0)), width, height)
            found = 0
            for row in range(env.grid_size):
                for col in range(env.grid_size):
                    if env.canvas.can_occupy((row, col), width, height):
                        candidates.append(
                            {
                                "block_id": block_id,
                                "x": row,
                                "y": col,
                                "height": height,
                                "width": width,
                                "area": height * width,
                                "action_type": "move",
                            }
                        )
                        found += 1
                        if found >= per_block_limit:
                            break
                if found >= per_block_limit:
                    break
            env.canvas.occupy_region((int(row0), int(col0)), width, height, block_num)

    candidates.sort(
        key=lambda item: (
            str(item.get("action_type", "place")),
            -float(item.get("area", 0.0)),
            str(item["block_id"]),
            int(item["x"]),
            int(item["y"]),
        )
    )
    return candidates


def build_prompt(
    step: int,
    placed_blocks: List[Dict[str, Any]],
    remaining_blocks: List[Dict[str, Any]],
    phase: str,
    instruction: str,
    block_summaries: List[Dict[str, Any]],
    placement_focus: Optional[Dict[str, Any]],
    density_map: List[List[float]],
    total_reward: float,
    recent_history: List[Dict[str, Any]],
    candidate_actions: List[Dict[str, Any]],
    previous_failure: str = "",
) -> str:
    focus_compact = compact_block_summary(placement_focus) if placement_focus else None
    summaries_compact = [compact_block_summary(summary) for summary in block_summaries[:6]]
    candidates_compact = candidate_actions[:8]
    return (
        "Choose one action from the candidate actions.\n"
        "Return JSON only with one of these forms: {\"block_id\":\"...\",\"x\":0,\"y\":0,\"action_type\":\"place\"}, {\"block_id\":\"...\",\"x\":0,\"y\":0,\"action_type\":\"move\"}, or {\"block_id\":\"...\",\"x\":0,\"y\":0,\"action_type\":\"commit\"}.\n"
        "Use action_type=\"move\" only when the episode is in repair or finalize and the candidate is a relocation.\n"
        "Use action_type=\"commit\" in repair or finalize when the current position should be locked and no further relocation is needed.\n"
        "Prefer the action that best reduces wirelength while staying legal.\n"
        "Use the focus block, placed neighbor positions, phase, instruction, and candidate scores.\n\n"
        f"step={step}\n"
        f"phase={phase}\n"
        f"instruction={instruction}\n"
        f"placed_count={len(placed_blocks)} remaining_count={len(remaining_blocks)} total_reward={total_reward:.2f}\n"
        f"focus={json.dumps(focus_compact)}\n"
        f"summaries={json.dumps(summaries_compact)}\n"
        f"density={json.dumps(density_map)}\n"
        f"recent={json.dumps(summarize_history(recent_history, limit=4))}\n"
        f"candidates={json.dumps(candidates_compact)}\n"
        + (f"failure={previous_failure}\n" if previous_failure else "")
    )


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    snippet = match.group(0)
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        try:
            parsed = ast.literal_eval(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def is_connection_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    network_markers = (
        "connection",
        "connect",
        "timeout",
        "timed out",
        "network",
        "dns",
        "name resolution",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "remoteprotocolerror",
    )
    return any(marker in name or marker in message for marker in network_markers)


def model_suggest_action(
    client: OpenAI,
    step: int,
    grid: List[List[int]],
    placed_blocks:List[Dict[str,Any]],
    remaining_blocks: List[Dict[str, Any]],
    phase: str,
    instruction: str,
    block_summaries: List[Dict[str, Any]],
    placement_focus: Optional[Dict[str, Any]],
    density_map: List[List[float]],
    total_reward: float,
    recent_history: List[Dict[str, Any]],
    candidate_actions: List[Dict[str, Any]],
    previous_failure: str = ""
) -> Tuple[Optional[Dict[str,Any]], str, Optional[str], bool]:
    if client is None:
        return None, "", "client_not_configured", True
    try:
        user_prompt = build_prompt(
            step,
            placed_blocks,
            remaining_blocks,
            phase,
            instruction,
            block_summaries,
            placement_focus,
            density_map,
            total_reward,
            recent_history,
            candidate_actions,
            previous_failure,
        )
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
            {
                "role": "system",
                "content": "Return only one JSON object with block_id, x, y, and optional action_type (place, move, or commit).",
            },
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = completion.choices[0].message.content or ""
        return extract_json_object(content), content, None, False
    except Exception as exc:
        if is_connection_error(exc):
            return None, "", "connection_error", True
        return None, "", str(exc.__class__.__name__), True


def normalize_action(
    env: ChipFlooringEnvironment,
    action_data: Optional[Dict[str, Any]],
) -> Tuple[Optional[ChipFlooringAction], Optional[Dict[str, Any]],Optional[str]]:
    
    if not action_data:
        return None,None,"Invalid_or_empty_model_output"
    
    block_id = action_data.get("block_id")
    x=action_data.get("x")
    y=action_data.get("y")
    action_type = str(action_data.get("action_type", "place") or "place").strip().lower()

    if not isinstance(block_id,str) or not isinstance(x,int) or not isinstance(y,int):
        return None,None,"invalid_action_fields"
    
    index=next((i for i, block in enumerate(env.state.blocks) if block.id==block_id),-1)

    return(
        ChipFlooringAction(x=x,y=y,choosen_block_index=index, action_type=action_type),
        {"block_id":block_id,"x":x,"y":y,"action_type":action_type},
        None,
    )


def action_to_string(action_repr: Optional[Dict[str, Any]]) -> str:
    if not action_repr:
        return "null"
    return json.dumps(action_repr, separators=(",", ":"))


def compute_score(env: ChipFlooringEnvironment, rewards: List[float]) -> float:
    block_count = max(1, len(env.state.blocks))
    completion = len(env.state.placed_blocks) / block_count
    hpwl_quality = 1.0 - min(1.0, float(env.state.current_hpwl) / max(1.0, block_count * 4.0))
    reward_signal = sum(rewards) / max(1.0, float(len(rewards)))
    score = (0.5 * completion) + (0.3 * hpwl_quality) + (0.2 * max(0.0, min(1.0, reward_signal)))
    return round(max(0.01, min(0.99, score)), 2)


def _is_local_api_base(url: str) -> bool:
    return url.startswith(
        ("http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost")
    )


def run_task(task_name: str, client: Optional[OpenAI]) -> float:
    if client is None:
        raise RuntimeError("Model-only mode requires a configured API client")

    previous_task_name = os.environ.get("TASK_NAME")
    os.environ["TASK_NAME"] = task_name

    env = ChipFlooringEnvironment()
    rewards: List[float] = []
    total_reward = 0.0
    steps_taken = 0
    success = False
    consecutive_invalids = 0
    previous_failure = ""
    recent_history: List[Dict[str, Any]] = []

    log_start(
        task=task_name,
        env=f"chip_flooring_{task_name}",
        model=f"{MODEL_NAME} ({'local' if _is_local_api_base(API_BASE_URL) else 'huggingface'})",
    )

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if not env.state.remaining_blocks and not is_long_horizon_task_name(task_name):
                success = True
                break

            grid = getattr(obs, "canva_space", env.canvas.grid if env.canvas else [])
            placed_blocks = getattr(obs, "placed_blocks", [])
            remaining_blocks = getattr(obs, "remaining_blocks", [])
            phase = getattr(obs, "phase", "placement")
            instruction = getattr(obs, "instruction", "")
            block_summaries = getattr(obs, "block_summaries", [])
            placement_focus = getattr(obs, "placement_focus", None)
            density_map = getattr(obs, "density_map", [])
            candidate_actions = getattr(obs, "candidate_positions", [])
            if not candidate_actions:
                candidate_actions = generate_candidate_actions(
                    env,
                    remaining_blocks,
                    placed_blocks=placed_blocks,
                    phase=phase,
                    per_block_limit=1,
                )

            action: Optional[ChipFlooringAction] = None
            action_repr: Optional[Dict[str, Any]] = None
            parse_error: Optional[str] = None
            attempt = 0
            model_retry_penalty = 0.0

            while action is None:
                attempt += 1
                suggested, _, request_error, had_transport_error = model_suggest_action(
                    client,
                    step,
                    grid,
                    placed_blocks,
                    remaining_blocks,
                    phase,
                    instruction,
                    block_summaries,
                    placement_focus,
                    density_map,
                    total_reward,
                    recent_history,
                    candidate_actions,
                    previous_failure=previous_failure,
                )

                if had_transport_error:
                    sleep(max(0.0, MODEL_CONNECTION_RETRY_SLEEP_MS / 1000.0))
                    continue

                action, action_repr, parse_error = normalize_action(env, suggested)

                if action is None:
                    previous_failure = parse_error or "invalid model output"
                    model_retry_penalty += float(env.get_model_output_penalty(parse_error))
                    if attempt == 1 or attempt % 3 == 0:
                        print(
                            f"Retrying model response for step={step}, attempt={attempt}, "
                            f"reason={previous_failure}, penalty={model_retry_penalty:.2f}",
                            flush=True,
                        )

            if action is None or action_repr is None:
                raise RuntimeError(f"Unable to obtain valid model action at step {step}")

            result = env.step(action)
            obs = result
            reward = float(getattr(result, "reward", 0.0) or 0.0) + model_retry_penalty
            done = bool(getattr(result, "done", False))
            invalid_reason = getattr(result, "invalid_reasons", None)

            rewards.append(reward)
            total_reward += reward
            steps_taken = step
            recent_history.append(
                {
                    "step": step,
                    "action": action_repr,
                    "reward": reward,
                    "done": done,
                    "invalid_reason": invalid_reason,
                    "source": "model" if parse_error is None else parse_error,
                    "model_retry_penalty": round(model_retry_penalty, 4),
                }
            )
            recent_history = recent_history[-24:]

            log_step(
                step=step,
                action=action_to_string(action_repr),
                reward=reward,
                done=done,
                error=invalid_reason,
            )

            if invalid_reason:
                consecutive_invalids += 1
                previous_failure = invalid_reason
                continue

            consecutive_invalids = 0
            previous_failure = ""

            if done and not env.state.remaining_blocks:
                success = True
                break
            if is_long_horizon_task_name(task_name) and done:
                success = not env.state.remaining_blocks
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        score = compute_score(env, rewards)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        if previous_task_name is None:
            os.environ.pop("TASK_NAME", None)
        else:
            os.environ["TASK_NAME"] = previous_task_name

    return score


def main() -> None:
    resolved_api_key = API_KEY or ("lm-studio" if _is_local_api_base(API_BASE_URL) else None)
    client = OpenAI(base_url=API_BASE_URL, api_key=resolved_api_key) if resolved_api_key else None

    task_scores: Dict[str, float] = {}
    for task_name in TASKS_TO_RUN:
        task_scores[task_name] = run_task(task_name, client)

    print(
        "[SUMMARY] "
        + ", ".join(f"{task}={score:.2f}" for task, score in task_scores.items()),
        flush=True,
    )


if __name__ == "__main__":
    main()
