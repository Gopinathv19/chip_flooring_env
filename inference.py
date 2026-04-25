import ast
import json
import os
import re
import sys
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
    print("\n" + "═" * 60)
    print(f"🚀 INFERENCE STARTED")
    print(f"► Task  : {task}")
    print(f"► Env   : {env}")
    print(f"► Model : {model}")
    print("═" * 60 + "\n", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "None"
    reward_str = f"+{reward:.2f}" if reward > 0 else f"{reward:.2f}"
    
    print(
        f"\n❖ Step {step:02d} " + "━" * 40 + "\n"
        f"  │ Action : {action}\n"
        f"  │ Reward : {reward_str}  |  Done: {done}  |  Error: {error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print("\n\n" + "═" * 60)
    print(f"{status} in {steps} steps")
    print(f"► Final Score : {score:.2f}")
    print(f"► Rewards     : [{rewards_str}]")
    print("═" * 60 + "\n", flush=True)


def is_long_horizon_task_name(task_name: str) -> bool:
    normalized = str(task_name or "").strip().lower()
    return normalized.endswith("_long_horizon") or normalized == "long_horizon"


def summarize_history(history: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    return history[-limit:]


def compact_block_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": summary.get("id"),
        "priority": round(float(summary.get("priority_score", 0.0)), 3),
        "degree": summary.get("degree", 0),
        "placed_neighbors": len(summary.get("placed_neighbors") or [])
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
    board_ascii: str,
    placed_count: int,
    remaining_count: int,
    phase: str,
    instruction: str,
    focus_block: Optional[Dict[str, Any]],
    current_hpwl: float,
    delta_hpwl: float,
    recent_history: List[Dict[str, Any]],
    candidate_actions: List[Dict[str, Any]],
    previous_failure: str = "",
) -> str:
    focus_compact = compact_block_summary(focus_block) if focus_block else None
    
    candidates_compact = []
    for c in candidate_actions[:12]:
        candidates_compact.append({
            "block_id": c.get("block_id"),
            "x": c.get("x"),
            "y": c.get("y"),
            "score": round(c.get("score", 0.0), 3),
            "congestion": round(c.get("congestion_score", 0.0), 3),
            "action_type": c.get("action_type", "place")
        })

    return (
        "Choose one action from the candidate actions.\n"
        "Return JSON only with one of these forms: {\"block_id\":\"...\",\"x\":0,\"y\":0,\"action_type\":\"place\"}, {\"block_id\":\"...\",\"x\":0,\"y\":0,\"action_type\":\"move\"}, or {\"block_id\":\"...\",\"x\":0,\"y\":0,\"action_type\":\"commit\"}.\n"
        "Use action_type=\"move\" only when the episode is in repair or finalize and the candidate is a relocation.\n"
        "Use action_type=\"commit\" in repair or finalize when the current position should be locked and no further relocation is needed.\n"
        "Prefer the action that best reduces wirelength while staying legal. Higher candidate 'score' is better. Lower 'congestion' is better.\n\n"
        f"step={step} phase={phase}\n"
        f"instruction={instruction}\n"
        f"placed_count={placed_count} remaining_count={remaining_count} current_hpwl={current_hpwl:.2f} delta_hpwl={delta_hpwl:.2f}\n"
        f"focus={json.dumps(focus_compact)}\n"
        "board_ascii:\n"
        f"{board_ascii}\n"
        f"recent_history={json.dumps(summarize_history(recent_history, limit=3))}\n"
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


def model_suggest_action(
    client: OpenAI,
    step: int,
    board_ascii: str,
    placed_count: int,
    remaining_count: int,
    phase: str,
    instruction: str,
    focus_block: Optional[Dict[str, Any]],
    current_hpwl: float,
    delta_hpwl: float,
    recent_history: List[Dict[str, Any]],
    candidate_actions: List[Dict[str, Any]],
    previous_failure: str = ""
) -> Tuple[Optional[Dict[str,Any]],str]:
    try:
        user_prompt = build_prompt(
            step,
            board_ascii,
            placed_count,
            remaining_count,
            phase,
            instruction,
            focus_block,
            current_hpwl,
            delta_hpwl,
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
        return extract_json_object(content),content
    except Exception as exc:
        return None,""


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


def action_is_in_candidates(
    action_repr: Optional[Dict[str, Any]],
    candidate_actions: List[Dict[str, Any]],
) -> bool:
    if not action_repr:
        return False
    return any(
        candidate["block_id"] == action_repr.get("block_id")
        and int(candidate["x"]) == int(action_repr.get("x", -1))
        and int(candidate["y"]) == int(action_repr.get("y", -1))
        for candidate in candidate_actions
    )
    

 


def action_to_string(action_repr: Optional[Dict[str, Any]]) -> str:
    if not action_repr:
        return "null"
    return json.dumps(action_repr, separators=(",", ":"))


def compute_score(env: ChipFlooringEnvironment, rewards: List[float], recent_history: List[Dict[str, Any]]) -> float:
    trajectory = recent_history
    done = env.state.done
    total_blocks = max(1, len(env.state.blocks))
    placed_count = len(env.state.placed_blocks)
    completion = placed_count / total_blocks
    current_hpwl = float(env.state.current_hpwl)
    hpwl_quality = 1.0 - min(1.0, current_hpwl / max(1.0, total_blocks * 4.0))
    invalid_steps = sum(1 for step in trajectory if step.get("invalid_reason") is not None)
    total_steps = len(trajectory) or 1
    valid_rate = 1.0 - min(1.0, invalid_steps / total_steps)
    
    phase_names = {str(step.get("phase", "")) for step in trajectory if step.get("phase")}
    repair_moves = sum(
        1 for step in trajectory
        if isinstance(step.get("action"), dict) and str(step["action"].get("action_type", "place")).lower() == "move"
    )
    phase_coverage = min(1.0, len(phase_names) / 3.0)
    repair_usage = min(1.0, repair_moves / max(1, total_blocks // 3))
    
    raw = (0.35 * completion) + (0.25 * hpwl_quality) + (0.15 * valid_rate) + (0.15 * phase_coverage) + (0.10 * repair_usage)
    bonus = 0.12 if done else 0.0
    score = raw + bonus
    return round(max(0.01, min(0.99, score)), 2)


def choose_fallback_action(
    env: ChipFlooringEnvironment,
    candidate_actions: List[Dict[str, Any]],
) -> Tuple[Optional[ChipFlooringAction], Optional[Dict[str, Any]]]:
    if not candidate_actions:
        return None, None

    candidate = candidate_actions[0]
    index = next(
        (i for i, block in enumerate(env.state.blocks) if block.id == candidate["block_id"]),
        -1,
    )
    if index < 0:
        return None, None

    return (
        ChipFlooringAction(
            x=int(candidate["x"]),
            y=int(candidate["y"]),
            choosen_block_index=index,
            action_type=str(candidate.get("action_type", "place")),
        ),
        candidate,
    )


def _is_local_api_base(url: str) -> bool:
    return url.startswith(
        ("http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost")
    )


def run_task(task_name: str, client: Optional[OpenAI]) -> float:
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
            board_ascii = getattr(obs, "board_ascii", "")
            placed_blocks = getattr(obs, "placed_blocks", [])
            remaining_blocks = getattr(obs, "remaining_blocks", [])
            phase = getattr(obs, "phase", "placement")
            instruction = getattr(obs, "instruction", "")
            placement_focus = getattr(obs, "placement_focus", None)
            current_hpwl = getattr(obs, "current_hpwl", 0.0)
            delta_hpwl = getattr(obs, "delta_hpwl", 0.0)
            candidate_actions = getattr(obs, "candidate_positions", [])
            if not candidate_actions:
                candidate_actions = generate_candidate_actions(
                    env,
                    remaining_blocks,
                    placed_blocks=placed_blocks,
                    phase=phase,
                    per_block_limit=1,
                )

            suggested, raw_content = model_suggest_action(
                client,
                step,
                board_ascii,
                len(placed_blocks),
                len(remaining_blocks),
                phase,
                instruction,
                placement_focus,
                current_hpwl,
                delta_hpwl,
                recent_history,
                candidate_actions,
                previous_failure=previous_failure,
            )

            action, action_repr, parse_error = normalize_action(env, suggested)
            if action is not None and not action_is_in_candidates(action_repr, candidate_actions):
                action = None
                action_repr = None
                parse_error = "model_action_not_in_candidates"

            if action is None:
                fallback_action, fallback_repr = choose_fallback_action(env, candidate_actions)
                if fallback_action is None:
                    reward = -1.0
                    rewards.append(reward)
                    total_reward += reward
                    steps_taken = step
                    consecutive_invalids += 1
                    previous_failure = parse_error or "invalid model output"
                    recent_history.append(
                        {
                            "step": step,
                            "action": None,
                            "reward": reward,
                            "done": False,
                            "invalid_reason": previous_failure,
                            "source": "model_failed_no_fallback",
                        }
                    )
                    log_step(step=step, action="null", reward=reward, done=False, error=previous_failure)
                    continue

                action = fallback_action
                action_repr = fallback_repr
                parse_error = "model_invalid_fallback_used"

            result = env.step(action)
            obs = result
            reward = float(getattr(result, "reward", 0.0) or 0.0)
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
                    "phase": phase,
                    "delta_hpwl": float(getattr(result, "delta_hpwl", 0.0) or 0.0),
                    "invalid_reason": invalid_reason,
                    "source": "model" if parse_error is None else parse_error,
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
        score = compute_score(env, rewards, recent_history)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        if previous_task_name is None:
            os.environ.pop("TASK_NAME", None)
        else:
            os.environ["TASK_NAME"] = previous_task_name

    return score


def main() -> None:
    resolved_api_key = API_KEY or ("lm-studio" if _is_local_api_base(API_BASE_URL) else None)
    client = OpenAI(base_url=API_BASE_URL, api_key=resolved_api_key) if resolved_api_key else None

    is_local = _is_local_api_base(API_BASE_URL)
    print("=" * 60, flush=True)
    if is_local:
        print(f"[INFO] Using LOCAL model: {MODEL_NAME}", flush=True)
        print(f"[INFO] Local API Base URL: {API_BASE_URL}", flush=True)
    else:
        print(f"[INFO] Using CLOUD HUGGINGFACE model: {MODEL_NAME}", flush=True)
        print(f"[INFO] Cloud API Base URL: {API_BASE_URL}", flush=True)
    print("=" * 60, flush=True)

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
