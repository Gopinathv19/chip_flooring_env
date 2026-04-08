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
    or "hard"
).strip().lower() or "hard"
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


def summarize_history(history: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    return history[-limit:]


def compact_block_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": summary.get("id"),
        "priority": round(float(summary.get("priority_score", 0.0)), 3),
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

    return candidates


def build_prompt(
    step: int,
    placed_blocks: List[Dict[str, Any]],
    remaining_blocks: List[Dict[str, Any]],
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
        "Choose one placement from the candidate actions.\n"
        "Return JSON only: {\"block_id\":\"...\",\"x\":0,\"y\":0}.\n"
        "Prefer the action that best reduces wirelength while staying legal.\n"
        "Use the focus block, placed neighbor positions, and candidate scores.\n\n"
        f"step={step}\n"
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


def model_suggest_action(
    client: OpenAI,
    step: int,
    grid: List[List[int]],
    placed_blocks:List[Dict[str,Any]],
    remaining_blocks: List[Dict[str, Any]],
    block_summaries: List[Dict[str, Any]],
    placement_focus: Optional[Dict[str, Any]],
    density_map: List[List[float]],
    total_reward: float,
    recent_history: List[Dict[str, Any]],
    candidate_actions: List[Dict[str, Any]],
    previous_failure: str = ""
) -> Tuple[Optional[Dict[str,Any]],str]:
    try:
        user_prompt = build_prompt(
            step,
            placed_blocks,
            remaining_blocks,
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
                    "content": "Return only one JSON object with block_id, x, and y.",
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

    if not isinstance(block_id,str) or not isinstance(x,int) or not isinstance(y,int):
        return None,None,"invalid_action_fields"
    
    index=next((i for i, block in enumerate(env.state.blocks) if block.id==block_id),-1)

    return(
        ChipFlooringAction(x=x,y=y,choosen_block_index=index),
        {"block_id":block_id,"x":x,"y":y},
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


def compute_score(env: ChipFlooringEnvironment, rewards: List[float]) -> float:
    block_count = max(1, len(env.state.blocks))
    completion = len(env.state.placed_blocks) / block_count
    hpwl_quality = 1.0 - min(1.0, float(env.state.current_hpwl) / max(1.0, block_count * 4.0))
    reward_signal = sum(rewards) / max(1.0, float(len(rewards)))
    score = (0.5 * completion) + (0.3 * hpwl_quality) + (0.2 * max(0.0, min(1.0, reward_signal)))
    return max(0.0, min(1.0, score))


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
        ),
        candidate,
    )


def _is_local_api_base(url: str) -> bool:
    return url.startswith(
        ("http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost")
    )




def main() -> None:
    resolved_api_key = API_KEY or ("lm-studio" if _is_local_api_base(API_BASE_URL) else None)
    client = OpenAI(base_url=API_BASE_URL, api_key=resolved_api_key) if resolved_api_key else None
    env = ChipFlooringEnvironment()
    rewards: List[float] = []
    total_reward = 0.0
    steps_taken = 0
    success = False
    consecutive_invalids=0
    previous_failure=""
    recent_history: List[Dict[str, Any]] = []

    log_start(
        task=TASK_NAME,
        env=BENCHMARK,
        model=f"{MODEL_NAME} ({'local' if _is_local_api_base(API_BASE_URL) else 'huggingface'})",
    )

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if not env.state.remaining_blocks:
                success = True
                break

            grid = getattr(obs, "canva_space", env.canvas.grid if env.canvas else [])
            placed_blocks = getattr(obs, "placed_blocks", [])
            remaining_blocks = getattr(obs, "remaining_blocks", [])
            block_summaries = getattr(obs, "block_summaries", [])
            placement_focus = getattr(obs, "placement_focus", None)
            density_map = getattr(obs, "density_map", [])
            candidate_actions = getattr(obs, "candidate_positions", [])
            if not candidate_actions:
                candidate_actions = generate_candidate_actions(env, remaining_blocks, per_block_limit=1)

            suggested,raw_content = model_suggest_action(
                client,
                step,
                grid,
                placed_blocks,
                remaining_blocks,
                block_summaries,
                placement_focus,
                density_map,
                total_reward,
                recent_history,
                candidate_actions,
                previous_failure=previous_failure,
            )

            action,action_repr,parse_error = normalize_action(env,suggested)
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
                    steps_taken=step
                    consecutive_invalids+=1
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

                    log_step(step=step,action="null",reward=reward,done=False,error=previous_failure)
                    continue

                action = fallback_action
                action_repr = fallback_repr
                parse_error = "model_invalid_fallback_used"

            result = env.step(action)
            obs = result
            reward = float(getattr(result,"reward",0.0) or 0.0)
            done = bool(getattr(result,"done",False))
            invalid_reason = getattr(result,"invalid_reason",None)

            rewards.append(reward)
            total_reward += reward
            steps_taken=step
            recent_history.append(
                {
                    "step": step,
                    "action": action_repr,
                    "reward": reward,
                    "done": done,
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

    finally:
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        score = compute_score(env, rewards)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        


 


if __name__ == "__main__":
    main()
