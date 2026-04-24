from __future__ import annotations

from typing import Any, Dict, List


def _parse_payload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    if kwargs:
        return dict(kwargs)
    if args and isinstance(args[0], dict):
        return dict(args[0])
    return {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_score(score: float) -> float:
    # Validator requires scores strictly inside (0, 1), not endpoints.
    return round(max(0.01, min(0.99, score)), 2)


def _compute_score(payload: Dict[str, Any], difficulty: str) -> float:
    trajectory: List[Dict[str, Any]] = payload.get("trajectory") or []
    done = bool(payload.get("done", False))

    total_blocks = _as_int(payload.get("total_blocks") or payload.get("block_count"), 0)
    placed_raw = payload.get("placed_blocks", 0)
    remaining_raw = payload.get("remaining_blocks", [])
    blocks_raw = payload.get("blocks", [])
    placed_list = placed_raw if isinstance(placed_raw, list) else []
    remaining_list = remaining_raw if isinstance(remaining_raw, list) else []
    blocks_list = blocks_raw if isinstance(blocks_raw, list) else []

    def _count_fixed(items: Any) -> int:
        if not isinstance(items, list):
            return 0
        return sum(1 for item in items if isinstance(item, dict) and bool(item.get("fixed")))

    def _count_unfixed(items: Any) -> int:
        if not isinstance(items, list):
            return 0
        return sum(1 for item in items if not (isinstance(item, dict) and bool(item.get("fixed"))))

    if placed_list:
        placed_count = len(placed_list)
    else:
        placed_count = _as_int(placed_raw, 0)

    fixed_count = 0
    movable_total = total_blocks
    if blocks_list:
        fixed_count = _count_fixed(blocks_list)
        movable_total = len(blocks_list) - fixed_count
    elif placed_list or remaining_list:
        fixed_count = _count_fixed(placed_list) + _count_fixed(remaining_list)
        movable_total = len(placed_list) + len(remaining_list) - fixed_count

    if total_blocks > 0 and fixed_count > 0:
        movable_total = max(0, total_blocks - fixed_count)

    if total_blocks <= 0 and trajectory:
        last_step = trajectory[-1]
        remaining_raw = last_step.get("remaining_blocks", [])
        placed_raw = last_step.get("placed_blocks", [])
        placed_list = placed_raw if isinstance(placed_raw, list) else []
        remaining_list = remaining_raw if isinstance(remaining_raw, list) else []
        if placed_list:
            placed_count = len(placed_list)
        if remaining_list:
            total_blocks = placed_count + len(remaining_list)
            fixed_count = _count_fixed(placed_list) + _count_fixed(remaining_list)
            movable_total = max(0, total_blocks - fixed_count)

    if movable_total > 0:
        total_blocks = movable_total
        if placed_list:
            placed_count = _count_unfixed(placed_list)
        elif fixed_count > 0:
            placed_count = max(0, placed_count - fixed_count)

    completion = placed_count / max(1, total_blocks) if total_blocks > 0 else (1.0 if done else 0.0)

    current_hpwl = _as_float(payload.get("current_hpwl"), 0.0)
    hpwl_budget = max(1.0, float(total_blocks) * 4.0)
    hpwl_quality = 1.0 - min(1.0, current_hpwl / hpwl_budget)

    invalid_steps = sum(1 for step in trajectory if step.get("invalid_reason") is not None)
    total_steps = len(trajectory) or 1
    valid_rate = 1.0 - min(1.0, invalid_steps / total_steps)

    if difficulty == "easy":
        raw = (0.60 * completion) + (0.20 * hpwl_quality) + (0.20 * valid_rate)
        bonus = 0.10 if done else 0.0
    elif difficulty == "medium":
        raw = (0.50 * completion) + (0.30 * hpwl_quality) + (0.20 * valid_rate)
        bonus = 0.08 if done else 0.0
    elif difficulty == "long_horizon":
        phase_names = {str(step.get("phase", "")) for step in trajectory if step.get("phase")}
        repair_moves = sum(
            1
            for step in trajectory
            if isinstance(step.get("action"), dict) and str(step["action"].get("action_type", "place")).lower() == "move"
        )
        phase_coverage = min(1.0, len(phase_names) / 3.0)
        repair_usage = min(1.0, repair_moves / max(1, total_blocks // 3))
        raw = (
            (0.35 * completion)
            + (0.25 * hpwl_quality)
            + (0.15 * valid_rate)
            + (0.15 * phase_coverage)
            + (0.10 * repair_usage)
        )
        bonus = 0.12 if done else 0.0
    else:
        raw = (0.45 * completion) + (0.35 * hpwl_quality) + (0.20 * valid_rate)
        bonus = 0.05 if done else 0.0

    if not trajectory and not done and placed_count == 0:
        fallback = _as_float(payload.get("score", payload.get("reward", 0.0)), 0.0)
        return _normalize_score(fallback)

    return _normalize_score(raw + bonus)


def easy_grader(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _parse_payload(*args, **kwargs)
    score = _compute_score(payload, "easy")
    return {
        "score": score,
        "grader_type": "deterministic",
        "difficulty": "easy",
    }


def grade_easy(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return easy_grader(*args, **kwargs)


def medium_grader(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _parse_payload(*args, **kwargs)
    score = _compute_score(payload, "medium")
    return {
        "score": score,
        "grader_type": "deterministic",
        "difficulty": "medium",
    }


def grade_medium(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return medium_grader(*args, **kwargs)


def hard_grader(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _parse_payload(*args, **kwargs)
    score = _compute_score(payload, "hard")
    return {
        "score": score,
        "grader_type": "deterministic",
        "difficulty": "hard",
    }


def grade_hard(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return hard_grader(*args, **kwargs)


def heterogeneous_grader(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _parse_payload(*args, **kwargs)
    score = _compute_score(payload, "hard")
    return {
        "score": score,
        "grader_type": "deterministic",
        "difficulty": "heterogeneous",
    }


def grade_heterogeneous(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return heterogeneous_grader(*args, **kwargs)


def fixed_obstacles_grader(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _parse_payload(*args, **kwargs)
    score = _compute_score(payload, "hard")
    return {
        "score": score,
        "grader_type": "deterministic",
        "difficulty": "fixed_obstacles",
    }


def grade_fixed_obstacles(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return fixed_obstacles_grader(*args, **kwargs)


def long_horizon_grader(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _parse_payload(*args, **kwargs)
    score = _compute_score(payload, "long_horizon")
    return {
        "score": score,
        "grader_type": "deterministic",
        "difficulty": "long_horizon",
    }


def grade_long_horizon(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return long_horizon_grader(*args, **kwargs)


GRADERS = {
    "easy": easy_grader,
    "medium": medium_grader,
    "hard": hard_grader,
    "heterogeneous": heterogeneous_grader,
    "fixed_obstacles": fixed_obstacles_grader,
    "long_horizon": long_horizon_grader,
}

for difficulty in ("easy", "medium", "hard"):
    for scenario in ("standard", "heterogeneous", "fixed_obstacles"):
        GRADERS[f"{difficulty}_{scenario}_long_horizon"] = long_horizon_grader
