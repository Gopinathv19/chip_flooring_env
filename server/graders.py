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
    if isinstance(placed_raw, list):
        placed_count = len(placed_raw)
    else:
        placed_count = _as_int(placed_raw, 0)

    if total_blocks <= 0 and trajectory:
        last_step = trajectory[-1]
        remaining_raw = last_step.get("remaining_blocks", [])
        placed_raw = last_step.get("placed_blocks", [])
        if isinstance(placed_raw, list):
            placed_count = len(placed_raw)
        if isinstance(remaining_raw, list):
            total_blocks = placed_count + len(remaining_raw)

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


GRADERS = {
    "easy": easy_grader,
    "medium": medium_grader,
    "hard": hard_grader,
}
