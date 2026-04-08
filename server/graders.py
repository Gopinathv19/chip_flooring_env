from __future__ import annotations

from typing import Any, Dict


def _extract_score(*args: Any, **kwargs: Any) -> float:
    value: Any = None

    if "score" in kwargs:
        value = kwargs["score"]
    elif "reward" in kwargs:
        value = kwargs["reward"]
    elif args:
        first = args[0]
        if isinstance(first, dict):
            value = first.get("score", first.get("reward", 0.0))
        else:
            value = first

    try:
        score = float(value if value is not None else 0.0)
    except (TypeError, ValueError):
        score = 0.0

    return max(0.0, min(1.0, score))


def _grade(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    score = _extract_score(*args, **kwargs)
    return {
        "score": score,
        "grader_type": "deterministic",
    }


def easy_grader(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _grade(*args, **kwargs)


def medium_grader(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _grade(*args, **kwargs)


def hard_grader(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _grade(*args, **kwargs)


GRADERS = {
    "easy": easy_grader,
    "medium": medium_grader,
    "hard": hard_grader,
}
