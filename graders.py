"""
Repo-root grader shim for validators that import `graders` directly.

The runtime source of truth remains `server/graders.py`.
"""

from __future__ import annotations

from server.graders import (  # noqa: F401
    GRADERS,
    easy_grader,
    grade_easy,
    grade_hard,
    grade_medium,
    hard_grader,
    medium_grader,
)

__all__ = [
    "GRADERS",
    "easy_grader",
    "medium_grader",
    "hard_grader",
    "grade_easy",
    "grade_medium",
    "grade_hard",
]
