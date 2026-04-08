"""
Root-level grader re-exports for repo-root validation.

This file lets validators import grader functions directly from the
repository checkout, without depending on setuptools package installation.
"""

from __future__ import annotations

from importlib import util as importlib_util
from pathlib import Path


_SERVER_GRADERS_PATH = Path(__file__).resolve().parent / "server" / "graders.py"
_SPEC = importlib_util.spec_from_file_location(
    "_chip_flooring_env_server_graders", _SERVER_GRADERS_PATH
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - defensive guard
    raise ImportError(f"Cannot load graders from {_SERVER_GRADERS_PATH}")

_MODULE = importlib_util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

GRADERS = _MODULE.GRADERS
easy_grader = _MODULE.easy_grader
medium_grader = _MODULE.medium_grader
hard_grader = _MODULE.hard_grader
grade_easy = _MODULE.grade_easy
grade_medium = _MODULE.grade_medium
grade_hard = _MODULE.grade_hard

__all__ = [
    "GRADERS",
    "easy_grader",
    "medium_grader",
    "hard_grader",
    "grade_easy",
    "grade_medium",
    "grade_hard",
]
