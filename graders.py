"""
Root-level grader re-exports for repo-root validation.

This file lets validators import grader functions directly from the
repository checkout, without depending on setuptools package installation.
"""

from __future__ import annotations

import os
import sys

_SERVER_PATH = os.path.join(os.path.dirname(__file__), "server")
if _SERVER_PATH not in sys.path:
    sys.path.insert(0, _SERVER_PATH)

try:
    from chip_flooring_env.server.graders import (
        easy_grader,
        hard_grader,
        medium_grader,
    )
except ImportError:
    from server.graders import easy_grader, hard_grader, medium_grader

__all__ = ["easy_grader", "medium_grader", "hard_grader"]
