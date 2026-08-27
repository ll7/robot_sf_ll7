#!/usr/bin/env python3
"""Thin CLI wrapper for the credential-free published-release audit."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.published_release_audit import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
