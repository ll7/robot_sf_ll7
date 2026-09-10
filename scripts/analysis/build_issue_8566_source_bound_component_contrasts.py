#!/usr/bin/env python3
"""Build the issue #8566 source-bound fixture report without running a campaign."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.source_bound_component_contrasts import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
