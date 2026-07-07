#!/usr/bin/env python3
"""Thin CLI wrapper for the forecast-lane capability inventory / preflight.

Run this before forecast-lane work to confirm the lane's components are present
and importable on the current checkout::

    uv run python scripts/benchmark/forecast_lane_preflight.py
    uv run python scripts/benchmark/forecast_lane_preflight.py --json
    uv run python scripts/benchmark/forecast_lane_preflight.py --status --json
    uv run python scripts/benchmark/forecast_lane_preflight.py --closure-audit --json

Exit code 0 means every required forecast capability is present; non-zero means a
required capability is missing or broken (the report names the blocker and its
owner). The check is read-only: it imports and inspects the canonical
``robot_sf/benchmark`` forecast owners. Status mode reports the issue #2835
checked-progress ledger and learned-predictor blockers. It never runs
predictors, training, or benchmark campaigns.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a bare script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.forecast_lane_inventory import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
