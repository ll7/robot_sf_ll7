#!/usr/bin/env python3
"""Reset-only spawn-clearance preflight over the release scenario matrix (issue #9725).

Builds and resets every scenario x seed exactly as the map runner does and fails
(exit 1) on any reset where the robot footprint overlaps a pedestrian or the static
map. Run it before a benchmark campaign::

    uv run python scripts/benchmark/preflight_spawn_clearance.py --workers 8 \
        --output output/preflight/spawn_clearance.json
    uv run python scripts/benchmark/preflight_spawn_clearance.py \
        --scenario classic_head_on_corridor_low --seeds 116 --step-zero
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a bare script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.spawn_preflight import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
