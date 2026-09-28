#!/usr/bin/env python3
"""Run the fail-closed release matrix preflight (issue #9731).

The command requires the selected release manifest. It does not fall back to the
development scenario matrix or accept a seed override::

    uv run python scripts/benchmark/preflight_spawn_clearance.py \
        --manifest configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml \
        --workers 8 \
        --json-output output/preflight/spawn_matrix.json \
        --markdown-output output/preflight/spawn_matrix.md
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
