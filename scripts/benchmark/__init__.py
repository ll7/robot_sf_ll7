"""Importable benchmark reporting and campaign helper scripts.

The retired root-level profiling benchmark remains available as
``python -m scripts.benchmark`` and exits with migration guidance.
"""

from __future__ import annotations

import sys

_MIGRATION_MESSAGE = """\
The root-level profiling benchmark is retired.

Use one of the maintained benchmark or smoke entrypoints instead:
  uv run python scripts/benchmark_workers.py --out output/benchmarks/bench_workers
  DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python scripts/validation/performance_smoke_test.py

For benchmark campaigns, prefer config-driven tools under scripts/tools/ or
scripts/validation/ and record outputs under output/.
"""


def main(_argv: object = None) -> int:
    """Exit non-zero with alternatives to the retired profiling benchmark."""
    sys.stderr.write(_MIGRATION_MESSAGE)
    return 2


__all__ = ["main"]
