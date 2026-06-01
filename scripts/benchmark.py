"""Fail-closed guard for the retired root-level profiling benchmark."""

from __future__ import annotations

import sys

_MIGRATION_MESSAGE = """\
scripts/benchmark.py is retired.

Use one of the maintained benchmark or smoke entrypoints instead:
  uv run python scripts/benchmark_workers.py --out output/benchmarks/bench_workers
  DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python scripts/validation/performance_smoke_test.py

For benchmark campaigns, prefer config-driven tools under scripts/tools/ or
scripts/validation/ and record outputs under output/.
"""


def main(_argv: object = None) -> int:
    """Exit non-zero with supported benchmark alternatives."""
    sys.stderr.write(_MIGRATION_MESSAGE)
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
