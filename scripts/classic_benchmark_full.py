"""CLI entry script for the Full Classic Interaction Benchmark.

Implements argument parsing only at this stage (Task T002). The invoked
`run_full_benchmark` function is a placeholder until later tasks implement it.
"""

from __future__ import annotations

import argparse
import sys

try:
    from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark  # type: ignore
except Exception:  # pragma: no cover - during early scaffolding

    def run_full_benchmark(cfg):  # fallback placeholder
        raise NotImplementedError("Full benchmark not yet implemented. Follow tasks T022+.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full Classic Interaction Benchmark (scaffold)")
    parser.add_argument(
        "--scenarios", required=True, help="Path to classic_interactions scenario matrix YAML"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory root for benchmark artifacts"
    )
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (placeholder)")
    parser.add_argument(
        "--smoke", action="store_true", help="Enable smoke mode (placeholder behavior)"
    )
    parser.add_argument("--seed", type=int, default=123, help="Master seed (placeholder)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # For now just echo arguments; real execution added in later tasks.
    print("[classic_benchmark_full] Arguments:", vars(args))
    try:
        run_full_benchmark(args)  # placeholder call with raw args for now
    except NotImplementedError as e:  # expected until implementation tasks complete
        print(str(e))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
