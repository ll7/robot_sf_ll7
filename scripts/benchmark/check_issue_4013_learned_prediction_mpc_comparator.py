#!/usr/bin/env python3
"""Check issue #4013 learned-prediction MPC comparator config contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.learned_prediction_mpc_comparator import build_comparator_preflight

DEFAULT_MODEL_FREE_CONFIG = Path("configs/benchmarks/issue_4013_model_free_smoke.yaml")
DEFAULT_MODEL_BASED_CONFIG = Path("configs/benchmarks/issue_4013_model_based_smoke.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-free-config", type=Path, default=DEFAULT_MODEL_FREE_CONFIG)
    parser.add_argument("--model-based-config", type=Path, default=DEFAULT_MODEL_BASED_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run comparator preflight and emit JSON."""

    args = parse_args(argv)
    report = build_comparator_preflight(
        model_free_config=args.model_free_config,
        model_based_config=args.model_based_config,
        repo_root=args.repo_root,
    ).to_dict()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "ready_diagnostic_smoke" else 1


if __name__ == "__main__":
    raise SystemExit(main())
