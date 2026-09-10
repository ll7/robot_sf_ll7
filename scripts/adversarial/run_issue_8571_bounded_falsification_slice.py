#!/usr/bin/env python3
"""Build the default-disabled issue #8571 falsification preflight ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.adversarial.bounded_falsification import (
    write_bounded_falsification_preflight,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = (
    REPO_ROOT / "configs/adversarial/issue_8570_bounded_falsification_answerability_v1.yaml"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/adversarial/issue_8571/preflight.json"


def _parser() -> argparse.ArgumentParser:
    """Build the config-first command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the issue #8571 bounded falsification ledger without running a "
            "simulator, planner, optimizer, campaign, or replay."
        )
    )
    parser.add_argument(
        "--packet",
        type=Path,
        default=DEFAULT_PACKET,
        help="validated issue #8570 answerability packet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path (defaults to the disposable output tree)",
    )
    return parser


def main() -> int:
    """Build, persist, and summarize the deterministic preflight ledger."""
    args = _parser().parse_args()
    report = write_bounded_falsification_preflight(
        args.packet,
        args.output,
        repo_root=REPO_ROOT,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema_version": report["schema_version"],
                "gate_status": report["gate"]["status"],
                "outcome_summary": report["outcome_summary"],
                "simulator_executed": report["execution"]["simulator_executed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
