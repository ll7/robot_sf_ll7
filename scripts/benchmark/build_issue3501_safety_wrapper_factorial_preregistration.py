#!/usr/bin/env python3
"""Build issue #3501 safety-wrapper factorial pre-registration planned rows."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from robot_sf.benchmark.safety_wrapper_factorial_preregistration import (
    build_preregistration_plan,
    load_factorial_preregistration_config,
    write_preregistration_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    "configs/benchmarks/issue_3501_safety_wrapper_factorial_preregistration_cpu_smoke.yaml"
)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Tracked pre-registration YAML.")
    parser.add_argument(
        "--out",
        default="output/issue_3501_safety_wrapper_factorial_preregistration",
        help="Output directory for deterministic planned-row JSON.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the planned-row builder.

    Returns:
        Process exit code.
    """

    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_factorial_preregistration_config(config_path)
    plan = build_preregistration_plan(config)
    plan["git_head"] = _git_head()
    out_path = write_preregistration_plan(plan, REPO_ROOT / args.out)
    try:
        display_path = out_path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = out_path
    print(display_path)
    print(f"pair_check.complete={plan['pair_check']['complete']}")
    print(f"row_count={plan['row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
