#!/usr/bin/env python3
"""Run one mechanism-aware diagnostic reproduction case from a single command."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from loguru import logger

# Keep the one-command reproduction readable by default.
logger.remove()
logger.add(sys.stderr, level="ERROR")

from scripts.validation import run_topology_hypothesis_diagnostics  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path("output/demo/mechanism_report")
CLAIM_BOUNDARY = "diagnostic_only_not_benchmark_success"


@dataclass(frozen=True)
class ReproductionCase:
    """Configuration for one supported mechanism reproduction case."""

    name: str
    description: str
    diagnostic_args: tuple[str, ...]
    output_subdir: str


CASES: dict[str, ReproductionCase] = {
    "topology-primary-route": ReproductionCase(
        name="topology-primary-route",
        description=(
            "Topology-hypothesis diagnostic on the double-bottleneck case used to inspect "
            "whether the planner exposes at least two local route hypotheses."
        ),
        diagnostic_args=(
            "--candidate",
            "hybrid_rule_v3_waypoint2_route_lookahead8",
            "--stage",
            "full_matrix",
            "--scenario-name",
            "classic_realworld_double_bottleneck_high",
            "--seed",
            "111",
            "--horizon",
            "160",
            "--min-hypotheses",
            "2",
        ),
        output_subdir="topology_primary_route",
    )
}


def build_parser() -> argparse.ArgumentParser:
    """Build the quickstart parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=sorted(CASES),
        default="topology-primary-route",
        help="Mechanism-aware reproduction case to run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Disposable local output root for generated diagnostic artifacts.",
    )
    return parser


def diagnostic_args_for_case(case: ReproductionCase, output_root: Path) -> list[str]:
    """Return the underlying topology diagnostic argv for a reproduction case."""
    output_dir = output_root / case.output_subdir
    return [*case.diagnostic_args, "--output-dir", str(output_dir)]


def run_case(*, case_name: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> int:
    """Run a supported reproduction case and preserve diagnostic exit semantics."""
    case = CASES[case_name]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    diagnostic_argv = diagnostic_args_for_case(case, output_root)

    print(f"Mechanism-aware reproduction case: {case.name}")
    print(case.description)
    print(f"Output root: {output_root}")
    print(f"Claim boundary: {CLAIM_BOUNDARY}")
    print("Running topology hypothesis diagnostic...")
    exit_code = run_topology_hypothesis_diagnostics.main(diagnostic_argv)
    if exit_code == 0:
        print("Diagnostic reproduction completed.")
    else:
        print(
            "Diagnostic reproduction did not produce sufficient topology hypotheses; "
            "preserving fail-closed exit code."
        )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """Run the mechanism reproduction CLI."""
    args = build_parser().parse_args(argv)
    return run_case(case_name=args.case, output_root=args.output_root)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
