#!/usr/bin/env python3
"""Run diagnostic-only feasibility probes for issue #3484 scenario families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.scenario_certification.feasibility_diagnostics import (
    DEFAULT_FAMILIES,
    FeasibilityDiagnosticConfig,
    run_feasibility_diagnostics,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=Path("configs/scenarios/classic_interactions.yaml"),
        help="Scenario manifest to inspect.",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Scenario family/archetype to include. May be supplied multiple times.",
    )
    parser.add_argument(
        "--seeds-per-scenario",
        type=int,
        default=1,
        help="Reserved report knob; this diagnostic slice runs the first scenario seed.",
    )
    parser.add_argument("--baseline-algo", default="goal", help="Actor-free rollout algorithm.")
    parser.add_argument(
        "--oracle-algo",
        default="goal",
        help="Scripted/oracle trajectory rollout algorithm for the diagnostic lane.",
    )
    parser.add_argument(
        "--include-extended-time",
        action="store_true",
        help="Also run the extended-time lane; off by default for this issue slice.",
    )
    parser.add_argument(
        "--extended-time-multiplier",
        type=float,
        default=2.0,
        help="Multiplier used when --include-extended-time is set.",
    )
    parser.add_argument(
        "--record-simulation-step-trace",
        action="store_true",
        help="Include map-runner simulation step trace in temporary episode records.",
    )
    parser.add_argument("--output", type=Path, help="Write JSON report path.")
    return parser


def main() -> int:
    """Run the command-line diagnostic report builder."""

    args = _build_parser().parse_args()
    config = FeasibilityDiagnosticConfig(
        scenario_path=args.scenario_config,
        families=tuple(args.family or DEFAULT_FAMILIES),
        seeds_per_scenario=args.seeds_per_scenario,
        baseline_algo=args.baseline_algo,
        oracle_algo=args.oracle_algo,
        extended_time_multiplier=args.extended_time_multiplier,
        run_extended_time=args.include_extended_time,
        record_simulation_step_trace=args.record_simulation_step_trace,
    )
    report = run_feasibility_diagnostics(config)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
