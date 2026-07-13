#!/usr/bin/env python3
"""Offline frozen-state counterfactual replay over controlled fixtures (issue #5442).

This runner drives the counterfactual-replay engine
(:mod:`robot_sf.benchmark.last_avoidable_replay`) over the controlled kinematic
fixtures in :mod:`robot_sf.benchmark.last_avoidable_fixtures` and emits one
``last_avoidable_replay.v1`` report per fixture. It runs no benchmark and makes no
metric or paper-grade claim; the fixtures are deterministic controlled models, not
the production simulator (which offers no snapshot seam — see
``docs/context/issue_5442_last_avoidable_replay.md``).

Reports are validated against
``robot_sf/benchmark/schemas/last_avoidable_replay.v1.json`` before they are
written. Runtime is reported per the offline contract; no online gate is applied.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jsonschema

from robot_sf.benchmark import last_avoidable_fixtures as fx
from robot_sf.benchmark.last_avoidable_replay import (
    SUBSTITUTION_HOLD,
    LastAvoidableReport,
    ReplayConfig,
    locate_last_avoidable,
)

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "last_avoidable_replay.v1.json"
)

# Fixture name -> (builder, action_set_id) for the acceptance-criteria cases.
_FIXTURES = {
    "preventable_late_braking": fx.preventable_late_braking_scenario,
    "already_unavoidable": fx.already_unavoidable_scenario,
    "two_action_interaction": fx.two_action_interaction_scenario,
    "nondeterministic_baseline": fx.nondeterministic_baseline_scenario,
    "missing_feasible_action": fx.missing_feasible_action_scenario,
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/issue_5442_last_avoidable_replay"),
        help="Directory for per-fixture JSON reports (git-ignored output).",
    )
    parser.add_argument(
        "--fixtures",
        nargs="*",
        choices=sorted(_FIXTURES),
        default=sorted(_FIXTURES),
        help="Subset of controlled fixtures to run (default: all).",
    )
    parser.add_argument(
        "--determinism-replays",
        type=int,
        default=20,
        help="Number of identical baseline replays used for the determinism check.",
    )
    return parser


def run_fixture(name: str, *, determinism_replays: int) -> LastAvoidableReport:
    """Run one named controlled fixture through the replay engine.

    Args:
        name: Fixture key from :data:`_FIXTURES`.
        determinism_replays: Baseline replays for the determinism check.

    Returns:
        The :class:`LastAvoidableReport` for the fixture, with runtime recorded.
    """
    scenario = _FIXTURES[name]()
    contact_step = fx.find_contact_step(scenario)
    if contact_step is None or contact_step < 1:
        raise ValueError(f"fixture {name!r} baseline does not collide (t_contact={contact_step})")
    horizon = contact_step + 6
    config = ReplayConfig(
        t_danger=0,
        t_contact=contact_step,
        horizon=horizon,
        substitution_mode=SUBSTITUTION_HOLD,
        determinism_replays=determinism_replays,
        action_set_id="decel_lattice",
        feasibility_filter="all_admissible_decel",
        collision_predicate="euclidean_distance<=collision_radius",
        pedestrian_response=scenario.pedestrian_response,
    )
    model = fx.KinematicCollisionModel(scenario)
    baseline_actions = fx.maintain_baseline_actions(contact_step + horizon + 2)
    start = time.perf_counter()
    report = locate_last_avoidable(model, baseline_actions, config)
    elapsed = time.perf_counter() - start
    return LastAvoidableReport(
        verdict=report.verdict,
        config=report.config,
        determinism=report.determinism,
        branches=report.branches,
        t_uca=report.t_uca,
        t_inevitable=report.t_inevitable,
        feasible_coverage=report.feasible_coverage,
        minimal_sufficient_interventions=report.minimal_sufficient_interventions,
        runtime_s=elapsed,
        abstained=report.abstained,
        abstain_reason=report.abstain_reason,
        notes=report.notes,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the requested fixtures, validate, write reports, and print a summary.

    Returns:
        Process exit code (0 on success).
    """
    args = _build_parser().parse_args(argv)
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for name in args.fixtures:
        report = run_fixture(name, determinism_replays=args.determinism_replays)
        payload = report.to_dict()
        jsonschema.validate(payload, schema)
        out_path = args.out_dir / f"{name}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        summary.append(
            {
                "fixture": name,
                "verdict": payload["verdict"],
                "t_uca": payload["t_uca"],
                "t_inevitable": payload["t_inevitable"],
                "abstain_reason": payload["abstain_reason"],
                "report": str(out_path),
            }
        )

    print(json.dumps({"schema_version": "last_avoidable_replay.v1", "results": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
