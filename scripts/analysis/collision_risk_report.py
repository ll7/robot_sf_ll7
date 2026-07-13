"""Latency and action-sensitivity report for the action-conditioned collision-risk API.

Issue #5444. This CLI runs the constant-velocity Monte Carlo baseline over a
frozen, small reference workload and reports:

- p50/p95/p99 estimate latency and deadline misses against the 100 ms control
  deadline, with an ``online`` / ``offline_only`` classification;
- the joint contact probability and its union-bound and (intentionally invalid)
  independence approximations for each candidate action;
- whether risk differs in the expected direction between two candidate actions
  from the same state.

Status: API + baseline fixture evidence, not a calibrated benchmark risk claim.
The report writes JSON to the git-ignored ``output/`` tree by default; it never
mutates benchmark metric semantics and emits no ``safe`` verdict.

Example:
    uv run python scripts/analysis/collision_risk_report.py \
        --config configs/research/collision_risk_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.nav.predictive_types import PedestrianState
from robot_sf.research.collision_risk import (
    RISK_SCHEMA_VERSION,
    RiskEstimatorConfig,
    action_from_constant_velocity,
    estimate_action_conditioned_risk,
    latency_summary_from_samples,
)

DEFAULT_CONFIG = Path("configs/research/collision_risk_baseline.yaml")
DEFAULT_OUTPUT = Path("output/collision_risk/collision_risk_report.json")


def _load_config(path: Path) -> dict[str, Any]:
    """Load the YAML workload/estimator config, failing closed on missing keys."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "estimator" not in data or "workload" not in data:
        raise ValueError(f"config {path} must define 'estimator' and 'workload' blocks")
    return data


def _build_estimator_config(estimator: dict[str, Any]) -> RiskEstimatorConfig:
    """Build a :class:`RiskEstimatorConfig` from the config block."""
    fields = {f.name for f in RiskEstimatorConfig.__dataclass_fields__.values()}
    return RiskEstimatorConfig(**{key: value for key, value in estimator.items() if key in fields})


def _build_pedestrians(workload: dict[str, Any]) -> list[PedestrianState]:
    """Construct the frozen pedestrian set from the workload block."""
    pedestrians = []
    for entry in workload["pedestrians"]:
        pedestrians.append(
            PedestrianState(
                id=int(entry["id"]),
                position=np.asarray(entry["position"], dtype=float),
                velocity=np.asarray(entry["velocity"], dtype=float),
            )
        )
    return pedestrians


def _run_action(
    action_spec: dict[str, Any],
    robot_start: list[float],
    pedestrians: list[PedestrianState],
    config: RiskEstimatorConfig,
    repeats: int,
) -> dict[str, Any]:
    """Time ``repeats`` estimates for one candidate action and summarise them."""
    action = action_from_constant_velocity(
        str(action_spec["action_id"]),
        robot_start,
        action_spec["velocity"],
        horizon_steps=config.horizon_steps,
        dt_s=config.dt_s,
    )
    latencies_ms: list[float] = []
    estimate = None
    for _ in range(repeats):
        start_ns = time.perf_counter_ns()
        estimate = estimate_action_conditioned_risk(
            action, pedestrians, config, measure_latency=False
        )
        latencies_ms.append((time.perf_counter_ns() - start_ns) / 1e6)

    assert estimate is not None
    latency = latency_summary_from_samples(latencies_ms, deadline_ms=config.deadline_ms)
    return {
        "action_id": action.action_id,
        "joint_contact_probability": estimate.joint_contact_probability,
        "union_bound_probability": estimate.union_bound_probability,
        "independence_approx_probability": estimate.independence_approx_probability,
        "deterministic_ttc_s": estimate.deterministic.ttc_s,
        "deterministic_min_clearance_m": estimate.deterministic.min_clearance_m,
        "abstained": estimate.uncertainty.abstained,
        "latency": latency.to_dict(),
        "estimate": estimate.to_dict(),
    }


def build_report(config_path: Path) -> dict[str, Any]:
    """Run the frozen reference workload and return a JSON-safe report dict."""
    data = _load_config(config_path)
    estimator_config = _build_estimator_config(data["estimator"])
    workload = data["workload"]
    pedestrians = _build_pedestrians(workload)
    repeats = int(workload.get("repeats", 200))
    robot_start = list(workload["robot_start"])

    action_results = [
        _run_action(action_spec, robot_start, pedestrians, estimator_config, repeats)
        for action_spec in workload["candidate_actions"]
    ]

    joint_by_action = {
        result["action_id"]: result["joint_contact_probability"] for result in action_results
    }
    higher = workload.get("higher_risk_action")
    lower = workload.get("lower_risk_action")
    action_sensitivity = None
    if higher in joint_by_action and lower in joint_by_action:
        action_sensitivity = {
            "higher_risk_action": higher,
            "lower_risk_action": lower,
            "higher_joint": joint_by_action[higher],
            "lower_joint": joint_by_action[lower],
            "direction_as_expected": joint_by_action[higher] > joint_by_action[lower],
        }

    p95_values = [result["latency"]["p95_ms"] for result in action_results]
    worst_p95 = max(p95_values)
    classification = "online" if worst_p95 <= estimator_config.deadline_ms else "offline_only"

    return {
        "schema_version": RISK_SCHEMA_VERSION,
        "config": str(config_path),
        "estimator_config_hash": estimator_config.config_hash(),
        "n_pedestrians": len(pedestrians),
        "repeats": repeats,
        "deadline_ms": estimator_config.deadline_ms,
        "worst_p95_ms": worst_p95,
        "latency_classification": classification,
        "action_sensitivity": action_sensitivity,
        "actions": action_results,
        "claim_boundary": (
            "API + baseline fixture evidence; not calibrated benchmark risk. "
            "Hard guards remain authoritative; no safe verdict is emitted."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build and write the collision-risk report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a compact human-readable summary to stdout.",
    )
    args = parser.parse_args(argv)

    report = build_report(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"wrote {args.output}")
    print(
        f"latency: worst_p95={report['worst_p95_ms']:.2f} ms "
        f"(deadline {report['deadline_ms']:.0f} ms) -> {report['latency_classification']}"
    )
    if report["action_sensitivity"] is not None:
        sens = report["action_sensitivity"]
        print(
            f"action-sensitivity: {sens['higher_risk_action']} joint={sens['higher_joint']:.3f} "
            f"vs {sens['lower_risk_action']} joint={sens['lower_joint']:.3f} "
            f"-> direction_as_expected={sens['direction_as_expected']}"
        )
    if args.print_summary:
        for result in report["actions"]:
            print(
                f"  {result['action_id']}: joint={result['joint_contact_probability']:.3f} "
                f"union={result['union_bound_probability']:.3f} "
                f"indep={result['independence_approx_probability']:.3f} "
                f"ttc={result['deterministic_ttc_s']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
