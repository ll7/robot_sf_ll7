#!/usr/bin/env python3
"""Fail-closed checker for the issue #8072 inverse-force smoke receipt."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_sf.prediction._contract_utils import stable_digest

SCHEMA_VERSION = "robot_sf.goal_force_inverse_smoke_report.v1"
CLAIM_BOUNDARY = "implementation_integrity_smoke"
ARM_NAMES = frozenset(
    {
        "h1_heading_baseline",
        "h2_observation_reconstructed",
        "h3_observation_reconstructed",
        "oracle_component_upper_bound",
    }
)
ROOT_KEYS = frozenset(
    {
        "schema_version",
        "issue",
        "status",
        "claim_boundary",
        "fixture_id",
        "seed",
        "track_id",
        "tracking_epoch_id",
        "source_hashes",
        "smoke_config_digest",
        "estimator_config",
        "estimator_config_hash",
        "candidate_generation",
        "oracle_trace_digest",
        "evaluation_truth",
        "leakage_canary",
        "arms",
        "notes",
        "report_digest",
    }
)
FORBIDDEN_ACTOR_KEYS = frozenset(
    {
        "active_goal_xy",
        "assigned_route",
        "future_trajectory",
        "oracle_goal",
        "oracle_route",
        "route_waypoint_index",
        "simulator_goal",
        "simulator_pedestrian_id",
        "true_goal",
        "true_goal_xy",
    }
)
METRIC_KEYS = frozenset(
    {
        "force_mae",
        "force_rmse",
        "direction_error_rad",
        "covariance_coverage_95",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{field_name} must be a mapping")
    return dict(value)


def _finite(value: Any, field_name: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), f"{field_name} must be finite")


def _assert_finite(value: Any, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")
    else:
        _finite(value, path)


def _assert_no_forbidden_actor_keys(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(
                str(key) not in FORBIDDEN_ACTOR_KEYS,
                f"{path}.{key} exposes an oracle-only actor field",
            )
            _assert_no_forbidden_actor_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_actor_keys(child, f"{path}[{index}]")


def _check_covariance(value: Any, field_name: str) -> None:
    _require(
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in value),
        f"{field_name} must be a 2x2 matrix",
    )
    _require(abs(value[0][1] - value[1][0]) <= 1e-9, f"{field_name} must be symmetric")
    determinant = value[0][0] * value[1][1] - value[0][1] ** 2
    _require(value[0][0] >= -1e-12, f"{field_name} must be positive semidefinite")
    _require(value[1][1] >= -1e-12, f"{field_name} must be positive semidefinite")
    _require(determinant >= -1e-12, f"{field_name} must be positive semidefinite")


def _check_estimate(name: str, arm: dict[str, Any]) -> None:
    required = {
        "history_length",
        "history_age_s",
        "mode",
        "information_source",
        "inferred_preferred_speed_mps",
        "component_availability",
        "saturation_status",
        "censoring_state",
        "arrival_probability",
        "braking_probability",
        "runtime_ms",
        "metrics",
        "estimate",
        "estimate_digest",
    }
    _require(required.issubset(arm), f"{name} arm is incomplete")
    estimate = _mapping(arm["estimate"], f"arms.{name}.estimate")
    _require(estimate.get("schema_version") == "goal_force_inverse.v1", f"{name} schema drifted")
    _require(estimate.get("claim_boundary") == CLAIM_BOUNDARY, f"{name} claim boundary drifted")
    _require(arm["estimate_digest"] == stable_digest(estimate), f"{name} estimate digest mismatch")
    _require(float(arm["runtime_ms"]) >= 0.0, f"{name} runtime must be non-negative")
    _require(0.0 <= float(arm["arrival_probability"]) <= 1.0, f"{name} arrival probability invalid")
    _require(0.0 <= float(arm["braking_probability"]) <= 1.0, f"{name} braking probability invalid")
    metrics = _mapping(arm["metrics"], f"arms.{name}.metrics")
    _require(set(metrics) == METRIC_KEYS, f"{name} metric fields drifted")
    for metric_name in ("force_mae", "force_rmse", "direction_error_rad"):
        if metrics[metric_name] is not None:
            _require(
                float(metrics[metric_name]) >= 0.0, f"{name}.{metric_name} must be non-negative"
            )
    if metrics["covariance_coverage_95"] is not None:
        _require(
            isinstance(metrics["covariance_coverage_95"], bool), f"{name} coverage must be bool"
        )
    force_estimate = estimate.get("force_estimate")
    if name == "h1_heading_baseline":
        _require(estimate.get("history_length") == 1, "H=1 history length drifted")
        _require(estimate.get("mode") == "heading_baseline", "H=1 mode drifted")
        _require(force_estimate is None, "H=1 must not emit force magnitude")
        _require(metrics["force_mae"] is None, "H=1 must not report force error")
        _require(estimate.get("belief") is not None, "H=1 must emit actor belief")
    elif name == "h2_observation_reconstructed":
        _require(estimate.get("history_length") == 2, "H=2 history length drifted")
        _require(estimate.get("mode") == "observation_reconstructed", "H=2 mode drifted")
        _require(force_estimate is not None, "H=2 must emit force estimate")
    elif name == "h3_observation_reconstructed":
        _require(estimate.get("history_length") == 3, "H=3 history length drifted")
        _require(estimate.get("mode") == "observation_reconstructed", "H=3 mode drifted")
        _require(force_estimate is not None, "H=3 must emit force estimate")
    else:
        _require(estimate.get("history_length") == 2, "oracle history length drifted")
        _require(estimate.get("mode") == "oracle_component_upper_bound", "oracle mode drifted")
        _require(estimate.get("belief") is None, "oracle arm must not emit actor belief")
        _require(force_estimate is not None, "oracle arm must emit force estimate")
    if force_estimate is not None:
        force_payload = _mapping(force_estimate, f"arms.{name}.force_estimate")
        _check_covariance(
            force_payload["covariance_xy"], f"arms.{name}.force_estimate.covariance_xy"
        )
    _check_covariance(
        estimate["acceleration_covariance_xy"],
        f"arms.{name}.acceleration_covariance_xy",
    )


def validate_report(payload: Mapping[str, Any]) -> None:
    """Validate a report without running an experiment or accepting fallbacks."""
    report = _mapping(payload, "report")
    _require(set(report) == ROOT_KEYS, "report fields drifted")
    _require(report["schema_version"] == SCHEMA_VERSION, "report schema_version drifted")
    _require(report["issue"] == 8072, "report issue must be 8072")
    _require(report["status"] == "diagnostic_only_valid", "report status is not valid")
    _require(report["claim_boundary"] == CLAIM_BOUNDARY, "report claim boundary drifted")
    _require(
        report["report_digest"]
        == stable_digest({key: value for key, value in report.items() if key != "report_digest"}),
        "report digest mismatch",
    )
    for field_name in ("smoke_config_digest", "estimator_config_hash", "oracle_trace_digest"):
        value = report[field_name]
        _require(
            isinstance(value, str) and len(value) == 64, f"{field_name} must be a SHA-256 digest"
        )
    source_hashes = _mapping(report["source_hashes"], "source_hashes")
    _require(
        bool(source_hashes)
        and all(isinstance(value, str) and len(value) == 64 for value in source_hashes.values()),
        "source hashes are incomplete",
    )
    generation = _mapping(report["candidate_generation"], "candidate_generation")
    _require(
        generation.get("claim_boundary") == "candidate_generation_only",
        "candidate claim boundary drifted",
    )
    _require(isinstance(generation.get("candidate_set_digest"), str), "candidate digest is missing")
    candidate_set = _mapping(generation.get("candidate_set"), "candidate_generation.candidate_set")
    candidates = candidate_set.get("candidates")
    _require(
        isinstance(candidates, list)
        and any(item.get("role") == "unknown" for item in candidates if isinstance(item, Mapping)),
        "candidate provider must retain unknown",
    )
    canary = _mapping(report["leakage_canary"], "leakage_canary")
    _require(canary.get("status") == "pass", "actor/oracle leakage canary failed")
    _require(
        canary.get("actor_digest_before") == canary.get("actor_digest_after"),
        "actor bytes changed under oracle mutation",
    )
    arms = _mapping(report["arms"], "arms")
    _require(set(arms) == ARM_NAMES, "the four comparator arms are required")
    for name, raw_arm in arms.items():
        _check_estimate(name, _mapping(raw_arm, f"arms.{name}"))
    actor_payload = {
        name: arms[name]["estimate"] for name in ARM_NAMES if name != "oracle_component_upper_bound"
    }
    _assert_no_forbidden_actor_keys(actor_payload, "actor_arms")
    _assert_finite(report)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the checker command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON smoke receipt to validate.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate one receipt and print a compact status."""
    args = build_arg_parser().parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        validate_report(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid goal-force smoke receipt: {exc}") from exc
    print(json.dumps({"status": "valid", "input": str(args.input)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
