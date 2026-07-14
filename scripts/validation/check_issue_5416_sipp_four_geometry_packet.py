#!/usr/bin/env python3
"""Fail-closed checker for the issue #5416 four-geometry SIPP packet.

The checker validates the tracked preregistration and runs the repository's
``scenario_cert.v1`` geometry/solvability check for the four selected rows. It
does not run planner episodes, submit compute, or interpret benchmark outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from robot_sf.benchmark.algorithm_metadata import canonical_algorithm_name
from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.scenario_certification import certificate_to_dict, certify_scenario_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = REPO_ROOT / "configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml"
SCHEMA_VERSION = "issue_5416_sipp_four_geometry_preregistration.v1"
EXPECTED_SCENARIOS = (
    "classic_head_on_corridor_low",
    "classic_doorway_low",
    "classic_station_platform_medium",
    "classic_merging_low",
)
EXPECTED_PLANNERS = (
    "sipp_lattice",
    "hybrid_rule_v0_minimal",
    "teb",
    "nmpc_social",
    "dwa",
)
EXPECTED_PRIMARY_OUTCOMES = (
    "collision_free_success_rate",
    "pedestrian_and_static_collision_count_rate",
    "deadlock_timeout_max_steps_rate",
    "paired_progress_and_time_to_goal_conditional_on_collision_free_completion",
    "planner_step_runtime_median_p95",
)
EXPECTED_DIAGNOSTICS = (
    "expansion_limit_hits",
    "runtime_bound_exits",
    "fallback_count",
    "commitment_invalidations",
)
ALLOWED_CERTIFICATION_ELIGIBILITY = {"eligible", "stress_only"}
FORBIDDEN_TRANSIENT_KEYS = {
    "target_host",
    "packet_lineage",
    "queue",
    "slurm_job_id",
    "route_command",
}


class PacketError(ValueError):
    """Raised when the tracked packet is incomplete or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    _require(isinstance(value, dict), f"{key} must be a mapping")
    return value


def _repo_path(value: Any, key: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{key} must be a path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts, f"{key} must be repo-relative")
    return value


def _walk_for_transient_keys(value: Any, path: str = "packet") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require(
                key not in FORBIDDEN_TRANSIENT_KEYS, f"{path}.{key} is transient routing state"
            )
            _walk_for_transient_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_for_transient_keys(child, f"{path}[{index}]")


def load_packet(path: Path) -> dict[str, Any]:
    """Load a YAML packet and require a mapping root."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "packet must be a YAML mapping")
    return payload


def _check_certification(row: dict[str, Any], *, root: Path) -> dict[str, Any]:
    """Certify one selected scenario and return a compact gate result."""
    scenario_id = str(row.get("scenario_id"))
    source_path = root / _repo_path(row.get("source_path"), f"{scenario_id}.source_path")
    certificates = certify_scenario_file(source_path, scenario_id=scenario_id)
    _require(len(certificates) == 1, f"{scenario_id} must yield exactly one certificate")
    payload = certificate_to_dict(certificates[0])
    eligibility = str(payload.get("benchmark_eligibility", ""))
    classification = str(payload.get("classification", ""))
    return {
        "scenario_id": scenario_id,
        "source_path": row["source_path"],
        "classification": classification,
        "benchmark_eligibility": eligibility,
        "reasons": payload.get("reasons", []),
        "route_count": len(payload.get("route_certificates", [])),
        "gate": "pass" if eligibility in ALLOWED_CERTIFICATION_ELIGIBILITY else "blocked",
    }


def validate_packet(packet: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    """Validate the preregistration and current geometry gate."""
    root = repo_root or REPO_ROOT
    _walk_for_transient_keys(packet)
    _require(packet.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    _require(packet.get("issue") == 5416, "issue must be 5416")
    _require(packet.get("parent_issue") == 5306, "parent_issue must be 5306")
    _require(packet.get("predecessor_issue") == 5413, "predecessor_issue must be 5413")
    claim = str(packet.get("claim_boundary", "")).lower()
    for phrase in (
        "preregistration",
        "does not run",
        "paper/dissertation",
        "exploratory_synthetic",
    ):
        _require(phrase in claim, f"claim_boundary must mention {phrase}")

    execution = _mapping(packet, "execution_boundary")
    for key in (
        "benchmark_campaign_run_in_this_pr",
        "compute_submit_authorized",
        "paper_claim_edits",
        "metric_semantics_changes",
        "fallback_or_degraded_success_allowed",
    ):
        _require(execution.get(key) is False, f"execution_boundary.{key} must be false")

    scenarios = _mapping(packet, "scenario_contract")
    _repo_path(scenarios.get("scenario_matrix"), "scenario_contract.scenario_matrix")
    _require((root / scenarios["scenario_matrix"]).is_file(), "scenario matrix is missing")
    rows = scenarios.get("selected_rows")
    _require(isinstance(rows, list), "selected_rows must be a list")
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"selected_rows[{index}] must be a mapping")
    _require(
        tuple(row.get("scenario_id") for row in rows) == EXPECTED_SCENARIOS,
        "scenario order mismatch",
    )
    _require(scenarios.get("smoke_seed") == 111, "smoke seed must be 111")
    _require(
        scenarios.get("result_producing_seeds") == [111, 112, 113, 114, 115], "seed set mismatch"
    )
    _require(scenarios.get("horizon_steps") == 500, "horizon must be 500")
    _require(scenarios.get("dt_seconds") == 0.1, "dt must be 0.1")
    pairing = _mapping(scenarios, "pairing")
    for key in (
        "same_scenario_seed_set_across_planners",
        "scenario_order_frozen",
        "no_seed_substitution",
    ):
        _require(pairing.get(key) is True, f"pairing.{key} must be true")
    gate = _mapping(scenarios, "geometry_gate")
    _require(gate.get("certifier") == "scenario_cert.v1", "scenario_cert.v1 is required")
    _require(
        set(gate.get("allowed_eligibility", [])) == ALLOWED_CERTIFICATION_ELIGIBILITY,
        "geometry eligibility policy mismatch",
    )

    certification: list[dict[str, Any]] = []
    for row in rows:
        certification.append(_check_certification(row, root=root))

    roster = _mapping(packet, "planner_roster")
    planner_rows = roster.get("required")
    _require(isinstance(planner_rows, list), "planner_roster.required must be a list")
    for index, row in enumerate(planner_rows):
        _require(isinstance(row, dict), f"planner_roster.required[{index}] must be a mapping")
    _require(
        tuple(row.get("planner_id") for row in planner_rows) == EXPECTED_PLANNERS,
        "planner roster mismatch",
    )
    for row in planner_rows:
        algorithm = str(row.get("algorithm", ""))
        readiness = get_algorithm_readiness(algorithm)
        _require(readiness is not None, f"unknown planner readiness key: {algorithm}")
        _require(
            readiness.requires_explicit_opt_in is True, f"{algorithm} must remain explicit-opt-in"
        )
        planner_id = str(row.get("planner_id"))
        _require(
            canonical_algorithm_name(algorithm) == canonical_algorithm_name(planner_id),
            f"planner alias mismatch for {planner_id}",
        )
        config_path = _repo_path(row.get("config_path"), f"{planner_id}.config_path")
        resolved = root / config_path
        _require(resolved.is_file(), f"{planner_id} config missing: {config_path}")
        config = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        _require(
            isinstance(config, dict) and config.get("allow_testing_algorithms") is True,
            f"{planner_id} config must opt in explicitly",
        )

    outcomes = _mapping(packet, "outcomes")
    _require(
        outcomes.get("primary_in_order") == list(EXPECTED_PRIMARY_OUTCOMES),
        "primary outcome order mismatch",
    )
    _require(
        outcomes.get("diagnostics") == list(EXPECTED_DIAGNOSTICS),
        "planner diagnostics mismatch",
    )
    _require(
        "collision regression" in str(outcomes.get("advancement_rule", "")),
        "advancement rule must protect collisions",
    )

    provenance = _mapping(packet, "provenance_contract")
    _require(
        provenance.get("durable_evidence_required_before_claim") is True,
        "durable evidence gate is required",
    )
    durable = _repo_path(
        provenance.get("durable_evidence_path"), "provenance_contract.durable_evidence_path"
    )
    _require(
        PurePosixPath(durable).parts[:3] == ("docs", "context", "evidence"),
        "durable evidence must be under docs/context/evidence",
    )
    validation = _mapping(packet, "validation")
    _require(
        validation.get("campaign_execution_allowed_in_this_pr") is False,
        "campaign execution must be disabled",
    )
    readiness_decision = _mapping(packet, "readiness_decision")
    for key in (
        "benchmark_campaign_run",
        "compute_submit_authorized",
        "ranking_claim_promotion",
        "paper_claim_edits",
        "fallback_degraded_success_allowed",
    ):
        _require(readiness_decision.get(key) is False, f"readiness_decision.{key} must be false")

    blocked_rows = [row["scenario_id"] for row in certification if row["gate"] != "pass"]
    return {
        "status": "ready" if not blocked_rows else "blocked",
        "issue": 5416,
        "schema_version": SCHEMA_VERSION,
        "planner_count": len(planner_rows),
        "scenario_count": len(certification),
        "certification": certification,
        "blocked_rows": blocked_rows,
        "campaign_execution_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Validate the packet and emit a compact JSON or text result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        result = validate_packet(load_packet(args.packet))
    except (
        OSError,
        PacketError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        yaml.YAMLError,
    ) as exc:
        if args.as_json:
            print(json.dumps({"status": "not_ready", "error": str(exc)}))
        else:
            print(f"not_ready: {exc}")
        return 1
    if args.as_json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"{result['status']}: issue #5416 packet ({result['scenario_count']} scenarios, {result['planner_count']} planners)"
        )
        for row in result["certification"]:
            print(
                f"  {row['gate']}: {row['scenario_id']} ({row['classification']}/{row['benchmark_eligibility']})"
            )
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
