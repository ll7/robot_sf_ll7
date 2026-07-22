#!/usr/bin/env python3
"""Fail-closed checker for the issue #5592 cross-matrix pre-registration.

The checker validates the tracked second-matrix contract, loads both scenario
matrices, and verifies that the promoted atomic rows carry the expected map and
failure-mode metadata. It does not run planner episodes, submit compute, or
interpret a structural ranking.
"""
# evidence-writer-exempt: references evidence paths but does not write to evidence tree; guarded by AST analysis


from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from loguru import logger

from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = REPO_ROOT / "configs/benchmarks/issue_5592_cross_matrix_preregistration.yaml"
SCHEMA_VERSION = "issue_5592_cross_matrix_preregistration.v1"
EXPECTED_SEEDS = (20, 21, 22, 23, 24)
EXPECTED_SCENARIOS = (
    ("corner_90_turn", "atomic_corner_90_test", "topology", "oscillation"),
    ("u_trap_local_minimum", "atomic_u_trap_test", "topology", "local_minima"),
    ("corridor_following", "atomic_corridor_test", "topology", "oscillation"),
)
EXPECTED_ROSTER = {
    "constraint_first_hybrid": (
        "scenario_adaptive_hybrid_orca_v1",
        "hybrid_rule_v3_fast_progress_static_escape",
    ),
    "learned_policy": ("ppo", "guarded_ppo"),
    "predictive": ("prediction_planner", "prediction_mpc", "prediction_mpc_cbf"),
    "baseline_reactive": ("goal", "social_force", "orca", "socnav_sampling", "sacadrl"),
}
EXPECTED_ARTIFACT_FILES = (
    "README.md",
    "metadata.json",
    "cross_matrix_agreement.csv",
    "integration_report.md",
    "SHA256SUMS",
)
FORBIDDEN_TRANSIENT_KEYS = {
    "target_host",
    "packet_lineage",
    "queue",
    "slurm_job_id",
    "route_command",
}


class PacketError(ValueError):
    """Raised when the tracked pre-registration is incomplete or unsafe."""


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


def _resolve_existing_path(root: Path, value: Any, key: str) -> Path:
    relative = _repo_path(value, key)
    resolved = root / relative
    _require(resolved.is_file(), f"{key} is missing: {relative}")
    return resolved


def _check_candidate_rows(packet: dict[str, Any], *, root: Path) -> dict[str, Any]:
    candidate = _mapping(packet, "candidate_contract")
    matrix_path = _resolve_existing_path(
        root, candidate.get("scenario_matrix"), "candidate_contract.scenario_matrix"
    )
    rows = load_scenarios(matrix_path, base_dir=matrix_path)
    expected_ids = tuple(row[0] for row in EXPECTED_SCENARIOS)
    _require(
        tuple(row.get("name") for row in rows) == expected_ids, "candidate scenario order mismatch"
    )
    expected_by_id = {row[0]: row for row in EXPECTED_SCENARIOS}
    for row in rows:
        scenario_id = str(row.get("name"))
        expected = expected_by_id[scenario_id]
        metadata = row.get("metadata")
        _require(isinstance(metadata, dict), f"{scenario_id} metadata must be a mapping")
        _require(row.get("map_id") == expected[1], f"{scenario_id} map_id mismatch")
        _require(
            metadata.get("primary_capability") == expected[2],
            f"{scenario_id} primary_capability mismatch",
        )
        _require(
            metadata.get("target_failure_mode") == expected[3],
            f"{scenario_id} target_failure_mode mismatch",
        )
        _require(
            list(row.get("seeds", [])) == list(EXPECTED_SEEDS),
            f"{scenario_id} seed schedule mismatch",
        )

    selected_rows = candidate.get("selected_rows")
    _require(isinstance(selected_rows, list), "candidate_contract.selected_rows must be a list")
    for index, row in enumerate(selected_rows):
        _require(isinstance(row, dict), f"selected_rows[{index}] must be a mapping")
    _require(
        tuple(
            (
                row.get("scenario_id"),
                row.get("map_id"),
                row.get("primary_capability"),
                row.get("target_failure_mode"),
            )
            for row in selected_rows
        )
        == EXPECTED_SCENARIOS,
        "candidate selected-row metadata mismatch",
    )
    metadata_contract = _mapping(candidate, "metadata_contract")
    _require(
        metadata_contract.get("required_fields")
        == [
            "map_id",
            "primary_capability",
            "target_failure_mode",
        ],
        "candidate metadata field contract mismatch",
    )
    _resolve_existing_path(
        root,
        metadata_contract.get("source_document"),
        "candidate_contract.metadata_contract.source_document",
    )
    return {
        "path": str(matrix_path.relative_to(root)),
        "scenario_count": len(rows),
        "seeds": list(EXPECTED_SEEDS),
    }


def validate_packet(packet: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    """Validate the pre-registration and both matrix input contracts."""
    root = repo_root or REPO_ROOT
    _walk_for_transient_keys(packet)
    _require(packet.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    _require(packet.get("issue") == 5592, "issue must be 5592")
    _require(packet.get("status") == "pre_registered", "status must be pre_registered")
    _require(packet.get("benchmark_evidence") is False, "benchmark_evidence must be false")

    claim = str(packet.get("claim_boundary", "")).lower()
    for phrase in (
        "generalization check",
        "two matrices",
        "does not establish",
        "paper/dissertation",
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

    reference = _mapping(packet, "reference_contract")
    reference_matrix = _resolve_existing_path(
        root, reference.get("scenario_matrix"), "reference_contract.scenario_matrix"
    )
    _resolve_existing_path(
        root,
        reference.get("structural_ranking_source"),
        "reference_contract.structural_ranking_source",
    )
    _resolve_existing_path(
        root, reference.get("seed_schedule_source"), "reference_contract.seed_schedule_source"
    )
    reference_rows = load_scenarios(reference_matrix, base_dir=reference_matrix)
    _require(
        len(reference_rows) == reference.get("expected_scenario_count"),
        "reference scenario count mismatch",
    )

    candidate_result = _check_candidate_rows(packet, root=root)

    pairing = _mapping(packet, "pairing_contract")
    _require(tuple(pairing.get("seeds", [])) == EXPECTED_SEEDS, "pairing seed schedule mismatch")
    _require(pairing.get("seed_count") == len(EXPECTED_SEEDS), "pairing seed count mismatch")
    _require(pairing.get("horizon_steps") == 600, "pairing horizon must be 600")
    _require(pairing.get("dt_seconds") == 0.1, "pairing dt must be 0.1")
    for key in (
        "same_seed_schedule_across_matrices",
        "same_roster_across_matrices",
        "scenario_order_frozen",
        "no_seed_substitution",
    ):
        _require(pairing.get(key) is True, f"pairing.{key} must be true")

    roster = _mapping(packet, "planner_roster").get("structural_classes")
    _require(isinstance(roster, dict), "planner_roster.structural_classes must be a mapping")
    _require(
        {key: tuple(value) for key, value in roster.items()} == EXPECTED_ROSTER,
        "planner roster mismatch",
    )

    comparison = _mapping(packet, "comparison_contract")
    _require(
        comparison.get("primary_output") == "cross_matrix_agreement.csv",
        "agreement table must be primary output",
    )
    _require(
        comparison.get("rank_unit") == "structural_class", "rank unit must be structural_class"
    )
    _require(
        comparison.get("metric") == "constraints_first_structural_rank", "comparison metric drift"
    )
    required_columns = comparison.get("required_columns")
    _require(isinstance(required_columns, list), "comparison required_columns must be a list")
    _require(
        "agreement_status" in required_columns and "caveat" in required_columns,
        "agreement table needs status and caveat columns",
    )
    _require(
        comparison.get("must_emit_disagreement_rows") is True, "disagreement rows must be emitted"
    )

    artifact = _mapping(packet, "artifact_contract")
    evidence_root = _repo_path(
        artifact.get("durable_evidence_root"), "artifact_contract.durable_evidence_root"
    )
    _require(
        PurePosixPath(evidence_root).parts[:3] == ("docs", "context", "evidence"),
        "durable evidence must be under docs/context/evidence",
    )
    _require(artifact.get("source_commit_required") is True, "source commit provenance is required")
    _require(
        artifact.get("no_worktree_output_as_durable_evidence") is True,
        "worktree output cannot be durable evidence",
    )
    required_files = artifact.get("required_files")
    _require(
        isinstance(required_files, (list, tuple))
        and tuple(required_files) == EXPECTED_ARTIFACT_FILES,
        "artifact contract must declare the complete integration artifact set",
    )

    readiness = _mapping(packet, "readiness_decision")
    for key in (
        "benchmark_campaign_run",
        "compute_submit_authorized",
        "ranking_claim_promotion",
        "paper_claim_edits",
        "fallback_degraded_success_allowed",
    ):
        _require(readiness.get(key) is False, f"readiness_decision.{key} must be false")

    return {
        "status": "ready",
        "issue": 5592,
        "schema_version": SCHEMA_VERSION,
        "reference_scenario_count": len(reference_rows),
        "candidate": candidate_result,
        "planner_count": sum(len(value) for value in EXPECTED_ROSTER.values()),
        "seed_count": len(EXPECTED_SEEDS),
        "campaign_execution_allowed": False,
        "agreement_table": comparison["primary_output"],
    }


def _configure_machine_readable_logging() -> None:
    """Route validation diagnostics to stderr so JSON stdout stays parseable."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")


def _run_machine_readable_validation(packet: Path) -> int:
    """Run JSON validation without mutating the caller's process-global Loguru sinks."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--packet",
        str(packet),
        "--json",
        "--machine-readable-child",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        print(f"isolated validation failed: {exc}", file=sys.stderr)
        print(json.dumps({"status": "not_ready", "error": str(exc)}))
        return 1

    if completed.stderr:
        sys.stderr.write(completed.stderr)
    try:
        json.loads(completed.stdout)
    except json.JSONDecodeError:
        if completed.stdout:
            sys.stderr.write(completed.stdout)
        print(
            json.dumps({"status": "not_ready", "error": "isolated validation emitted invalid JSON"})
        )
        return completed.returncode or 1

    sys.stdout.write(completed.stdout)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    """Validate the packet and emit a compact JSON or text result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--machine-readable-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.as_json and not args.machine_readable_child:
        return _run_machine_readable_validation(args.packet)
    if args.as_json:
        _configure_machine_readable_logging()
    try:
        result = validate_packet(load_packet(args.packet))
    except (OSError, PacketError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        if args.as_json:
            print(json.dumps({"status": "not_ready", "error": str(exc)}))
        else:
            print(f"not_ready: {exc}")
        return 1
    if args.as_json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"{result['status']}: issue #5592 packet "
            f"({result['candidate']['scenario_count']} topology scenarios, "
            f"{result['planner_count']} planners, {result['seed_count']} seeds)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
