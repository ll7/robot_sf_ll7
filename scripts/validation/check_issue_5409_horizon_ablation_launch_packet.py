#!/usr/bin/env python3
"""Fail-closed readiness check for the issue #5409 h500/h600 horizon-ablation launch packet.

This checker validates the launch contract only. It never submits SLURM, runs a campaign,
stages a checkpoint, aggregates results, or promotes evidence. It fails closed if the packet
would let an operator submit without a staged checkpoint receipt, submit without an explicit
results directory, lose the roster/seed match, drop the scenario-matrix hash binding, count
fallback or degraded rows as evidence, or self-authorize compute.

It additionally re-verifies on disk that both referenced campaign configs exist and still
hash to the values the packet pins, so config drift blocks the launch instead of silently
invalidating the comparison.

Exit codes:
- 0: ``ready`` -- packet is a valid, fail-closed launch contract.
- 1: ``blocked`` -- packet is well-formed but a contract requirement is unmet.
- 2: ``malformed`` -- packet is missing or not a YAML mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = REPO_ROOT / "configs/benchmarks/issue_5409_horizon_ablation_launch_packet.yaml"
SCHEMA_VERSION = "issue-5409-horizon-ablation-launch-packet.v1"

EXPECTED_SCENARIO_MATRIX_HASH = "c10df617a87c"
EXPECTED_PLANNER_COUNT = 12
EXPECTED_SCENARIO_COUNT = 48
EXPECTED_SEEDS = (111, 112, 113)
EXPECTED_ROWS_PER_HORIZON = 1728
EXPECTED_HORIZONS = (500, 600)
EXPECTED_ROLES = ("h500", "h600")

REQUIRED_FAIL_CLOSED_STATUSES = frozenset(
    {"fallback", "degraded", "unavailable", "failed", "partial", "not_available", "diagnostic_only"}
)
REQUIRED_PER_ARM_ARTIFACTS = frozenset(
    {
        "campaign_manifest.json",
        "preflight/validate_config.json",
        "preflight/checkpoint_staging.json",
        "reports/matrix_summary.json",
        "reports/comparability_matrix.json",
    }
)
REQUIRED_PAIRED_ARTIFACTS = frozenset(
    {
        "matched_key_completeness.json",
        "paired_horizon_deltas.json",
    }
)
REQUIRED_SHARED_IDENTITY_FIELDS = frozenset(
    {"scenario_matrix_hash", "comparability_mapping_hash", "observation_noise_hash"}
)
RESULTS_DIR_PLACEHOLDER = "{submit_worktree}"

READY = "ready"
BLOCKED = "blocked"
MALFORMED = "malformed"
EXIT_CODES = {READY: 0, BLOCKED: 1, MALFORMED: 2}


class PacketError(ValueError):
    """Raised when the launch packet would not fail closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PacketError(f"{key} must be a mapping")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_matrix(packet: dict[str, Any]) -> dict[str, Any]:
    matrix = _require_mapping(packet, "matrix")
    _require(
        matrix.get("planner_count") == EXPECTED_PLANNER_COUNT,
        f"matrix.planner_count must be {EXPECTED_PLANNER_COUNT}",
    )
    _require(
        matrix.get("scenario_count") == EXPECTED_SCENARIO_COUNT,
        f"matrix.scenario_count must be {EXPECTED_SCENARIO_COUNT}",
    )
    _require(
        tuple(matrix.get("resolved_seeds") or ()) == EXPECTED_SEEDS,
        f"matrix.resolved_seeds must be {list(EXPECTED_SEEDS)}",
    )
    _require(
        matrix.get("seed_count") == len(EXPECTED_SEEDS),
        "matrix.seed_count must match resolved_seeds",
    )
    _require(
        matrix.get("rows_per_horizon") == EXPECTED_ROWS_PER_HORIZON,
        f"matrix.rows_per_horizon must be {EXPECTED_ROWS_PER_HORIZON}",
    )
    _require(
        matrix.get("rows_total") == 2 * EXPECTED_ROWS_PER_HORIZON,
        "matrix.rows_total must be both horizons combined",
    )
    _require(
        EXPECTED_PLANNER_COUNT * EXPECTED_SCENARIO_COUNT * len(EXPECTED_SEEDS)
        == EXPECTED_ROWS_PER_HORIZON,
        "row arithmetic does not close",
    )
    _require(matrix.get("varying_field") == "horizon", "matrix.varying_field must be horizon")
    _require(
        matrix.get("expected_scenario_matrix_hash") == EXPECTED_SCENARIO_MATRIX_HASH,
        "matrix.expected_scenario_matrix_hash drifted from the frozen comparability hash",
    )
    return matrix


def _validate_pair_validation(packet: dict[str, Any]) -> dict[str, Any]:
    pair = _require_mapping(packet, "pair_validation")
    _require(
        pair.get("validator") == "scripts/benchmark/validate_horizon_ablation_pair.py",
        "pair_validation.validator must be the checked-in pair validator",
    )
    _require(pair.get("is_valid") is True, "pair_validation.is_valid must be true")
    _require(pair.get("mismatch_count") == 0, "pair_validation.mismatch_count must be 0")
    validator = REPO_ROOT / str(pair["validator"])
    _require(validator.is_file(), f"pair validator missing on disk: {pair['validator']}")
    return pair


def _validate_arms(packet: dict[str, Any]) -> list[dict[str, Any]]:
    arms = packet.get("arms")
    _require(isinstance(arms, list) and len(arms) == 2, "arms must be a two-entry list")
    assert isinstance(arms, list)  # narrowed by _require

    roles = tuple(str(arm.get("role")) for arm in arms)
    _require(roles == EXPECTED_ROLES, f"arm roles must be {list(EXPECTED_ROLES)} in order")
    horizons = tuple(arm.get("horizon") for arm in arms)
    _require(horizons == EXPECTED_HORIZONS, f"arm horizons must be {list(EXPECTED_HORIZONS)}")

    for arm in arms:
        role = arm.get("role")
        config_rel = str(arm.get("config", ""))
        _require(bool(config_rel), f"arm {role} must declare a config")
        config_path = REPO_ROOT / config_rel
        _require(config_path.is_file(), f"arm {role} config missing on disk: {config_rel}")

        declared = str(arm.get("config_sha256", ""))
        _require(len(declared) == 64, f"arm {role} must pin a sha256 config digest")
        actual = _sha256_file(config_path)
        _require(
            actual == declared,
            f"arm {role} config drifted: {config_rel} is {actual}, packet pins {declared}",
        )

        campaign_id = str(arm.get("campaign_id", ""))
        _require(
            campaign_id == f"issue5409_horizon_ablation_{role}",
            f"arm {role} campaign_id must be issue5409_horizon_ablation_{role}",
        )

        results_dir = str(arm.get("results_dir_template", ""))
        _require(
            RESULTS_DIR_PLACEHOLDER in results_dir,
            f"arm {role} results_dir_template must be rooted at {RESULTS_DIR_PLACEHOLDER}",
        )
        _require(
            results_dir.endswith(campaign_id),
            f"arm {role} results_dir_template must end at its own campaign id",
        )

        report = str(arm.get("checkpoint_gate_report_template", ""))
        _require(
            report.startswith(results_dir) and report.endswith("checkpoint_staging.json"),
            f"arm {role} checkpoint gate receipt must live under its own results dir",
        )

    _require(
        arms[0]["results_dir_template"] != arms[1]["results_dir_template"],
        "the two horizons must not share a results directory",
    )
    return arms


def _validate_environment_identity(packet: dict[str, Any]) -> dict[str, Any]:
    env = _require_mapping(packet, "environment_identity")
    _require(
        env.get("entry_point") == "scripts/tools/run_camera_ready_benchmark.py",
        "environment_identity.entry_point must be the camera-ready runner",
    )
    entry = REPO_ROOT / str(env["entry_point"])
    _require(entry.is_file(), "environment_identity.entry_point missing on disk")
    _require(
        env.get("dependency_sync") == "uv sync --all-extras",
        "environment_identity.dependency_sync must pin the full-extras environment",
    )
    _require(
        env.get("arm_isolation") == "subprocess",
        "environment_identity.arm_isolation must be subprocess",
    )
    _require(
        bool(str(env.get("preflight_command_template", "")).strip()),
        "environment_identity must supply a preflight command",
    )
    _require(
        "--mode preflight" in str(env["preflight_command_template"]),
        "preflight_command_template must invoke mode=preflight",
    )

    measured = _require_mapping(env, "measured")
    _require(
        measured.get("scenario_matrix_hash") == EXPECTED_SCENARIO_MATRIX_HASH,
        "environment_identity.measured.scenario_matrix_hash drifted",
    )
    _require(
        measured.get("h500_config_hash") != measured.get("h600_config_hash"),
        "the two horizons must not share a config hash",
    )

    shared = frozenset(env.get("shared_across_pair") or ())
    _require(
        REQUIRED_SHARED_IDENTITY_FIELDS <= shared,
        "shared_across_pair must bind scenario matrix, comparability, and observation-noise hashes",
    )
    for field_name in REQUIRED_SHARED_IDENTITY_FIELDS:
        _require(
            bool(measured.get(field_name)),
            f"environment_identity.measured must record {field_name}",
        )
    return env


def _validate_checkpoint_gate(packet: dict[str, Any]) -> dict[str, Any]:
    gate = _require_mapping(packet, "checkpoint_gate")
    script_rel = str(gate.get("script", ""))
    _require(
        script_rel == "scripts/benchmark/submit_camera_ready_checkpoint_gate.sh",
        "checkpoint_gate.script must be the public pre-sbatch gate",
    )
    _require((REPO_ROOT / script_rel).is_file(), f"checkpoint gate missing on disk: {script_rel}")
    _require(
        gate.get("mode") == "enforced_staged",
        "checkpoint_gate.mode must be enforced_staged; metadata_only is not submit-safe",
    )
    _require(gate.get("submit_safe_required") is True, "checkpoint_gate must require submit-safe")
    _require(gate.get("submits_slurm") is False, "checkpoint_gate must not submit SLURM")
    _require(gate.get("runs_episodes") is False, "checkpoint_gate must not run episodes")
    _require(
        gate.get("receipt_filename") == "checkpoint_staging.json",
        "checkpoint_gate.receipt_filename must be checkpoint_staging.json",
    )
    _require(
        gate.get("status") == "pending_submit_node_execution",
        "checkpoint_gate.status must stay pending until receipts exist on the submit node",
    )
    local = _require_mapping(gate, "local_resolvability_check")
    _require(
        local.get("result") == "resolvable_not_submit_safe",
        "a resolvability probe must never be recorded as submit-safe",
    )
    _require(
        "{config}" in str(gate.get("command_template", ""))
        and "{checkpoint_gate_report}" in str(gate.get("command_template", "")),
        "checkpoint_gate.command_template must bind both the config and the receipt path",
    )
    return gate


def _validate_artifact_manifest(packet: dict[str, Any]) -> dict[str, Any]:
    manifest = _require_mapping(packet, "artifact_manifest")
    per_arm = frozenset(manifest.get("per_arm_required") or ())
    _require(
        REQUIRED_PER_ARM_ARTIFACTS <= per_arm,
        "artifact_manifest.per_arm_required is missing a required reconstruction artifact",
    )
    paired = frozenset(manifest.get("paired_required") or ())
    _require(
        REQUIRED_PAIRED_ARTIFACTS <= paired,
        "artifact_manifest.paired_required is missing the matched-key or delta artifact",
    )
    provenance = frozenset(manifest.get("per_arm_required_provenance") or ())
    _require(
        {"generating_commit", "config_sha256", "scenario_matrix_hash"} <= provenance,
        "per_arm_required_provenance must bind commit, config digest, and scenario hash",
    )
    _require(
        tuple(manifest.get("matched_key") or ()) == ("planner_key", "scenario_id", "seed"),
        "matched_key must be planner_key/scenario_id/seed",
    )
    for flag in ("raw_episode_jsonl_in_git", "checkpoints_in_git", "videos_in_git"):
        _require(manifest.get(flag) is False, f"artifact_manifest.{flag} must be false")
    return manifest


def _validate_fail_closed(packet: dict[str, Any]) -> dict[str, Any]:
    policy = _require_mapping(packet, "fail_closed_policy")
    statuses = frozenset(policy.get("fail_closed_statuses") or ())
    _require(
        REQUIRED_FAIL_CLOSED_STATUSES <= statuses,
        "fail_closed_statuses missing required weak-row statuses",
    )
    valid = frozenset(policy.get("valid_row_statuses") or ())
    _require(
        not (valid & REQUIRED_FAIL_CLOSED_STATUSES),
        "valid_row_statuses must not overlap the fail-closed statuses",
    )
    _require(
        policy.get("hash_drift_blocks_comparison") is True,
        "hash drift must block the comparison",
    )
    _require(
        policy.get("missing_paired_row_blocks_comparison") is True,
        "a missing paired row must block the comparison",
    )
    return policy


def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Return a compact validation summary for a fail-closed launch packet."""
    _require(packet.get("schema_version") == SCHEMA_VERSION, "unexpected schema_version")
    _require(packet.get("parent_issue") == 5409, "parent_issue must be 5409")
    _require(
        packet.get("no_benchmark_result_claim") is True,
        "no_benchmark_result_claim required",
    )
    commit = str(packet.get("generating_commit", ""))
    _require(len(commit) == 40, "generating_commit must be a full 40-character commit sha")

    claim_boundary = str(packet.get("claim_boundary", ""))
    _require(
        "no SLURM/GPU submission" in claim_boundary,
        "claim_boundary must forbid SLURM/GPU submission",
    )
    _require(
        "does not exist until both campaigns complete" in claim_boundary,
        "claim_boundary must state the paired result does not yet exist",
    )

    matrix = _validate_matrix(packet)
    pair = _validate_pair_validation(packet)
    arms = _validate_arms(packet)
    env = _validate_environment_identity(packet)
    gate = _validate_checkpoint_gate(packet)
    manifest = _validate_artifact_manifest(packet)
    _validate_fail_closed(packet)

    preservation = _require_mapping(packet, "preservation")
    _require(
        bool(preservation.get("durable_destination")),
        "preservation.durable_destination must be declared before dispatch",
    )
    _require(
        preservation.get("raw_artifacts_external") is True,
        "raw artifacts must stay in external durable storage",
    )

    boundary = _require_mapping(packet, "execution_boundary")
    _require(
        boundary.get("full_campaign_in_this_packet") is False,
        "full_campaign_in_this_packet must be false",
    )
    _require(
        boundary.get("submit_slurm_from_this_packet") is False,
        "execution_boundary must not submit SLURM from this packet",
    )
    _require(
        boundary.get("compute_submit_authorized") is False,
        "a packet must never self-authorize compute submission",
    )
    _require(
        boundary.get("status_until_run") == "ready_pending_gate_receipts_and_compute_authorization",
        "status_until_run must name the two remaining launch preconditions",
    )

    return {
        "ok": True,
        "issue": 5409,
        "schema_version": SCHEMA_VERSION,
        "generating_commit": commit,
        "planner_count": matrix.get("planner_count"),
        "scenario_count": matrix.get("scenario_count"),
        "resolved_seeds": list(matrix.get("resolved_seeds") or ()),
        "rows_per_horizon": matrix.get("rows_per_horizon"),
        "rows_total": matrix.get("rows_total"),
        "scenario_matrix_hash": matrix.get("expected_scenario_matrix_hash"),
        "pair_is_valid": pair.get("is_valid"),
        "pair_mismatch_count": pair.get("mismatch_count"),
        "arm_roles": [arm.get("role") for arm in arms],
        "results_dirs": [arm.get("results_dir_template") for arm in arms],
        "checkpoint_gate_status": gate.get("status"),
        "checkpoint_gate_mode": gate.get("mode"),
        "preflight_command_template": env.get("preflight_command_template"),
        "paired_required_artifacts": sorted(manifest.get("paired_required") or ()),
        "durable_destination": preservation.get("durable_destination"),
        "compute_submit_authorized": boundary.get("compute_submit_authorized"),
        "status_until_run": boundary.get("status_until_run"),
    }


def _load_packet(path: Path) -> dict[str, Any]:
    resolved_path = path if path.is_absolute() else REPO_ROOT / path
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PacketError(f"{resolved_path} must contain a YAML mapping")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: load the packet, validate it, and print a status summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet", type=Path, default=DEFAULT_PACKET, help="Path to the launch packet YAML."
    )
    parser.add_argument("--json", action="store_true", help="Emit a compact JSON summary.")
    args = parser.parse_args(argv)

    try:
        packet = _load_packet(args.packet)
    except (FileNotFoundError, PacketError) as exc:
        summary = {"ok": False, "status": MALFORMED, "issue": 5409, "error": str(exc)}
        print(json.dumps(summary) if args.json else f"malformed: {exc}")
        return EXIT_CODES[MALFORMED]

    try:
        summary = validate_packet(packet)
    except PacketError as exc:
        summary = {"ok": False, "status": BLOCKED, "issue": 5409, "error": str(exc)}
        print(json.dumps(summary) if args.json else f"blocked: {exc}")
        return EXIT_CODES[BLOCKED]

    summary["status"] = READY
    if args.json:
        print(json.dumps(summary))
    else:
        print(
            "ready: issue #5409 horizon-ablation launch packet is a valid fail-closed contract "
            f"({summary['planner_count']} planners x {summary['scenario_count']} scenarios x "
            f"{len(summary['resolved_seeds'])} seeds = {summary['rows_per_horizon']} rows per "
            f"horizon, matrix hash {summary['scenario_matrix_hash']}, "
            f"checkpoint gate {summary['checkpoint_gate_status']})"
        )
    return EXIT_CODES[READY]


if __name__ == "__main__":
    raise SystemExit(main())
