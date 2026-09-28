#!/usr/bin/env python3
"""Run and report the preregistered H400 paired doorway slice.

The private operations queue owns Slurm submission and retrieval. This producer
requires an explicit fresh result root and never edits historical release rows.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.benchmark.map_runner.map_runner import _run_map_episode
from robot_sf.benchmark.map_runner.map_runner_jsonl import write_validated_to_handle
from robot_sf.benchmark.schema_validator import load_schema
from robot_sf.benchmark.three_width_doorway_application import (
    DEFAULT_MANIFEST_PATH,
    EMPTY_PLANNER_CONFIG_HASH,
    GOAL_PLANNER_CONFIG_HASH,
    DoorwayPairingSession,
    build_pair_manifest,
    check_pair_receipts,
    load_three_width_manifest,
    non_width_config_sha256,
    run_three_width_preflight,
)
from robot_sf.evidence.writers import review_marker_json, sha256_file, write_json
from robot_sf.training.scenario_loader import load_scenarios

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = _ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
_WIDTHS = (2.2, 2.8, 3.6)
_PLANNERS = ("goal", "social_force")
_SEEDS = (225, 226, 227)
_HORIZON = 400
_CONFIRMATION_HORIZON = 10
_DT = 0.1
_BOOTSTRAP_DRAWS = 10000
_BOOTSTRAP_SEED = 9348
_ENDPOINTS = {
    "success": ("success", "binary"),
    "total_collisions": ("total_collision_count", "runner_count"),
    "pedestrian_collisions": ("ped_collision_count", "runner_count"),
    "obstacle_collisions": ("obstacle_collision_count", "runner_count"),
    "agent_collisions": ("agent_collision_count", "runner_count"),
    "near_misses": ("near_misses", "steps"),
    "minimum_pedestrian_clearance_m": ("min_clearance", "m"),
    "arrival_time_success_only_s": ("time_to_goal_norm_success_only", "s"),
    "path_length_m": ("socnavbench_path_length", "m"),
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=_ROOT, text=True).strip()


def _source_identity() -> str:
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("doorway campaign requires a clean tracked source tree")
    return _git("rev-parse", "HEAD")


def _source_tree_custody(source_sha: str) -> dict[str, Any]:
    """Describe the complete tracked Git tree that produced a campaign bundle.

    Returns:
        The commit tree object, digest of its recursive entry manifest, and entry count.
    """
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("doorway campaign source commit must be a lowercase Git SHA-1")
    tree_oid = _git("rev-parse", f"{source_sha}^{{tree}}")
    entries = subprocess.check_output(
        ["git", "ls-tree", "-r", "-z", "--full-tree", source_sha],
        cwd=_ROOT,
    )
    return {
        "git_tree_oid": tree_oid,
        "entry_manifest_sha256": hashlib.sha256(entries).hexdigest(),
        "tracked_entry_count": sum(bool(entry) for entry in entries.split(b"\0")),
    }


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _row_value(row: dict[str, Any], endpoint: str) -> float | None:
    metric_key, _ = _ENDPOINTS[endpoint]
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    if endpoint == "arrival_time_success_only_s" and metrics.get("success") is not True:
        return None
    value = _finite_number(metrics.get(metric_key))
    return value * _HORIZON * _DT if value is not None and endpoint.endswith("_s") else value


def _paired_interval(differences: list[float], *, binary: bool) -> dict[str, Any]:
    """Bootstrap complete seed blocks with explicit binary degeneracy handling."""
    if not differences:
        return {
            "mean_difference": None,
            "ci95": None,
            "paired_seed_count": 0,
            "status": "no_complete_pairs",
        }
    if binary:
        allowed = {-1.0, 0.0, 1.0}
        if not set(differences).issubset(allowed):
            raise ValueError("binary paired differences must be in {-1, 0, 1}")
        if len(set(differences)) == 1:
            return {
                "mean_difference": float(differences[0]),
                "ci95": None,
                "paired_seed_count": len(differences),
                "status": "degenerate_binary_pair",
            }
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    draws = np.mean(
        values[rng.integers(0, len(values), size=(_BOOTSTRAP_DRAWS, len(values)))], axis=1
    )
    return {
        "mean_difference": float(np.mean(values)),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "paired_seed_count": len(differences),
        "status": "bootstrap",
    }


def _ancillary_telemetry_gaps(metadata: dict[str, Any]) -> list[str]:
    """List unavailable auxiliary traces without treating them as planner fallback."""
    gaps: list[str] = []
    trace = metadata.get("simulation_step_trace")
    reset = trace.get("reset") if isinstance(trace, dict) else None
    if isinstance(reset, dict):
        for key in ("routes", "spawn"):
            value = reset.get(key)
            if isinstance(value, dict) and value.get("status") == "unavailable":
                gaps.append(f"simulation_step_trace.reset.{key}.status=unavailable")

    def _unavailable(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{path}.{key}"
                if key == "status" and item == "unavailable":
                    gaps.append(f"{child}=unavailable")
                else:
                    _unavailable(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                _unavailable(item, f"{path}[{index}]")

    _unavailable(metadata.get("paired_effect_metric_producer"), "paired_effect_metric_producer")
    return sorted(set(gaps))


def _valid_spawn_evidence(row: dict[str, Any] | None) -> bool:
    """Require typed reset clearance and no known spawn-caused contact."""
    if row is None:
        return True
    spawn = row.get("spawn_validity")
    # v1 is retained for historical traces; current benchmark rows use v2.
    return (
        isinstance(spawn, dict)
        and spawn.get("schema_version") in {"spawn_validity.v1", "spawn_validity.v2"}
        and spawn.get("reset_clearance_status") == "available"
        and isinstance(spawn.get("reset_clearance"), dict)
        and spawn["reset_clearance"].get("overlap") is False
        and spawn["reset_clearance"].get("obstacle_overlap") is False
        and spawn["reset_clearance"].get("pedestrian_overlap") is False
        and spawn.get("reset_clearance_error") is None
        and spawn.get("reset_overlap") is False
        and spawn.get("invalid_run") is False
        and spawn.get("invalid_reason") is None
        and isinstance(spawn.get("respawn_overlap_events"), list)
        and isinstance(spawn.get("respawn_overlap_collisions"), list)
        and not spawn["respawn_overlap_collisions"]
    )


def _command_execution_mode(metadata: dict[str, Any]) -> str | None:
    kinematics = metadata.get("planner_kinematics")
    if not isinstance(kinematics, dict):
        return None
    value = kinematics.get("execution_mode")
    return value if isinstance(value, str) else None


def _valid_action_trace(trace: Any, row: dict[str, Any] | None) -> bool:
    """Require one finite selected and applied planner action per executed step."""
    if (
        not isinstance(trace, dict)
        or trace.get("schema_version") != "simulation-step-trace.v1"
        or not isinstance(trace.get("dt"), (int, float))
        or isinstance(trace["dt"], bool)
        or not math.isclose(trace["dt"], _DT, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        return False
    steps = trace.get("steps")
    if (
        row is None
        or not isinstance(row.get("steps"), int)
        or isinstance(row["steps"], bool)
        or row["steps"] < 1
        or not isinstance(steps, list)
        or len(steps) != row["steps"]
    ):
        return False
    for index, item in enumerate(steps):
        if not isinstance(item, dict) or item.get("step") != index:
            return False
        time_s = item.get("time_s")
        if (
            not isinstance(time_s, (int, float))
            or isinstance(time_s, bool)
            or not math.isclose(time_s, (index + 1) * _DT, rel_tol=0.0, abs_tol=1.0e-9)
        ):
            return False
        planner = item.get("planner")
        if not isinstance(planner, dict) or planner.get("event") != "step":
            return False
        for key in ("selected_action", "applied_environment_action"):
            action = planner.get(key)
            if not isinstance(action, dict) or any(
                not isinstance(action.get(field), (int, float))
                or isinstance(action.get(field), bool)
                or not math.isfinite(action[field])
                for field in ("linear_velocity", "angular_velocity")
            ):
                return False
    return True


def _execution_runtime_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Select execution-bearing fields for canonical fallback scanning."""
    runtime_fields = {
        "status",
        "fallback_used",
        "fallback_or_degraded",
        "degraded",
        "planner_kinematics",
        "planner_diagnostics",
        "planner_runtime",
        "planner_decision_trace",
        "distributional_disruption",
        "guard_stats",
        "shield_stats",
        "native_command",
    }
    runtime_metadata = {
        key: value for key, value in metadata.items() if key in runtime_fields or "fallback" in key
    }
    trace = metadata.get("simulation_step_trace")
    if isinstance(trace, dict):
        runtime_metadata["simulation_step_trace"] = {"steps": trace.get("steps")}
    return runtime_metadata


def _baseline_execution_reasons(
    metadata: dict[str, Any], row: dict[str, Any] | None, expected_algorithm: str | None
) -> list[str]:
    """Bind each baseline to its declared command route, without requiring absent row axes."""
    expected = {
        "goal": ("native", False, "none"),
        "social_force": ("adapter", True, "SocialForcePlannerAdapter"),
    }.get(expected_algorithm)
    kinematics = metadata.get("planner_kinematics")
    reasons = []
    if (
        expected is None
        or not isinstance(kinematics, dict)
        or (
            kinematics.get("execution_mode"),
            kinematics.get("adapter_active"),
            kinematics.get("adapter_name"),
        )
        != expected
    ):
        reasons.append("planner_kinematics_mode_or_adapter_mismatch")
    if (
        expected_algorithm == "social_force"
        and isinstance(kinematics, dict)
        and (kinematics.get("projection_documented") is not True)
    ):
        reasons.append("social_force_adapter_projection_unverified")
    if metadata.get("canonical_algorithm") != expected_algorithm:
        reasons.append("canonical_planner_identity_mismatch")
    if (
        row is not None
        and row.get("execution_mode") is not None
        and (expected is None or row["execution_mode"] != expected[0])
    ):
        reasons.append(f"unexpected_execution_mode:{row.get('execution_mode')!r}")
    if (
        row is not None
        and row.get("readiness_status") is not None
        and (row["readiness_status"] != "native")
    ):
        reasons.append(f"non_native_readiness_status:{row.get('readiness_status')!r}")
    return reasons


def _trace_exclusion_reasons(
    metadata: dict[str, Any],
    *,
    row: dict[str, Any] | None = None,
    expected_algorithm: str | None = None,
) -> list[str]:
    """Flag invalid baseline execution and actual runtime fallback markers."""
    trace = metadata.get("simulation_step_trace")
    planner_trace = metadata.get("planner_decision_trace")
    reasons = []
    if not _valid_action_trace(trace, row):
        reasons.append("missing_or_invalid_simulation_action_trace")
    if not isinstance(planner_trace, dict) or not isinstance(planner_trace.get("steps"), list):
        reasons.append("missing_planner_decision_trace")
    else:
        # Goal and Social Force expose no specialized internal decisions. Their
        # empty arrays are expected; simulation_step_trace is the action record.
        for step in planner_trace["steps"]:
            if isinstance(step, dict) and (
                step.get("fallback_used") is True
                or str(step.get("planner_mode", "")).lower() in {"fallback", "degraded"}
                or str(step.get("execution_mode", "")).lower() in {"fallback", "degraded"}
            ):
                reasons.append("fallback_or_degraded_planner_step")
                break
    reasons.extend(_baseline_execution_reasons(metadata, row, expected_algorithm))
    if not _valid_spawn_evidence(row):
        reasons.append("invalid_or_unknown_spawn_validity")
    # Scan execution-bearing fields only. Reset routes/spawn report whether
    # sampler objects were retained, and paired-effect fields describe separate
    # wrapper endpoints; neither reports planner fallback. Their unavailable
    # statuses are retained separately in ancillary_telemetry_gaps.
    runtime_payload = {
        "row": {
            "execution_mode": row.get("execution_mode") if row is not None else None,
            "readiness_status": row.get("readiness_status") if row is not None else None,
        },
        "algorithm_metadata": _execution_runtime_metadata(metadata),
    }
    if row is not None and "planner_runtime" in row:
        runtime_payload["planner_runtime"] = row["planner_runtime"]
    runtime_marker = runtime_fallback_or_degraded_marker(
        runtime_payload,
        expected_algorithm=expected_algorithm,
        algorithm_metadata=metadata,
    )
    if runtime_marker is not None:
        marker_path, marker_value = runtime_marker
        reasons.append(f"fallback_or_degraded_runtime:{marker_path}={marker_value}")
    return reasons


def _row_inventory_item(
    identity: tuple[str, int, float],
    row: dict[str, Any],
    cell: dict[str, Any],
    expected_receipt: dict[str, Any],
    *,
    source_sha: str | None,
) -> dict[str, Any]:
    """Verify one row's immutable identity and classify its execution evidence."""
    planner, seed, width = identity
    metadata = row.get("algorithm_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"algorithm metadata missing: {identity}")
    expected_config_hash = {
        "goal": GOAL_PLANNER_CONFIG_HASH,
        "social_force": EMPTY_PLANNER_CONFIG_HASH,
    }[planner]
    if metadata.get("config_hash") != expected_config_hash:
        raise ValueError(
            f"planner config identity mismatch: {identity}; "
            f"expected {expected_config_hash}, observed {metadata.get('config_hash')!r}"
        )
    observed = metadata.get("doorway_pair_receipt")
    if not isinstance(observed, dict) or any(
        observed.get(field) != expected_receipt.get(field)
        for field in (
            "map_sha256",
            "initial_actor_state_sha256",
            "external_rng_state_sha256",
            "non_width_config_sha256",
        )
    ):
        raise ValueError(f"paired reset custody mismatch: {identity}")
    if (row.get("algo"), row.get("seed"), row.get("scenario_id"), row.get("horizon")) != (
        planner,
        seed,
        cell["scenario_id"],
        _HORIZON,
    ):
        raise ValueError(f"episode identity or H400 mismatch: {identity}")
    if source_sha is not None and row.get("git_hash") != source_sha:
        raise ValueError(f"episode source commit differs: {identity}")
    trace = metadata.get("simulation_step_trace")
    reasons = _trace_exclusion_reasons(metadata, row=row, expected_algorithm=planner)
    if metadata.get("status") != "ok":
        reasons.append(f"planner_status={metadata.get('status')}")
    if any(metadata.get(key) is True for key in ("fallback_used", "degraded")):
        reasons.append("fallback_or_degraded_planner")
    if row.get("status") not in {"success", "collision", "failure"}:
        reasons.append("unknown_episode_status")
    if isinstance(row.get("integrity"), dict) and row["integrity"].get("contradictions"):
        reasons.append("episode_integrity_contradiction")
    return {
        "planner": planner,
        "seed": seed,
        "gap_width_m": width,
        "episode_id": row.get("episode_id"),
        "outcome": row.get("outcome"),
        "termination_reason": row.get("termination_reason"),
        "steps": row.get("steps"),
        "evidence_status": "native" if not reasons else "excluded",
        "exclusion_reasons": reasons,
        "command_execution_mode": _command_execution_mode(metadata),
        "ancillary_telemetry_gaps": _ancillary_telemetry_gaps(metadata),
        "trace_path": f"episodes.jsonl:line:{cell['line_number']}:algorithm_metadata.simulation_step_trace.steps",
        "trace_steps": len(trace.get("steps", [])) if isinstance(trace, dict) else 0,
    }


def _require_h1_preflight(preflight: dict[str, Any]) -> None:
    """Require bounded H1 geometry/binding evidence, retaining its oracle diagnostic."""
    checks = preflight.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("doorway confirmation preflight checks are unavailable")
    expected_fallbacks = checks.get("oracle_expected_fallbacks")
    if not isinstance(expected_fallbacks, list):
        raise ValueError("doorway confirmation fallback admission check is unavailable")
    if expected_fallbacks not in (
        [],
        [
            {
                "variant_id": "gap_3p60__depth_1p00",
                "reason": "expected_distributional_metric_unavailable",
                "marker": (
                    "metrics.distributional_disruption.missing_data."
                    "slow_speed_tier.status=unavailable"
                ),
            }
        ],
    ):
        raise ValueError("doorway H1 oracle has an undeclared fallback or degraded diagnostic")
    if preflight.get("go") is not True:
        raise ValueError("doorway geometry/oracle preflight did not admit policy execution")
    for field in (
        "baseline_passes",
        "all_widths_positive_clearance",
        "oracle_available_for_every_variant",
        "oracle_required_checks_known",
        "h1_execution_binding_ready",
        "planner_records_are_not_run",
        "no_campaign_evidence",
    ):
        if checks.get(field) is not True:
            raise ValueError(f"doorway H1 preflight lacks required check: {field}")
    if checks.get("variant_count") != 3:
        raise ValueError("doorway H1 preflight must retain three widths")


def _confirmation_row_blockers(
    row: dict[str, Any],
    cell: dict[str, Any],
    receipt: dict[str, Any] | None,
    *,
    source_sha: str,
) -> list[str]:
    """Reject any short probe without native planner and exact reset custody."""
    planner, seed = cell["planner"], cell["seed"]
    metadata = row.get("algorithm_metadata")
    reasons = []
    if not isinstance(metadata, dict):
        reasons.append("missing_algorithm_metadata")
        metadata = {}
    expected_hash = {"goal": GOAL_PLANNER_CONFIG_HASH, "social_force": EMPTY_PLANNER_CONFIG_HASH}[
        planner
    ]
    if metadata.get("config_hash") != expected_hash:
        reasons.append("planner_config_hash_mismatch")
    if (row.get("algo"), row.get("seed"), row.get("scenario_id"), row.get("horizon")) != (
        planner,
        seed,
        cell.get("scenario_id"),
        _CONFIRMATION_HORIZON,
    ):
        reasons.append("episode_identity_or_horizon_mismatch")
    if row.get("git_hash") != source_sha:
        reasons.append("source_commit_mismatch")
    observed = metadata.get("doorway_pair_receipt")
    if (
        not isinstance(receipt, dict)
        or not isinstance(observed, dict)
        or any(
            observed.get(key) != receipt.get(key)
            for key in (
                "map_sha256",
                "initial_actor_state_sha256",
                "external_rng_state_sha256",
                "non_width_config_sha256",
            )
        )
    ):
        reasons.append("paired_reset_receipt_mismatch")
    if not isinstance(row.get("steps"), int) or not 1 <= row["steps"] <= _CONFIRMATION_HORIZON:
        reasons.append("invalid_confirmation_step_count")
    reasons.extend(_trace_exclusion_reasons(metadata, row=row, expected_algorithm=planner))
    if metadata.get("status") != "ok":
        reasons.append("planner_status_not_ok")
    if row.get("status") not in {"success", "collision", "failure"}:
        reasons.append("unknown_episode_status")
    if isinstance(row.get("integrity"), dict) and row["integrity"].get("contradictions"):
        reasons.append("episode_integrity_contradiction")
    return sorted(set(reasons))


def assess_confirmation_rows(
    rows: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    pair_manifest: dict[str, Any],
    *,
    source_sha: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Classify a separate actor-present H10 probe before H400 dispatch."""
    expected = {(p, s, w) for p in _PLANNERS for s in _SEEDS for w in _WIDTHS}
    identities = [(c.get("planner"), c.get("seed"), c.get("gap_width_m")) for c in cells]
    if (
        len(rows) != 18
        or len(cells) != 18
        or set(identities) != expected
        or len(set(identities)) != 18
    ):
        raise ValueError("confirmation probe requires all 18 frozen identities")
    pair_errors = check_pair_receipts(pair_manifest)
    if pair_manifest.get("manifest_sha256") != manifest_sha256:
        raise ValueError("confirmation pair manifest checksum identity mismatch")
    receipts = {
        (pair["planner"], pair["seed"], cell["gap_width_m"]): cell
        for pair in pair_manifest["pairs"]
        for cell in pair["cells"]
    }
    inventory = []
    for row, cell, identity in zip(rows, cells, identities, strict=True):
        planner, seed, width = identity
        metadata = row.get("algorithm_metadata")
        reasons = _confirmation_row_blockers(
            row, cell, receipts.get(identity), source_sha=source_sha
        )
        inventory.append(
            {
                "planner": planner,
                "seed": seed,
                "gap_width_m": width,
                "status": "eligible" if not reasons else "diagnostic",
                "blockers": sorted(set(reasons)),
                "command_execution_mode": _command_execution_mode(metadata)
                if isinstance(metadata, dict)
                else None,
                "ancillary_telemetry_gaps": _ancillary_telemetry_gaps(
                    metadata if isinstance(metadata, dict) else {}
                ),
            }
        )
    return {
        "schema_version": "issue_9348_confirmation_preflight.v1",
        "claim_boundary": "actor-present H10 probe only; no H400 outcome or width effect",
        "source_commit": source_sha,
        "application_manifest_sha256": manifest_sha256,
        "planner_config_hashes": {
            "goal": GOAL_PLANNER_CONFIG_HASH,
            "social_force": EMPTY_PLANNER_CONFIG_HASH,
        },
        "probe_horizon_steps": _CONFIRMATION_HORIZON,
        "planned_rows": 18,
        "pair_errors": pair_errors,
        "rows": inventory,
        "admit_h400": not pair_errors and all(item["status"] == "eligible" for item in inventory),
    }


def _require_confirmation_preflight(
    preflight: dict[str, Any], confirmation: dict[str, Any]
) -> None:
    """Keep actor-free oracle findings outside a strict actor-present gate."""
    _require_h1_preflight(preflight)
    if confirmation.get("schema_version") != "issue_9348_confirmation_preflight.v1":
        raise ValueError("doorway actor-present confirmation report is unavailable")
    if confirmation.get("admit_h400") is not True:
        raise ValueError(
            "doorway actor-present confirmation has fallback, degraded, or incomplete rows"
        )


def _run_actor_present_confirmation(
    manifest: dict[str, Any],
    assets: list[dict[str, Any]],
    output_root: Path,
    *,
    source_sha: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Run and preserve 18 short planner probes with independent pair receipts."""
    session = DoorwayPairingSession()
    schema = load_schema(_SCHEMA)
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    raw_path = output_root / "confirmation_episodes.jsonl"
    configs = manifest["_resolved"]["planner_configs"]
    hashes = manifest["_resolved"]["planner_config_hashes"]
    with raw_path.open("x", encoding="utf-8") as handle:
        for planner in _PLANNERS:
            for seed in _SEEDS:
                for asset in assets:
                    scenario_path = Path(asset["scenario_path"])
                    scenario = load_scenarios(scenario_path)[0]
                    receipt_hook = session.hook(
                        planner=planner,
                        seed=seed,
                        map_sha256=asset["map_sha256"],
                        non_width_config_sha256=non_width_config_sha256(
                            scenario, planner=planner, planner_config_hash=hashes[planner]
                        ),
                    )
                    row = _run_map_episode(
                        scenario,
                        seed,
                        horizon=_CONFIRMATION_HORIZON,
                        dt=_DT,
                        record_forces=True,
                        snqi_weights=None,
                        snqi_baseline=None,
                        algo=planner,
                        scenario_path=scenario_path,
                        algo_config=dict(configs[planner]),
                        algo_config_path=None,
                        record_planner_decision_trace=True,
                        record_simulation_step_trace=True,
                        pair_reset_hook=receipt_hook,
                    )
                    serialized = io.StringIO()
                    write_validated_to_handle(serialized, schema, row)
                    line = serialized.getvalue()
                    handle.write(line)
                    handle.flush()
                    rows.append(row)
                    cells.append(
                        {
                            "planner": planner,
                            "seed": seed,
                            "gap_width_m": asset["gap_width_m"],
                            "variant_id": asset["variant_id"],
                            "scenario_id": scenario["name"],
                            "scenario_sha256": asset["scenario_sha256"],
                            "map_sha256": asset["map_sha256"],
                            "line_number": len(rows),
                            "line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                        }
                    )
    pairs = session.fill_pair_manifest(build_pair_manifest(assets, _SEEDS, manifest_sha256))
    write_json(output_root / "confirmation_pair_manifest.json", pairs)
    report = assess_confirmation_rows(
        rows, cells, pairs, source_sha=source_sha, manifest_sha256=manifest_sha256
    )
    report["cells"] = cells
    report["episodes_jsonl_sha256"] = sha256_file(raw_path)
    write_json(output_root / "confirmation_preflight.json", report)
    return report


def _contrast_endpoint(
    by_identity: dict[tuple[str, int, float], tuple[dict[str, Any], dict[str, Any]]],
    valid: set[tuple[str, int, float]],
    planner: str,
    low: float,
    high: float,
    endpoint: str,
) -> dict[str, Any]:
    """Use only complete valid seed pairs for one endpoint."""
    differences = []
    paired_seed_ids = []
    excluded = []
    for seed in _SEEDS:
        low_key, high_key = (planner, seed, low), (planner, seed, high)
        left = _row_value(by_identity[low_key][0], endpoint)
        right = _row_value(by_identity[high_key][0], endpoint)
        if low_key not in valid or high_key not in valid or left is None or right is None:
            excluded.append(seed)
        else:
            differences.append(right - left)
            paired_seed_ids.append(seed)
    binary = endpoint == "success"
    estimate = _paired_interval(differences, binary=binary)
    return {
        "unit": _ENDPOINTS[endpoint][1],
        "planned_pair_count": len(_SEEDS),
        "denominator_pairs": len(differences),
        "paired_seed_ids": paired_seed_ids,
        "excluded_seed_ids": excluded,
        "raw_paired_differences": differences,
        "binary_pair_status": estimate["status"] if binary else "not_binary",
        "estimate": estimate,
        "arrival_time_condition": "both_widths_successful"
        if endpoint.startswith("arrival")
        else None,
    }


def _cell_denominators(
    by_identity: dict[tuple[str, int, float], tuple[dict[str, Any], dict[str, Any]]],
    valid: set[tuple[str, int, float]],
) -> list[dict[str, Any]]:
    """Report planned, native and endpoint-specific counts for each cell."""
    cells = []
    for planner in _PLANNERS:
        for width in _WIDTHS:
            native_seed_ids = [seed for seed in _SEEDS if (planner, seed, width) in valid]
            endpoint_seed_ids = {
                endpoint: [
                    seed
                    for seed in _SEEDS
                    if (planner, seed, width) in valid
                    and _row_value(by_identity[(planner, seed, width)][0], endpoint) is not None
                ]
                for endpoint in _ENDPOINTS
            }
            cells.append(
                {
                    "planner": planner,
                    "gap_width_m": width,
                    "planned_seed_count": len(_SEEDS),
                    "native_row_count": len(native_seed_ids),
                    "native_seed_ids": native_seed_ids,
                    "endpoint_denominators": {
                        endpoint: len(seed_ids) for endpoint, seed_ids in endpoint_seed_ids.items()
                    },
                    "endpoint_seed_ids": endpoint_seed_ids,
                }
            )
    return cells


def analyze_rows(
    rows: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    pair_manifest: dict[str, Any],
    *,
    source_sha: str | None = None,
) -> dict[str, Any]:
    """Fail closed on identity/custody and report endpoint-specific missingness.

    Returns:
        Complete descriptive and paired-uncertainty report.
    """
    if len(rows) != 18 or len(cells) != 18 or check_pair_receipts(pair_manifest):
        raise ValueError("doorway report requires 18 rows and six verified reset pairs")
    expected = {(p, s, w) for p in _PLANNERS for s in _SEEDS for w in _WIDTHS}
    identities = [(c["planner"], c["seed"], c["gap_width_m"]) for c in cells]
    if len(set(identities)) != 18 or set(identities) != expected:
        raise ValueError("doorway cell roster differs from the frozen 18 identities")
    by_identity = dict(zip(identities, zip(rows, cells, strict=True), strict=True))
    pair_receipts = {
        (p["planner"], p["seed"], c["gap_width_m"]): c
        for p in pair_manifest["pairs"]
        for c in p["cells"]
    }
    row_inventory = [
        _row_inventory_item(identity, row, cell, pair_receipts[identity], source_sha=source_sha)
        for identity, (row, cell) in by_identity.items()
    ]
    valid = {
        (item["planner"], item["seed"], item["gap_width_m"])
        for item in row_inventory
        if item["evidence_status"] == "native"
    }
    failure_cases = [
        item
        for item in row_inventory
        if item["evidence_status"] != "native" or item["termination_reason"] != "success"
    ]
    excluded_pair_ids = sorted(
        {
            f"{item['planner']}_pair_{item['seed']:05d}"
            for item in row_inventory
            if item["evidence_status"] != "native"
        }
    )
    contrasts = [
        {
            "planner": planner,
            "low_width_m": low,
            "high_width_m": high,
            "endpoints": {
                endpoint: _contrast_endpoint(by_identity, valid, planner, low, high, endpoint)
                for endpoint in _ENDPOINTS
            },
        }
        for planner in _PLANNERS
        for low, high in ((2.2, 2.8), (2.8, 3.6), (2.2, 3.6))
    ]
    cell_denominators = _cell_denominators(by_identity, valid)
    return {
        "schema_version": "issue_9348_three_width_report.v1",
        "claim_boundary": "simulator-internal paired width contrasts; three seeds per planner",
        "evidence_status": "candidate" if len(valid) == 18 else "diagnostic_incomplete",
        "planned_rows": 18,
        "observed_rows": len(rows),
        "native_rows": len(valid),
        "native_rows_semantics": (
            "eligible baseline rows: goal native commands or declared Social Force adapter; "
            "not a claim that both planners use native command mode"
        ),
        "excluded_rows": 18 - len(valid),
        "excluded_pair_ids": excluded_pair_ids,
        "failure_cases": failure_cases,
        "bootstrap": {
            "unit": "paired_seed",
            "draws": _BOOTSTRAP_DRAWS,
            "seed": _BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "row_inventory": row_inventory,
        "cell_denominators": cell_denominators,
        "contrasts": contrasts,
        "pedestrian_delay_or_impairment": {
            "status": "unavailable_without_matched_no_robot_control_trace",
            "denominator_pedestrians": 0,
            "value": None,
        },
        "mechanism_trace_policy": "Every row has a trace reference; interpret only reviewed recorded mechanisms.",
        "confirmation_admission": "pending_independent_artifact_and_mechanism_review",
        "limitations": [
            "Three seeds per planner; uncertainty intervals are descriptive.",
            "Arrival time is success-conditional; failures are censored at termination.",
            "Conservative grid A* has no route at 2.2 and 2.8 m; executed policies do not use it.",
        ],
    }


def _write_checksums(root: Path) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p != root / "SHA256SUMS")
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
    )


def _verify_generated_assets(root: Path, cells: list[dict[str, Any]]) -> None:
    """Bind each reported cell to the actual generated scenario and map bytes."""
    preflight_assets = {
        item["variant_id"]: item["assets"] for item in _json(root / "preflight.json")["variants"]
    }
    for cell in cells:
        variant_id = cell["variant_id"]
        if Path(variant_id).name != variant_id or variant_id not in preflight_assets:
            raise ValueError("doorway variant identity or path is invalid")
        asset = preflight_assets[variant_id]
        if (
            sha256_file(root / "assets" / variant_id / "variant.svg") != cell["map_sha256"]
            or sha256_file(root / "assets" / variant_id / "scenario.yaml")
            != cell["scenario_sha256"]
            or asset["map_sha256"] != cell["map_sha256"]
            or asset["scenario_sha256"] != cell["scenario_sha256"]
        ):
            raise ValueError("doorway generated asset digest differs from row custody")


def _verify_confirmation_files(
    root: Path, *, source_sha: str, manifest_sha256: str
) -> dict[str, Any]:
    """Rebuild short-probe admission from raw rows and sealed pair receipts."""
    report = _json(root / "confirmation_preflight.json")
    cells = report.get("cells")
    if not isinstance(cells, list) or len(cells) != 18:
        raise ValueError("doorway confirmation cell custody is incomplete")
    _verify_generated_assets(root, cells)
    raw_path = root / "confirmation_episodes.jsonl"
    if sha256_file(raw_path) != report.get("episodes_jsonl_sha256"):
        raise ValueError("doorway confirmation raw episode checksum mismatch")
    lines = raw_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) != 18 or any(
        cell.get("line_number") != number
        or hashlib.sha256(line.encode()).hexdigest() != cell.get("line_sha256")
        for number, (line, cell) in enumerate(zip(lines, cells, strict=True), start=1)
    ):
        raise ValueError("doorway confirmation line identity or digest mismatch")
    rebuilt = assess_confirmation_rows(
        [json.loads(line) for line in lines],
        cells,
        _json(root / "confirmation_pair_manifest.json"),
        source_sha=source_sha,
        manifest_sha256=manifest_sha256,
    )
    rebuilt["cells"] = cells
    rebuilt["episodes_jsonl_sha256"] = sha256_file(raw_path)
    if report != {"review_marker": review_marker_json(), **rebuilt}:
        raise ValueError("doorway confirmation report differs from sealed raw probes")
    return rebuilt


def verify_campaign_bundle(root: Path) -> dict[str, Any]:
    """Read back every produced file and row against sealed SHA-256 receipts.

    Returns:
        The unchanged paired report after all custody checks pass.
    """
    root = root.resolve()
    checksum_path = root / "SHA256SUMS"
    listed: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        path = Path(relative)
        if (
            not separator
            or not relative
            or path.is_absolute()
            or ".." in path.parts
            or relative in listed
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("malformed or duplicate bundle checksum path")
        listed[relative] = digest
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("doorway bundle must not contain symlinks")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if set(listed) != actual or any(
        sha256_file(root / relative) != digest for relative, digest in listed.items()
    ):
        raise ValueError("doorway bundle file checksum or full-tree coverage mismatch")
    run_manifest = _json(root / "run_manifest.json")
    source_sha = run_manifest.get("source_commit")
    if not isinstance(source_sha, str) or run_manifest.get(
        "source_tree_custody"
    ) != _source_tree_custody(source_sha):
        raise ValueError("doorway source tree custody differs from the declared source commit")
    if (
        sha256_file(root / "inputs/application_manifest.yaml")
        != run_manifest.get("application_manifest_sha256")
        or sha256_file(root / "inputs/episode.schema.v1.json")
        != run_manifest.get("episode_schema_sha256")
        or run_manifest.get("planner_config_hash")
        != {"goal": GOAL_PLANNER_CONFIG_HASH, "social_force": EMPTY_PLANNER_CONFIG_HASH}
        or (root / "inputs/social_force.yaml").exists()
    ):
        raise ValueError("doorway copied scientific input digest mismatch")
    confirmation = _verify_confirmation_files(
        root,
        source_sha=run_manifest["source_commit"],
        manifest_sha256=run_manifest["application_manifest_sha256"],
    )
    _require_confirmation_preflight(_json(root / "preflight.json"), confirmation)
    pair_manifest = _json(root / "pair_manifest.json")
    raw_path = root / "episodes.jsonl"
    if sha256_file(raw_path) != run_manifest.get("episodes_jsonl_sha256"):
        raise ValueError("episodes JSONL digest mismatch")
    lines = raw_path.read_text(encoding="utf-8").splitlines(keepends=True)
    cells = run_manifest["cells"]
    _verify_generated_assets(root, cells)
    if (
        len(lines) != 18
        or len(cells) != 18
        or any(
            cell.get("line_number") != number
            or hashlib.sha256(line.encode()).hexdigest() != cell.get("line_sha256")
            for number, (line, cell) in enumerate(zip(lines, cells, strict=True), start=1)
        )
    ):
        raise ValueError("doorway JSONL line identity or digest mismatch")
    rows = [json.loads(line) for line in lines]
    rebuilt = analyze_rows(rows, cells, pair_manifest, source_sha=run_manifest["source_commit"])
    report = _json(root / "report.json")
    if report != {"review_marker": review_marker_json(), **rebuilt}:
        raise ValueError("doorway report differs from sealed raw episodes")
    return rebuilt


def run_campaign(manifest_path: Path, output_root: Path) -> Path:
    """Execute H400 serially and seal all 18 raw rows and the paired report.

    Returns:
        Path to the completed result root.
    """
    source_sha = _source_identity()
    source_tree_custody = _source_tree_custody(source_sha)
    manifest_path = manifest_path.resolve()
    manifest = load_three_width_manifest(manifest_path)
    output_root = output_root.resolve()
    if output_root == _ROOT or _ROOT in output_root.parents:
        raise ValueError("doorway H400 output must use an explicit external durable root")
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        (output_root / "inputs").mkdir()
        shutil.copy2(manifest_path, output_root / "inputs/application_manifest.yaml")
        preflight = run_three_width_preflight(manifest_path, output_dir=output_root / "assets")
        write_json(output_root / "preflight.json", preflight)
        _require_h1_preflight(preflight)
        assets = [
            record["assets"]
            | {"variant_id": record["variant_id"], "gap_width_m": record["geometry"]["gap_width_m"]}
            for record in preflight["variants"]
        ]
        session = DoorwayPairingSession()
        planner_configs = manifest["_resolved"]["planner_configs"]
        planner_config_hashes = manifest["_resolved"]["planner_config_hashes"]
        shutil.copy2(_SCHEMA, output_root / "inputs/episode.schema.v1.json")
        confirmation = _run_actor_present_confirmation(
            manifest,
            assets,
            output_root,
            source_sha=source_sha,
            manifest_sha256=sha256_file(manifest_path),
        )
        _require_confirmation_preflight(preflight, confirmation)
        schema = load_schema(_SCHEMA)
        rows: list[dict[str, Any]] = []
        cells: list[dict[str, Any]] = []
        raw_path = output_root / "episodes.jsonl"
        with raw_path.open("x", encoding="utf-8") as handle:
            for planner in _PLANNERS:
                planner_config = dict(planner_configs[planner])
                planner_config_hash = planner_config_hashes[planner]
                for seed in _SEEDS:
                    for asset in assets:
                        scenario_path = Path(asset["scenario_path"])
                        scenario = load_scenarios(scenario_path)[0]
                        receipt_hook = session.hook(
                            planner=planner,
                            seed=seed,
                            map_sha256=asset["map_sha256"],
                            non_width_config_sha256=non_width_config_sha256(
                                scenario, planner=planner, planner_config_hash=planner_config_hash
                            ),
                        )
                        row = _run_map_episode(
                            scenario,
                            seed,
                            horizon=_HORIZON,
                            dt=_DT,
                            record_forces=True,
                            snqi_weights=None,
                            snqi_baseline=None,
                            algo=planner,
                            scenario_path=scenario_path,
                            algo_config=planner_config,
                            algo_config_path=None,
                            record_planner_decision_trace=True,
                            record_simulation_step_trace=True,
                            pair_reset_hook=receipt_hook,
                        )
                        serialized = io.StringIO()
                        write_validated_to_handle(serialized, schema, row)
                        line = serialized.getvalue()
                        handle.write(line)
                        handle.flush()
                        rows.append(row)
                        cells.append(
                            {
                                "planner": planner,
                                "seed": seed,
                                "gap_width_m": asset["gap_width_m"],
                                "variant_id": asset["variant_id"],
                                "scenario_id": scenario["name"],
                                "scenario_sha256": asset["scenario_sha256"],
                                "map_sha256": asset["map_sha256"],
                                "line_number": len(rows),
                                "line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                            }
                        )
        paired = session.fill_pair_manifest(
            build_pair_manifest(assets, _SEEDS, sha256_file(manifest_path))
        )
        write_json(output_root / "pair_manifest.json", paired)
        report = analyze_rows(rows, cells, paired, source_sha=source_sha)
        write_json(output_root / "report.json", report)
        write_json(
            output_root / "run_manifest.json",
            {
                "schema_version": "issue_9348_three_width_run.v1",
                "source_commit": source_sha,
                "source_tree_custody": source_tree_custody,
                "application_manifest_path": manifest_path.as_posix(),
                "application_manifest_sha256": sha256_file(manifest_path),
                "planner_config_hash": dict(planner_config_hashes),
                "episode_schema_sha256": sha256_file(_SCHEMA),
                "horizon_steps": _HORIZON,
                "dt_s": _DT,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "episodes_jsonl_sha256": sha256_file(raw_path),
                "cells": cells,
            },
        )
        _write_checksums(output_root)
    finally:
        error = sys.exception()
        if error is not None:
            write_json(
                output_root / "run_failure.json",
                {
                    "schema_version": "issue_9348_run_failure.v1",
                    "source_commit": source_sha,
                    "source_tree_custody": source_tree_custody,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            _write_checksums(output_root)
    return output_root


def main() -> int:
    """Run a fresh H400 result root or verify a retrieved sealed bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "verify"), required=True)
    parser.add_argument("--manifest", type=Path, default=_ROOT / DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run":
        root = run_campaign(args.manifest, args.output_root)
        print(f"issue #9348 H400: 18 cells sealed at {root}")
    else:
        report = verify_campaign_bundle(args.output_root)
        print(f"issue #9348 bundle verified: rows={report['observed_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
