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
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.map_runner.map_runner import _run_map_episode
from robot_sf.benchmark.map_runner.map_runner_jsonl import write_validated_to_handle
from robot_sf.benchmark.schema_validator import load_schema
from robot_sf.benchmark.three_width_doorway_application import (
    DEFAULT_MANIFEST_PATH,
    DoorwayPairingSession,
    build_pair_manifest,
    check_pair_receipts,
    load_three_width_manifest,
    non_width_config_sha256,
    run_three_width_preflight,
)
from robot_sf.evidence.writers import sha256_file, write_json
from robot_sf.training.scenario_loader import load_scenarios

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = _ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
_WIDTHS = (2.2, 2.8, 3.6)
_PLANNERS = ("goal", "social_force")
_SEEDS = (225, 226, 227)
_HORIZON = 400
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


def _paired_interval(differences: list[float]) -> dict[str, Any]:
    """Bootstrap complete seed blocks; time steps never become independent units."""
    if not differences:
        return {"mean_difference": None, "ci95": None, "paired_seed_count": 0}
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    draws = np.mean(
        values[rng.integers(0, len(values), size=(_BOOTSTRAP_DRAWS, len(values)))], axis=1
    )
    return {
        "mean_difference": float(np.mean(values)),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "paired_seed_count": len(differences),
    }


def _trace_exclusion_reasons(metadata: dict[str, Any]) -> list[str]:
    """Flag missing step traces without promoting an incomplete row."""
    trace = metadata.get("simulation_step_trace")
    planner_trace = metadata.get("planner_decision_trace")
    reasons = []
    if (
        not isinstance(trace, dict)
        or not isinstance(trace.get("steps"), list)
        or not trace["steps"]
    ):
        reasons.append("missing_simulation_step_trace")
    if not isinstance(planner_trace, dict) or not isinstance(planner_trace.get("steps"), list):
        reasons.append("missing_planner_decision_trace")
    else:
        for step in planner_trace["steps"]:
            if isinstance(step, dict) and (
                step.get("fallback_used") is True
                or str(step.get("planner_mode", "")).lower() in {"fallback", "degraded"}
                or str(step.get("execution_mode", "")).lower() in {"fallback", "degraded"}
            ):
                reasons.append("fallback_or_degraded_planner_step")
                break
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
    reasons = _trace_exclusion_reasons(metadata)
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
        "trace_path": f"episodes.jsonl:line:{cell['line_number']}:algorithm_metadata.simulation_step_trace.steps",
        "trace_steps": len(trace.get("steps", [])) if isinstance(trace, dict) else 0,
    }


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
    excluded = []
    for seed in _SEEDS:
        low_key, high_key = (planner, seed, low), (planner, seed, high)
        left = _row_value(by_identity[low_key][0], endpoint)
        right = _row_value(by_identity[high_key][0], endpoint)
        if low_key not in valid or high_key not in valid or left is None or right is None:
            excluded.append(seed)
        else:
            differences.append(right - left)
    return {
        "unit": _ENDPOINTS[endpoint][1],
        "denominator_pairs": len(differences),
        "excluded_seed_ids": excluded,
        "estimate": _paired_interval(differences),
        "arrival_time_condition": "both_widths_successful"
        if endpoint.startswith("arrival")
        else None,
    }


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
    return {
        "schema_version": "issue_9348_three_width_report.v1",
        "claim_boundary": "simulator-internal paired width contrasts; three seeds per planner",
        "evidence_status": "candidate" if len(valid) == 18 else "diagnostic_incomplete",
        "planned_rows": 18,
        "observed_rows": len(rows),
        "native_rows": len(valid),
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
    if (
        sha256_file(root / "inputs/application_manifest.yaml")
        != run_manifest.get("application_manifest_sha256")
        or sha256_file(root / "inputs/social_force.yaml")
        != run_manifest.get("planner_config_sha256", {}).get("social_force")
        or sha256_file(root / "inputs/episode.schema.v1.json")
        != run_manifest.get("episode_schema_sha256")
    ):
        raise ValueError("doorway copied scientific input digest mismatch")
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
    if any(
        report.get(key) != rebuilt.get(key)
        for key in ("row_inventory", "contrasts", "native_rows", "excluded_rows")
    ):
        raise ValueError("doorway report differs from sealed raw episodes")
    return report


def run_campaign(manifest_path: Path, output_root: Path) -> Path:
    """Execute H400 serially and seal all 18 raw rows and the paired report.

    Returns:
        Path to the completed result root.
    """
    source_sha = _source_identity()
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
        if not preflight["go"]:
            raise ValueError("doorway geometry/oracle preflight did not admit policy execution")
        assets = [
            record["assets"]
            | {"variant_id": record["variant_id"], "gap_width_m": record["geometry"]["gap_width_m"]}
            for record in preflight["variants"]
        ]
        session = DoorwayPairingSession()
        sf_path = Path(manifest["_resolved"]["social_force_config_path"])
        sf_config = yaml.safe_load(sf_path.read_text(encoding="utf-8"))
        shutil.copy2(sf_path, output_root / "inputs/social_force.yaml")
        shutil.copy2(_SCHEMA, output_root / "inputs/episode.schema.v1.json")
        schema = load_schema(_SCHEMA)
        rows: list[dict[str, Any]] = []
        cells: list[dict[str, Any]] = []
        raw_path = output_root / "episodes.jsonl"
        with raw_path.open("x", encoding="utf-8") as handle:
            for planner in _PLANNERS:
                config_sha = sha256_file(sf_path) if planner == "social_force" else None
                for seed in _SEEDS:
                    for asset in assets:
                        scenario_path = Path(asset["scenario_path"])
                        scenario = dict(load_scenarios(scenario_path)[0])
                        receipt_hook = session.hook(
                            planner=planner,
                            seed=seed,
                            map_sha256=asset["map_sha256"],
                            non_width_config_sha256=non_width_config_sha256(
                                scenario, planner=planner, planner_config_sha256=config_sha
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
                            algo_config=sf_config if planner == "social_force" else None,
                            algo_config_path=sf_path.as_posix()
                            if planner == "social_force"
                            else None,
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
                "application_manifest_path": manifest_path.as_posix(),
                "application_manifest_sha256": sha256_file(manifest_path),
                "planner_config_sha256": {"social_force": sha256_file(sf_path)},
                "episode_schema_sha256": sha256_file(_SCHEMA),
                "horizon_steps": _HORIZON,
                "dt_s": _DT,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "episodes_jsonl_sha256": sha256_file(raw_path),
                "cells": cells,
            },
        )
        _write_checksums(output_root)
    except Exception as exc:
        write_json(
            output_root / "run_failure.json",
            {
                "schema_version": "issue_9348_run_failure.v1",
                "source_commit": source_sha,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        _write_checksums(output_root)
        raise
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
