"""Smallest reusable comparison helper for release 0.0.2 cold-start subset replay.

Compares actual numeric replay outcomes against frozen release expectations and tolerances.
Fails closed on missing data, fallback/degraded execution, absent provenance, or tolerance breaches.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def extract_subset_run_metrics(campaign_root: Path) -> dict[str, Any]:
    """Extract actual per-planner scenario/seed metrics from a benchmark run output directory."""
    if not campaign_root.is_dir():
        return {
            "status": "fail",
            "error": f"Campaign output directory does not exist: {campaign_root}",
            "planners": {},
        }

    planners: dict[str, Any] = {}
    episodes_paths = list(campaign_root.glob("runs/*/episodes.jsonl"))
    if not episodes_paths:
        return {
            "status": "fail",
            "error": f"No episode JSONL files found under {campaign_root}/runs",
            "planners": {},
        }

    for ep_path in episodes_paths:
        try:
            with open(ep_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        continue
                    algo = (
                        data.get("algo")
                        or data.get("planner_key")
                        or ep_path.parent.name.split("__")[0]
                    )
                    if not algo:
                        continue

                    algo_meta = data.get("algorithm_metadata", {})
                    planner_kin = algo_meta.get("planner_kinematics", {})
                    exec_mode = (
                        planner_kin.get("execution_mode") or data.get("execution_mode") or "unknown"
                    )

                    metrics = data.get("metrics", {})
                    planners[algo] = {
                        "scenario_id": data.get("scenario_id"),
                        "seed": data.get("seed"),
                        "status": data.get("status"),
                        "execution_mode": exec_mode,
                        "algorithm_metadata_status": algo_meta.get("status"),
                        "git_hash": data.get("git_hash"),
                        "config_hash": data.get("config_hash"),
                        "metrics": {
                            "success": bool(metrics.get("success", False)),
                            "collisions": int(metrics.get("collisions", 0)),
                            "near_misses": float(metrics.get("near_misses", 0.0)),
                            "time_to_goal_norm": float(metrics.get("time_to_goal_norm", 1.0)),
                            "snqi": float(metrics.get("snqi", 0.0)),
                        },
                    }
        except (json.JSONDecodeError, OSError) as exc:
            return {
                "status": "fail",
                "error": f"Error reading episode log {ep_path}: {exc}",
                "planners": {},
            }

    return {
        "status": "pass",
        "planners": planners,
    }


def _compare_planner_metrics(
    planner: str,
    exp_metrics: dict[str, Any],
    act_metrics: dict[str, Any],
    tolerances: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Compare numeric metrics against expected values and tolerances."""
    match = True
    deviations: list[str] = []

    default_snqi_tol = float(tolerances.get("default_abs_snqi", 0.35))
    near_miss_tol = float(tolerances.get("near_misses", 0.31))
    ttg_tol = float(tolerances.get("time_to_goal_norm", 0.05))
    collision_tol = int(tolerances.get("collisions", 1))

    if act_metrics.get("success") != exp_metrics.get("success"):
        match = False
        deviations.append(
            f"Success mismatch for '{planner}': actual={act_metrics.get('success')} "
            f"vs expected={exp_metrics.get('success')}"
        )

    col_diff = abs(act_metrics.get("collisions", 0) - exp_metrics.get("collisions", 0))
    if col_diff > collision_tol:
        match = False
        deviations.append(
            f"Collision count breach for '{planner}': diff={col_diff} > tol={collision_tol} "
            f"(actual={act_metrics.get('collisions')} vs expected={exp_metrics.get('collisions')})"
        )

    nm_diff = abs(act_metrics.get("near_misses", 0.0) - exp_metrics.get("near_misses", 0.0))
    if nm_diff > near_miss_tol:
        match = False
        deviations.append(
            f"Near-misses breach for '{planner}': diff={nm_diff:.4f} > tol={near_miss_tol:.4f}"
        )

    ttg_diff = abs(
        act_metrics.get("time_to_goal_norm", 1.0) - exp_metrics.get("time_to_goal_norm", 1.0)
    )
    if ttg_diff > ttg_tol:
        match = False
        deviations.append(
            f"Time-to-goal-norm breach for '{planner}': diff={ttg_diff:.4f} > tol={ttg_tol:.4f}"
        )

    snqi_diff = abs(act_metrics.get("snqi", 0.0) - exp_metrics.get("snqi", 0.0))
    if snqi_diff > default_snqi_tol:
        match = False
        deviations.append(
            f"SNQI score breach for '{planner}': diff={snqi_diff:.4f} > tol={default_snqi_tol:.4f} "
            f"(actual={act_metrics.get('snqi'):.4f} vs expected={exp_metrics.get('snqi'):.4f})"
        )

    return match, deviations


def _compare_single_planner(
    planner: str,
    expected_info: dict[str, Any] | None,
    actual_info: dict[str, Any] | None,
    tolerances: dict[str, Any],
) -> dict[str, Any]:
    """Compare a single planner's actual outcome against its expected contract."""
    deviations: list[str] = []

    if not expected_info:
        return {
            "planner": planner,
            "expected": None,
            "actual": actual_info,
            "match": False,
            "deviations": [f"Missing expected row contract for planner '{planner}'"],
        }

    if not actual_info:
        return {
            "planner": planner,
            "expected": expected_info,
            "actual": None,
            "match": False,
            "deviations": [f"Missing actual replay output for planner '{planner}'"],
        }

    match = True
    exec_mode = str(actual_info.get("execution_mode", "")).lower()
    if exec_mode in ("fallback", "degraded", "unknown", "unavailable"):
        match = False
        deviations.append(f"Planner '{planner}' executed in non-native/untrusted mode: {exec_mode}")

    git_hash = actual_info.get("git_hash")
    config_hash = actual_info.get("config_hash")
    if not git_hash or not config_hash:
        match = False
        deviations.append(
            f"Planner '{planner}' missing provenance hashes "
            f"(git_hash={git_hash}, config_hash={config_hash})"
        )

    metrics_match, metric_deviations = _compare_planner_metrics(
        planner,
        expected_info.get("metrics", {}),
        actual_info.get("metrics", {}),
        tolerances,
    )
    if not metrics_match:
        match = False
        deviations.extend(metric_deviations)

    return {
        "planner": planner,
        "expected": expected_info,
        "actual": actual_info,
        "match": match,
        "deviations": deviations,
    }


def compare_subset_results(
    extracted_actual: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Compare extracted actual replay metrics against the frozen subset contract in manifest.

    Fails closed on missing planners/rows, fallback/degraded execution, absent provenance,
    or tolerance breaches.
    """
    contract = manifest.get("subset_replay_contract", {})
    if not contract:
        return {
            "status": "fail",
            "error": "Manifest does not contain a subset_replay_contract",
            "comparison_rows": [],
        }

    expected_planners = contract.get("planners", [])
    expected_rows = contract.get("expected_rows", {})
    tolerances = contract.get("tolerances", {})
    actual_planners = extracted_actual.get("planners", {})

    comparison_rows: list[dict[str, Any]] = []
    overall_match = True
    global_deviations: list[str] = []

    if extracted_actual.get("status") == "fail":
        overall_match = False
        global_deviations.append(extracted_actual.get("error", "Replay extraction failed"))

    for planner in expected_planners:
        row = _compare_single_planner(
            planner,
            expected_rows.get(planner),
            actual_planners.get(planner),
            tolerances,
        )
        if not row["match"]:
            overall_match = False
        comparison_rows.append(row)

    return {
        "status": "pass" if overall_match else "fail",
        "overall_match": overall_match,
        "comparison_rows": comparison_rows,
        "global_deviations": global_deviations,
    }
