"""Fail-closed comparison for the release-0.0.2 numeric subset replay."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

REQUIRED_METRICS = (
    "success",
    "collisions",
    "near_misses",
    "time_to_goal_norm",
    "snqi",
)


def _failure(error: str) -> dict[str, Any]:
    return {"status": "fail", "error": error, "planners": {}}


def extract_subset_run_metrics(  # noqa: C901, PLR0912
    campaign_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Extract exactly one target scenario/seed row for every expected planner."""
    if not campaign_root.is_dir():
        return _failure(f"Campaign output directory does not exist: {campaign_root}")

    contract = manifest.get("subset_replay_contract")
    if not isinstance(contract, dict):
        return _failure("Manifest does not contain a subset_replay_contract")
    scenario_id = contract.get("scenario_id")
    seed = contract.get("seed")
    expected_planners = contract.get("planners")
    if not isinstance(scenario_id, str) or not isinstance(seed, int):
        return _failure("Subset replay contract has invalid scenario_id or seed")
    if not isinstance(expected_planners, list) or not all(
        isinstance(planner, str) for planner in expected_planners
    ):
        return _failure("Subset replay contract has invalid planners")
    expected_planner_set = set(expected_planners)

    episodes_paths = sorted(campaign_root.glob("runs/*/episodes.jsonl"))
    if not episodes_paths:
        return _failure(f"No episode JSONL files found under {campaign_root}/runs")

    planners: dict[str, Any] = {}
    for ep_path in episodes_paths:
        try:
            with ep_path.open(encoding="utf-8") as episode_file:
                for line_number, line in enumerate(episode_file, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    data = json.loads(stripped)
                    if not isinstance(data, dict):
                        return _failure(f"Episode row must be an object: {ep_path}:{line_number}")
                    if data.get("scenario_id") != scenario_id or data.get("seed") != seed:
                        continue

                    planner = data.get("algo") or data.get("planner_key")
                    if not isinstance(planner, str) or not planner:
                        return _failure(
                            f"Target replay row has no planner identity: {ep_path}:{line_number}"
                        )
                    if planner not in expected_planner_set:
                        return _failure(
                            f"Unexpected planner '{planner}' in target replay row: "
                            f"{ep_path}:{line_number}"
                        )
                    if planner in planners:
                        return _failure(
                            f"Duplicate replay row for planner '{planner}', scenario "
                            f"'{scenario_id}', seed {seed}"
                        )

                    algorithm_metadata = data.get("algorithm_metadata")
                    if not isinstance(algorithm_metadata, dict):
                        algorithm_metadata = {}
                    planner_kinematics = algorithm_metadata.get("planner_kinematics")
                    if not isinstance(planner_kinematics, dict):
                        planner_kinematics = {}
                    metrics = data.get("metrics")
                    if not isinstance(metrics, dict):
                        metrics = {}

                    planners[planner] = {
                        "scenario_id": data.get("scenario_id"),
                        "seed": data.get("seed"),
                        "status": data.get("status"),
                        "execution_mode": (
                            planner_kinematics.get("execution_mode") or data.get("execution_mode")
                        ),
                        "algorithm_metadata_status": algorithm_metadata.get("status"),
                        "git_hash": data.get("git_hash"),
                        "config_hash": data.get("config_hash"),
                        "metrics": {
                            metric: metrics[metric]
                            for metric in REQUIRED_METRICS
                            if metric in metrics
                        },
                    }
        except (json.JSONDecodeError, OSError) as exc:
            return _failure(f"Error reading episode log {ep_path}: {exc}")

    return {"status": "pass", "planners": planners}


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _compare_planner_metrics(  # noqa: C901, PLR0912
    planner: str,
    expected: dict[str, Any],
    actual: Any,
    tolerances: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Compare required metrics without manufacturing defaults for absent data."""
    deviations: list[str] = []
    if not isinstance(actual, dict):
        return False, [f"Planner '{planner}' metrics must be an object"]

    for metric in REQUIRED_METRICS:
        if metric not in actual:
            deviations.append(f"Planner '{planner}' missing required metric '{metric}'")
    if deviations:
        return False, deviations

    if not isinstance(actual["success"], bool):
        deviations.append(f"Planner '{planner}' metric 'success' must be boolean")
    elif actual["success"] != expected.get("success"):
        deviations.append(
            f"Success mismatch for '{planner}': actual={actual['success']} "
            f"vs expected={expected.get('success')}"
        )

    collisions = actual["collisions"]
    if not isinstance(collisions, int) or isinstance(collisions, bool):
        deviations.append(f"Planner '{planner}' metric 'collisions' must be an integer")
    elif collisions != expected.get("collisions"):
        deviations.append(
            f"Collision count mismatch for '{planner}': actual={collisions} "
            f"vs expected={expected.get('collisions')}"
        )

    near_misses = actual["near_misses"]
    if not _is_finite_number(near_misses):
        deviations.append(f"Planner '{planner}' metric 'near_misses' must be finite numeric")
    else:
        near_miss_tolerance = float(tolerances["near_misses_abs"])
        difference = abs(float(near_misses) - float(expected["near_misses"]))
        if difference > near_miss_tolerance:
            deviations.append(
                f"Near-misses breach for '{planner}': diff={difference:.4f} "
                f"> tol={near_miss_tolerance:.4f}"
            )

    time_to_goal = actual["time_to_goal_norm"]
    if not _is_finite_number(time_to_goal):
        deviations.append(f"Planner '{planner}' metric 'time_to_goal_norm' must be finite numeric")
    elif float(time_to_goal) != float(expected["time_to_goal_norm"]):
        deviations.append(
            f"Time-to-goal-norm mismatch for '{planner}': actual={time_to_goal} "
            f"vs expected={expected['time_to_goal_norm']}"
        )

    snqi = actual["snqi"]
    if not _is_finite_number(snqi):
        deviations.append(f"Planner '{planner}' metric 'snqi' must be finite numeric")
    else:
        snqi_tolerance = float(tolerances["snqi_abs"])
        difference = abs(float(snqi) - float(expected["snqi"]))
        if difference > snqi_tolerance:
            deviations.append(
                f"SNQI score breach for '{planner}': diff={difference:.4f} "
                f"> tol={snqi_tolerance:.4f} (actual={float(snqi):.4f} "
                f"vs expected={float(expected['snqi']):.4f})"
            )

    return not deviations, deviations


def _compare_single_planner(  # noqa: C901
    planner: str,
    expected_info: dict[str, Any] | None,
    actual_info: dict[str, Any] | None,
    *,
    scenario_id: str,
    seed: int,
    replay_source_commit: str,
    tolerances: dict[str, Any],
) -> dict[str, Any]:
    """Compare identity, provenance, execution semantics, and metrics for one planner."""
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

    deviations: list[str] = []
    if actual_info.get("scenario_id") != scenario_id:
        deviations.append(
            f"Scenario mismatch for '{planner}': actual={actual_info.get('scenario_id')} "
            f"vs expected={scenario_id}"
        )
    if actual_info.get("seed") != seed:
        deviations.append(
            f"Seed mismatch for '{planner}': actual={actual_info.get('seed')} vs expected={seed}"
        )
    if actual_info.get("status") != expected_info.get("status"):
        deviations.append(
            f"Status mismatch for '{planner}': actual={actual_info.get('status')} "
            f"vs expected={expected_info.get('status')}"
        )
    actual_execution_mode = actual_info.get("execution_mode")
    if str(actual_execution_mode).lower() in {
        "fallback",
        "degraded",
        "unknown",
        "unavailable",
    }:
        deviations.append(
            f"Planner '{planner}' executed in non-native/untrusted mode: {actual_execution_mode}"
        )
    if actual_execution_mode != expected_info.get("execution_mode"):
        deviations.append(
            f"Execution mode mismatch for '{planner}': "
            f"actual={actual_execution_mode} "
            f"vs expected={expected_info.get('execution_mode')}"
        )

    metadata_status = actual_info.get("algorithm_metadata_status")
    expected_metadata_status = expected_info.get("algorithm_metadata_status")
    if metadata_status != expected_metadata_status:
        deviations.append(
            f"Planner '{planner}' has untrusted algorithm metadata status: {metadata_status} "
            f"(expected {expected_metadata_status})"
        )

    git_hash = actual_info.get("git_hash")
    config_hash = actual_info.get("config_hash")
    if not git_hash or not config_hash:
        deviations.append(
            f"Planner '{planner}' missing provenance hashes "
            f"(git_hash={git_hash}, config_hash={config_hash})"
        )
    else:
        if git_hash != replay_source_commit:
            deviations.append(
                f"Git hash mismatch for '{planner}': actual={git_hash} "
                f"vs expected={replay_source_commit}"
            )
        if config_hash != expected_info.get("config_hash"):
            deviations.append(
                f"Config hash mismatch for '{planner}': actual={config_hash} "
                f"vs expected={expected_info.get('config_hash')}"
            )

    metrics_match, metric_deviations = _compare_planner_metrics(
        planner,
        expected_info.get("metrics", {}),
        actual_info.get("metrics"),
        tolerances,
    )
    if not metrics_match:
        deviations.extend(metric_deviations)

    return {
        "planner": planner,
        "expected": expected_info,
        "actual": actual_info,
        "match": not deviations,
        "deviations": deviations,
    }


def compare_subset_results(  # noqa: C901
    extracted_actual: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Compare the extracted replay against the frozen, sourced subset contract."""
    contract = manifest.get("subset_replay_contract")
    if not isinstance(contract, dict):
        return {
            "status": "fail",
            "overall_match": False,
            "error": "Manifest does not contain a subset_replay_contract",
            "comparison_rows": [],
            "global_deviations": ["Missing subset replay contract"],
        }

    expected_planners = contract.get("planners")
    expected_rows = contract.get("expected_rows")
    tolerances = contract.get("tolerances")
    scenario_id = contract.get("scenario_id")
    seed = contract.get("seed")
    replay_source_commit = contract.get("replay_source_commit")
    global_deviations: list[str] = []

    if not isinstance(expected_planners, list) or not all(
        isinstance(planner, str) for planner in expected_planners
    ):
        expected_planners = []
        global_deviations.append("Subset contract planners must be a list of strings")
    elif len(expected_planners) != len(set(expected_planners)):
        global_deviations.append("Subset contract contains duplicate planners")
    if not isinstance(expected_rows, dict):
        expected_rows = {}
        global_deviations.append("Subset contract expected_rows must be an object")
    if not isinstance(tolerances, dict) or not all(
        key in tolerances for key in ("source", "near_misses_abs", "snqi_abs")
    ):
        tolerances = {"source": "invalid", "near_misses_abs": 0.0, "snqi_abs": 0.0}
        global_deviations.append("Subset contract tolerances are missing or malformed")
    if not isinstance(scenario_id, str) or not isinstance(seed, int):
        global_deviations.append("Subset contract scenario_id or seed is malformed")
    if not isinstance(replay_source_commit, str) or len(replay_source_commit) != 40:
        global_deviations.append("Subset contract replay_source_commit must be a full commit hash")

    actual_planners = extracted_actual.get("planners")
    if not isinstance(actual_planners, dict):
        actual_planners = {}
        global_deviations.append("Extracted replay planners must be an object")
    if extracted_actual.get("status") != "pass":
        global_deviations.append(extracted_actual.get("error", "Replay extraction failed"))

    expected_set = set(expected_planners)
    unexpected_actual = sorted(set(actual_planners) - expected_set)
    if unexpected_actual:
        global_deviations.append(f"Unexpected actual planners: {unexpected_actual}")
    unexpected_expected = sorted(set(expected_rows) - expected_set)
    if unexpected_expected:
        global_deviations.append(f"Unexpected expected rows: {unexpected_expected}")

    comparison_rows = [
        _compare_single_planner(
            planner,
            expected_rows.get(planner),
            actual_planners.get(planner),
            scenario_id=scenario_id,
            seed=seed,
            replay_source_commit=replay_source_commit,
            tolerances=tolerances,
        )
        for planner in expected_planners
    ]
    overall_match = not global_deviations and all(row["match"] for row in comparison_rows)
    return {
        "status": "pass" if overall_match else "fail",
        "overall_match": overall_match,
        "comparison_rows": comparison_rows,
        "global_deviations": global_deviations,
    }
