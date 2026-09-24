#!/usr/bin/env python3
"""Compare the immutable 0.0.6 and corrected 0.0.7 S30/H600 rows.

The comparator is deliberately independent of the benchmark runner.  It streams
episode JSONL from the predecessor publication archive and the successor
campaign root, pairs rows by planner/scenario/seed, and emits a compact JSON and
Markdown diff.  Trace-dependent goal-adjacent labels remain ``unavailable`` when
the input row did not record a simulation trace; a missing label is never
silently treated as ``false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.release_acceptance import _algorithm_metadata_runtime_marker

OUTCOME_KEYS = ("route_complete", "collision_event", "timeout_event")
# A release row may be wholly native, use an explicit adapter, or combine
# native and adapter components (for example a guarded PPO wrapper).  All
# three are contract-valid; fallback/degraded flags remain disqualifying.
EXECUTION_MODES = {"native", "adapter", "mixed"}
# Frozen from the accepted issue #9431 S30/H600 campaign identity. Counts alone
# are not enough: each arm must contain this exact planner × scenario × seed
# Cartesian product. The scenario IDs resolve from the pinned 0.0.7 source
# matrix (configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml).
EXPECTED_ARM_KEYS = frozenset(
    {
        "goal",
        "guarded_ppo",
        "hybrid_rule_v3_fast_progress_static_escape",
        "hybrid_rule_v3_fast_progress_static_escape_continuous",
        "orca",
        "ppo",
        "prediction_planner",
        "predictive_mppi",
        "risk_dwa",
        "sacadrl",
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield",
        "scenario_adaptive_hybrid_orca_v2_collision_guard",
        "social_force",
        "socnav_sampling",
    }
)
EXPECTED_SCENARIO_IDS = frozenset(
    {
        "classic_bottleneck_high",
        "classic_bottleneck_low",
        "classic_bottleneck_medium",
        "classic_cross_trap_high",
        "classic_cross_trap_low",
        "classic_cross_trap_medium",
        "classic_doorway_high",
        "classic_doorway_low",
        "classic_doorway_medium",
        "classic_group_crossing_high",
        "classic_group_crossing_low",
        "classic_group_crossing_medium",
        "classic_head_on_corridor_low",
        "classic_head_on_corridor_medium",
        "classic_merging_low",
        "classic_merging_medium",
        "classic_overtaking_low",
        "classic_overtaking_medium",
        "classic_realworld_double_bottleneck_high",
        "classic_station_platform_medium",
        "classic_t_intersection_low",
        "classic_t_intersection_medium",
        "classic_urban_crossing_medium",
        "francis2023_accompanying_peer",
        "francis2023_blind_corner",
        "francis2023_circular_crossing",
        "francis2023_crowd_navigation",
        "francis2023_down_path",
        "francis2023_entering_elevator",
        "francis2023_entering_room",
        "francis2023_exiting_elevator",
        "francis2023_exiting_room",
        "francis2023_following_human",
        "francis2023_frontal_approach",
        "francis2023_intersection_no_gesture",
        "francis2023_intersection_proceed",
        "francis2023_intersection_wait",
        "francis2023_join_group",
        "francis2023_leading_human",
        "francis2023_leave_group",
        "francis2023_narrow_doorway",
        "francis2023_narrow_hallway",
        "francis2023_parallel_traffic",
        "francis2023_pedestrian_obstruction",
        "francis2023_pedestrian_overtaking",
        "francis2023_perpendicular_traffic",
        "francis2023_robot_crowding",
        "francis2023_robot_overtaking",
    }
)
EXPECTED_SEEDS = frozenset(range(111, 141))
EXPECTED_PREDECESSOR_SCENARIO_MANIFEST = "configs/scenarios/classic_interactions_francis2023.yaml"
EXPECTED_PREDECESSOR_SCENARIO_MANIFEST_SHA256 = (
    "d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5"
)
EXPECTED_SUCCESSOR_SCENARIO_MANIFEST = (
    "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml"
)
EXPECTED_SUCCESSOR_SCENARIO_MANIFEST_SHA256 = (
    "03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c"
)
EXPECTED_ARM_COUNT = len(EXPECTED_ARM_KEYS)
EXPECTED_ROWS_PER_ARM = len(EXPECTED_SCENARIO_IDS) * len(EXPECTED_SEEDS)
EXPECTED_TOTAL_ROWS = EXPECTED_ARM_COUNT * EXPECTED_ROWS_PER_ARM
BOOTSTRAP_SAMPLES = 3000
BOOTSTRAP_SEED = 123


def _row_outcome(row: Mapping[str, Any]) -> dict[str, bool]:
    outcome = row.get("outcome")
    if not isinstance(outcome, Mapping):
        raise ValueError("row outcome must be an object")
    invalid = [key for key in OUTCOME_KEYS if type(outcome.get(key)) is not bool]
    if invalid:
        raise ValueError("row outcome requires explicit boolean values for " + ", ".join(invalid))
    return {key: outcome[key] for key in OUTCOME_KEYS}


def _goal_adjacent_status(row: Mapping[str, Any]) -> str:
    """Return true/false only for explicit labels; otherwise unavailable."""
    candidates: list[Any] = []
    for key in ("goal_adjacent_timeout", "goal_adjacent_timeout_event", "goal_adjacent"):
        if key in row:
            candidates.append(row[key])
    failure = row.get("failure_mechanism")
    if isinstance(failure, Mapping):
        label = failure.get("mechanism_label")
        if isinstance(label, str) and "goal_adjacent" in label.lower():
            candidates.append(True)
    for value in candidates:
        if isinstance(value, bool):
            return "true" if value else "false"
    return "unavailable"


def _execution_audit(row: Mapping[str, Any], *, expected_algorithm: str | None = None) -> list[str]:
    issues: list[str] = []
    metadata = row.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        issues.append("algorithm_metadata=missing")
    else:
        kinematics = metadata.get("planner_kinematics")
        if not isinstance(kinematics, Mapping):
            issues.append("planner_kinematics=missing")
        else:
            mode = kinematics.get("execution_mode")
            if not isinstance(mode, str) or mode not in EXECUTION_MODES:
                issues.append(f"execution_mode={mode if mode is not None else 'missing'}")
        runtime_marker = _algorithm_metadata_runtime_marker(
            metadata, expected_algorithm=expected_algorithm
        )
        if runtime_marker is not None:
            marker_path, marker_value = runtime_marker
            issues.append(f"algorithm_metadata.{marker_path}={marker_value}")
        if metadata.get("unavailable") is True:
            issues.append("unavailable=true")
    # Outcome statuses are not execution degradation.  A valid benchmark row
    # may finish as success, collision, timeout, or failure; only explicit
    # runtime/error/unavailable markers are disqualifying here.
    if str(row.get("status", "")).lower() not in {
        "",
        "ok",
        "success",
        "completed",
        "collision",
        "failure",
        "timeout",
        "timed_out",
        "timeout_event",
    }:
        issues.append(f"status={row.get('status')}")
    return issues


def _arm_from_name(name: str) -> str:
    return name.removesuffix("__differential_drive")


def _normalize_sha256(value: str, *, label: str) -> str:
    digest = value.removeprefix("sha256:").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: str, *, label: str) -> str:
    normalized = _normalize_sha256(expected, label=label)
    actual = _sha256(path)
    if actual != normalized:
        raise ValueError(f"{label} mismatch: expected {normalized}, got {actual}")
    return normalized


def _row_source_commit(row: Mapping[str, Any]) -> str:
    provenance = row.get("result_provenance")
    provenance_commit: Any = None
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise ValueError("result_provenance must be an object when present")
        provenance_commit = provenance.get("repo_commit")
        if provenance_commit is not None and not isinstance(provenance_commit, str):
            raise ValueError("result_provenance.repo_commit must be a string")
    git_hash = row.get("git_hash")
    if git_hash is not None and not isinstance(git_hash, str):
        raise ValueError("git_hash must be a string")
    if provenance_commit and git_hash and provenance_commit != git_hash:
        raise ValueError(
            "row source commits disagree: "
            f"result_provenance.repo_commit={provenance_commit}, git_hash={git_hash}"
        )
    commit = provenance_commit or git_hash
    if not commit:
        raise ValueError("row is missing result_provenance.repo_commit and git_hash")
    return commit


def _read_jsonl(path: Path, *, arm: str) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["scenario_id"]), int(row["seed"]))
            if key in rows:
                raise ValueError(f"duplicate {arm} row {key} in {path}:{line_number}")
            rows[key] = row
    return rows


def _read_predecessor(archive: Path) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    result: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    with tarfile.open(archive, "r:gz") as handle:
        members = sorted(
            (
                member
                for member in handle.getmembers()
                if member.isfile()
                and member.name.endswith("/episodes.jsonl")
                and "/payload/runs/" in member.name
            ),
            key=lambda member: member.name,
        )
        if not members:
            raise ValueError(f"predecessor archive has no payload run rows: {archive}")
        for member in members:
            stem = Path(member.name).parent.name
            arm = _arm_from_name(stem)
            if arm in result:
                raise ValueError(f"duplicate predecessor arm {arm} in {archive}")
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read archive member {member.name}")
            rows: dict[tuple[str, int], dict[str, Any]] = {}
            for line_number, raw in enumerate(extracted, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                key = (str(row["scenario_id"]), int(row["seed"]))
                if key in rows:
                    raise ValueError(f"duplicate predecessor row {arm} {key}:{line_number}")
                rows[key] = row
            result[arm] = rows
    return result


def _bundle_resolved_release_manifest(bundle: Path) -> Mapping[str, Any]:
    """Read the unique resolved release manifest from a publication bundle."""
    manifest_suffix = "/payload/release/release_manifest.resolved.json"
    with tarfile.open(bundle, "r:gz") as handle:
        matches = [
            member
            for member in handle.getmembers()
            if member.isfile() and member.name.endswith(manifest_suffix)
        ]
        if len(matches) != 1:
            raise ValueError(
                "publication bundle must contain exactly one resolved release manifest; "
                f"found {len(matches)}"
            )
        extracted = handle.extractfile(matches[0])
        if extracted is None:
            raise ValueError(f"cannot read resolved release manifest {matches[0].name}")
        manifest = json.load(extracted)
    if not isinstance(manifest, Mapping):
        raise ValueError("resolved release manifest must be a JSON object")
    return manifest


def _archive_source_commit(archive: Path) -> str:
    """Read the source identity from the publication archive's resolved manifest."""
    manifest = _bundle_resolved_release_manifest(archive)
    source_commit = manifest.get("source_sha") or manifest.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("resolved release manifest is missing a source commit")
    return source_commit


def _bundle_scenario_identity(bundle: Path) -> dict[str, str]:
    """Read the scenario source path and digest pinned by a publication bundle."""
    manifest = _bundle_resolved_release_manifest(bundle)
    scenario = manifest.get("scenario")
    if not isinstance(scenario, Mapping):
        raise ValueError("resolved release manifest is missing scenario identity")
    matrix_path = scenario.get("matrix_path")
    matrix_sha256 = scenario.get("matrix_sha256")
    if not isinstance(matrix_path, str) or not matrix_path.strip():
        raise ValueError("resolved release manifest scenario is missing matrix_path")
    if not isinstance(matrix_sha256, str):
        raise ValueError("resolved release manifest scenario is missing matrix_sha256")
    return {
        "manifest": matrix_path,
        "sha256": _normalize_sha256(matrix_sha256, label="scenario manifest SHA-256"),
    }


def _release_scenario_identities(
    predecessor_archive: Path, successor_bundle: Path
) -> dict[str, dict[str, str]]:
    """Require and return the two expected release-specific scenario identities."""
    identities = {
        "predecessor": _bundle_scenario_identity(predecessor_archive),
        "successor": _bundle_scenario_identity(successor_bundle),
    }
    expected = {
        "predecessor": {
            "manifest": EXPECTED_PREDECESSOR_SCENARIO_MANIFEST,
            "sha256": EXPECTED_PREDECESSOR_SCENARIO_MANIFEST_SHA256,
        },
        "successor": {
            "manifest": EXPECTED_SUCCESSOR_SCENARIO_MANIFEST,
            "sha256": EXPECTED_SUCCESSOR_SCENARIO_MANIFEST_SHA256,
        },
    }
    for release, identity in identities.items():
        if identity != expected[release]:
            raise ValueError(
                f"{release} scenario identity mismatch: "
                f"expected {expected[release]}, got {identity}"
            )
    return identities


def _bundle_campaign_id(bundle: Path) -> str:
    """Read the stable campaign ID from the verified successor bundle."""
    manifest_suffix = "/payload/campaign_manifest.json"
    with tarfile.open(bundle, "r:gz") as handle:
        matches = [
            member
            for member in handle.getmembers()
            if member.isfile() and member.name.endswith(manifest_suffix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"successor bundle must contain exactly one campaign manifest; found {len(matches)}"
            )
        extracted = handle.extractfile(matches[0])
        if extracted is None:
            raise ValueError(f"cannot read campaign manifest {matches[0].name}")
        manifest = json.load(extracted)
    if not isinstance(manifest, Mapping):
        raise ValueError("successor campaign manifest must be a JSON object")
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("successor campaign manifest is missing a campaign ID")
    return campaign_id


def _read_successor(root: Path) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    result: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for path in sorted(root.glob("runs/*/episodes.jsonl")):
        result[_arm_from_name(path.parent.name)] = _read_jsonl(
            path, arm=_arm_from_name(path.parent.name)
        )
    if not result:
        raise ValueError(f"successor root has no run rows: {root}")
    return result


def _bundle_episode_digests(bundle: Path) -> dict[str, str]:
    """Return arm-to-digest bindings for episode rows inside a release bundle."""
    suffix = "/payload/runs/"
    result: dict[str, str] = {}
    with tarfile.open(bundle, "r:gz") as handle:
        members = sorted(
            (
                member
                for member in handle.getmembers()
                if member.isfile()
                and suffix in member.name
                and member.name.endswith("/episodes.jsonl")
            ),
            key=lambda member: member.name,
        )
        if not members:
            raise ValueError(f"successor bundle has no payload run rows: {bundle}")
        for member in members:
            relative = member.name.rsplit(suffix, 1)[1]
            parts = Path(relative).parts
            if len(parts) != 2 or parts[1] != "episodes.jsonl":
                raise ValueError(
                    f"successor bundle episode member must be one run directory deep: {member.name}"
                )
            arm = _arm_from_name(parts[0])
            if arm in result:
                raise ValueError(f"duplicate successor bundle arm {arm} in {bundle}")
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read successor bundle member {member.name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
            result[arm] = digest.hexdigest()
    return result


def _validate_successor_bundle_root(bundle: Path, root: Path) -> None:
    """Require the compared root's episode bytes to come from the pinned bundle."""
    bundle_digests = _bundle_episode_digests(bundle)
    root_digests = {
        _arm_from_name(path.parent.name): _sha256(path)
        for path in sorted(root.glob("runs/*/episodes.jsonl"))
    }
    if set(bundle_digests) != set(root_digests):
        raise ValueError(
            "successor bundle/root arm sets differ: "
            f"bundle_only={sorted(set(bundle_digests) - set(root_digests))}, "
            f"root_only={sorted(set(root_digests) - set(bundle_digests))}"
        )
    mismatched = [arm for arm in sorted(bundle_digests) if bundle_digests[arm] != root_digests[arm]]
    if mismatched:
        raise ValueError(
            "successor root episode bytes do not match the pinned bundle for arms: "
            + ", ".join(mismatched)
        )


def _validate_matrix(
    old: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]],
    new: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]],
    *,
    expected_arm_keys: AbstractSet[str] = EXPECTED_ARM_KEYS,
    expected_scenario_ids: AbstractSet[str] = EXPECTED_SCENARIO_IDS,
    expected_seeds: AbstractSet[int] = EXPECTED_SEEDS,
) -> None:
    old_arms = set(old)
    new_arms = set(new)
    expected_arms = set(expected_arm_keys)
    if old_arms != expected_arms or new_arms != expected_arms:
        raise ValueError(
            "predecessor/successor arm sets do not match the canonical matrix: "
            f"predecessor_missing={sorted(expected_arms - old_arms)}, "
            f"predecessor_unexpected={sorted(old_arms - expected_arms)}, "
            f"successor_missing={sorted(expected_arms - new_arms)}, "
            f"successor_unexpected={sorted(new_arms - expected_arms)}"
        )
    expected_keys = {
        (scenario_id, seed) for scenario_id in expected_scenario_ids for seed in expected_seeds
    }
    for arm in sorted(old_arms):
        old_keys = set(old[arm])
        new_keys = set(new[arm])
        if old_keys != new_keys:
            raise ValueError(
                f"{arm} scenario/seed keys differ: "
                f"predecessor_only={sorted(old_keys - new_keys)}, "
                f"successor_only={sorted(new_keys - old_keys)}"
            )
        if old_keys != expected_keys:
            raise ValueError(
                f"{arm} does not match the canonical scenario/seed matrix: "
                f"missing={sorted(expected_keys - old_keys)[:10]}, "
                f"unexpected={sorted(old_keys - expected_keys)[:10]}"
            )
    old_total = sum(len(rows) for rows in old.values())
    new_total = sum(len(rows) for rows in new.values())
    expected_total_rows = len(expected_arms) * len(expected_keys)
    if old_total != expected_total_rows or new_total != expected_total_rows:
        raise ValueError(
            f"expected {expected_total_rows} rows in each release, "
            f"got predecessor={old_total}, successor={new_total}"
        )


def _validate_successor_rows(
    rows_by_arm: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]],
    *,
    successor_source_sha: str,
) -> None:
    issues: list[str] = []
    for arm, rows in sorted(rows_by_arm.items()):
        for key, row in sorted(rows.items()):
            context = f"{arm}/{key[0]}/{key[1]}"
            try:
                _row_outcome(row)
                commit = _row_source_commit(row)
            except ValueError as exc:
                issues.append(f"{context}: {exc}")
                continue
            if commit != successor_source_sha:
                issues.append(
                    f"{context}: source commit {commit} does not match {successor_source_sha}"
                )
            for issue in _execution_audit(row, expected_algorithm=arm):
                issues.append(f"{context}: {issue}")
    if issues:
        preview = "; ".join(issues[:20])
        suffix = f"; ... {len(issues) - 20} more" if len(issues) > 20 else ""
        raise ValueError(f"successor row validation failed: {preview}{suffix}")


def _metric(row: Mapping[str, Any], metric: str) -> float:
    return float(_row_outcome(row)[metric])


def _bootstrap_seed_delta(
    old: Mapping[tuple[str, int], Mapping[str, Any]],
    new: Mapping[tuple[str, int], Mapping[str, Any]],
    metric: str,
) -> dict[str, Any] | None:
    seed_values: dict[int, list[float]] = defaultdict(list)
    for key, old_row in old.items():
        if key not in new:
            continue
        seed_values[key[1]].append(_metric(new[key], metric) - _metric(old_row, metric))
    if not seed_values:
        return None
    seeds = sorted(seed_values)
    seed_means = np.asarray([np.mean(seed_values[seed]) for seed in seeds], dtype=float)
    observed = float(np.mean(seed_means))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(seed_means), size=(BOOTSTRAP_SAMPLES, len(seed_means)))
    distribution = seed_means[sampled].mean(axis=1)
    return {
        "metric": metric,
        "estimand": "success/collision/timeout risk difference (0.0.7 - 0.0.6)",
        "seed_block_count": len(seeds),
        "seeds": seeds,
        "observed_delta": observed,
        "ci95": [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))],
        "method": "percentile bootstrap over whole seed blocks",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def compare(
    predecessor_archive: Path,
    successor_root: Path,
    successor_bundle: Path,
    *,
    predecessor_sha256: str,
    predecessor_source_sha: str,
    successor_source_sha: str,
    successor_bundle_sha256: str,
) -> dict[str, Any]:
    """Pair predecessor and successor rows and derive the governed release diff."""
    verified_predecessor_sha256 = _verify_sha256(
        predecessor_archive, predecessor_sha256, label="predecessor archive SHA-256"
    )
    verified_successor_bundle_sha256 = _verify_sha256(
        successor_bundle, successor_bundle_sha256, label="successor bundle SHA-256"
    )
    _validate_successor_bundle_root(successor_bundle, successor_root)
    archive_source_commit = _archive_source_commit(predecessor_archive)
    if archive_source_commit != predecessor_source_sha:
        raise ValueError(
            "predecessor source commit mismatch: "
            f"expected {predecessor_source_sha}, archive declares {archive_source_commit}"
        )
    old = _read_predecessor(predecessor_archive)
    new = _read_successor(successor_root)
    scenario_identities = _release_scenario_identities(predecessor_archive, successor_bundle)
    scenario_definitions_identical = (
        scenario_identities["predecessor"] == scenario_identities["successor"]
    )
    _validate_matrix(old, new)
    _validate_successor_rows(new, successor_source_sha=successor_source_sha)
    arms = sorted(old)
    execution_issues: dict[str, list[str]] = defaultdict(list)
    arm_reports: dict[str, Any] = {}
    changed: list[dict[str, Any]] = []
    bootstrap: dict[str, dict[str, Any]] = {}

    for arm in arms:
        old_rows = old.get(arm, {})
        new_rows = new.get(arm, {})
        paired_keys = sorted(set(old_rows) & set(new_rows))
        old_counts: Counter[str] = Counter()
        new_counts: Counter[str] = Counter()
        old_goal: Counter[str] = Counter()
        new_goal: Counter[str] = Counter()
        for key, row in old_rows.items():
            outcome = _row_outcome(row)
            old_counts["success" if outcome["route_complete"] else "failure"] += 1
            old_counts["collision"] += int(outcome["collision_event"])
            old_counts["timeout"] += int(outcome["timeout_event"])
            old_goal[_goal_adjacent_status(row)] += 1
        for key, row in new_rows.items():
            outcome = _row_outcome(row)
            new_counts["success" if outcome["route_complete"] else "failure"] += 1
            new_counts["collision"] += int(outcome["collision_event"])
            new_counts["timeout"] += int(outcome["timeout_event"])
            new_goal[_goal_adjacent_status(row)] += 1
        for key in paired_keys:
            old_outcome = _row_outcome(old_rows[key])
            new_outcome = _row_outcome(new_rows[key])
            if old_outcome != new_outcome:
                changed.append(
                    {
                        "arm": arm,
                        "scenario_id": key[0],
                        "seed": key[1],
                        "old": old_outcome,
                        "new": new_outcome,
                    }
                )
        arm_reports[arm] = {
            "predecessor_rows": len(old_rows),
            "successor_rows": len(new_rows),
            "paired_rows": len(paired_keys),
            "predecessor_outcomes": dict(sorted(old_counts.items())),
            "successor_outcomes": dict(sorted(new_counts.items())),
            "predecessor_goal_adjacent_timeout": dict(sorted(old_goal.items())),
            "successor_goal_adjacent_timeout": dict(sorted(new_goal.items())),
        }
        for metric in ("route_complete", "collision_event", "timeout_event"):
            result = _bootstrap_seed_delta(old_rows, new_rows, metric)
            if result is not None:
                bootstrap[f"{arm}:{metric}"] = result

    return {
        "schema_version": "issue-9431-release-diff.v1",
        "predecessor": {
            "release": "0.0.6",
            "archive_sha256": verified_predecessor_sha256,
            "archive": predecessor_archive.name,
            "source_commit": archive_source_commit,
        },
        "successor": {
            "release": "0.0.7",
            "source_commit": successor_source_sha,
            "bundle_sha256": verified_successor_bundle_sha256,
            "bundle": successor_bundle.name,
            "campaign_root": _bundle_campaign_id(successor_bundle),
        },
        "matrix_identity": {
            "planner_arms": sorted(EXPECTED_ARM_KEYS),
            "scenario_ids": sorted(EXPECTED_SCENARIO_IDS),
            "seeds": sorted(EXPECTED_SEEDS),
            "episodes_per_arm": EXPECTED_ROWS_PER_ARM,
            "total_episodes": EXPECTED_TOTAL_ROWS,
            "scenario_manifests": scenario_identities,
            "scenario_definitions_identical": scenario_definitions_identical,
        },
        "comparison_design": {
            "classification": "descriptive multi-change release comparison",
            "causal_ablation": False,
            "scenario_definitions_identical": scenario_definitions_identical,
            "note": (
                "Rows are matched by planner, scenario ID, and seed, but the predecessor and "
                "successor scenario definitions differ. The source range includes the "
                "social-force goal-approach repair, the goal-zone success-definition repair, "
                "and runtime-admission corrections; it does not isolate one causal change."
            ),
        },
        "arms": arm_reports,
        "arm_count": len(arms),
        "paired_rows": sum(report["paired_rows"] for report in arm_reports.values()),
        "changed_episode_count": len(changed),
        "changed_episodes": changed,
        "seed_block_bootstrap": bootstrap,
        "execution_audit": {
            "successor_rows_with_fallback_or_degraded": sum(
                len(values) for values in execution_issues.values()
            ),
            "issues_by_arm": {arm: sorted(values) for arm, values in execution_issues.items()},
            "admitted": not execution_issues,
        },
        "goal_adjacent_timeout_policy": {
            "missing_trace_is": "unavailable",
            "note": "Full S30/H600 release rows do not carry simulation-step traces; trace-derived labels are not inferred from outcome rows.",
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Issue #9431 release diff: 0.0.6 → corrected 0.0.7",
        "",
        "This report pairs rows by planner, scenario, and seed. Successor execution is admitted only when every row is native, adapter, or mixed mode with no fallback/degraded row. Trace-dependent goal-adjacent timeout labels remain `unavailable` when step traces were not recorded.",
        "",
        "This is a **descriptive multi-change release comparison**, not a single-change causal ablation. Rows match by planner, scenario ID, and seed, but the scenario definitions differ between releases. The source range includes the social-force goal-approach repair, the goal-zone success-definition repair, and runtime-admission corrections.",
        "",
        f"- validated matrix: **{len(report['matrix_identity']['planner_arms'])} arms × {len(report['matrix_identity']['scenario_ids'])} scenarios × {len(report['matrix_identity']['seeds'])} seeds** = {report['matrix_identity']['total_episodes']} rows",
        f"- scenario sources: 0.0.6 `{report['matrix_identity']['scenario_manifests']['predecessor']['manifest']}` (SHA-256 `{report['matrix_identity']['scenario_manifests']['predecessor']['sha256']}`) → 0.0.7 `{report['matrix_identity']['scenario_manifests']['successor']['manifest']}` (SHA-256 `{report['matrix_identity']['scenario_manifests']['successor']['sha256']}`); definitions identical: **{'yes' if report['matrix_identity']['scenario_definitions_identical'] else 'no'}**",
        "",
        f"- predecessor archive SHA-256: `{report['predecessor']['archive_sha256']}`",
        f"- exact source range: `{report['predecessor']['source_commit']}..{report['successor']['source_commit']}`",
        f"- successor source commit: `{report['successor']['source_commit']}`",
        f"- successor publication bundle SHA-256: `{report['successor']['bundle_sha256']}`",
        f"- paired rows: **{report['paired_rows']}**; changed outcome rows: **{report['changed_episode_count']}**",
        f"- successor execution audit: **{'pass' if report['execution_audit']['admitted'] else 'fail'}**",
        "",
        "## Per-arm outcome counts",
        "",
        "| arm | paired | 0.0.6 success | 0.0.7 success | 0.0.6 collisions | 0.0.7 collisions | 0.0.6 timeouts | 0.0.7 timeouts | goal-adjacent labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm, values in sorted(report["arms"].items()):
        old = values["predecessor_outcomes"]
        new = values["successor_outcomes"]
        old_goal = values["predecessor_goal_adjacent_timeout"]
        new_goal = values["successor_goal_adjacent_timeout"]
        old_goal_text = ", ".join(f"{key}={value}" for key, value in sorted(old_goal.items()))
        new_goal_text = ", ".join(f"{key}={value}" for key, value in sorted(new_goal.items()))
        lines.append(
            f"| `{arm}` | {values['paired_rows']} | {old.get('success', 0)} | {new.get('success', 0)} | {old.get('collision', 0)} | {new.get('collision', 0)} | {old.get('timeout', 0)} | {new.get('timeout', 0)} | old `{old_goal_text}`; new `{new_goal_text}` |"
        )
    lines += [
        "",
        "## Seed-block bootstrap intervals",
        "",
        "The interval is a deterministic 95% percentile bootstrap over whole seed blocks (3000 replicates, seed 123). The estimand is the risk difference 0.0.7 minus 0.0.6.",
        "",
        "| arm | metric | observed delta | 95% interval | seed blocks |",
        "|---|---|---:|---|---:|",
    ]
    for key, values in sorted(report["seed_block_bootstrap"].items()):
        lines.append(
            f"| `{key.split(':', 1)[0]}` | {values['metric']} | {values['observed_delta']:.6f} | [{values['ci95'][0]:.6f}, {values['ci95'][1]:.6f}] | {values['seed_block_count']} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The full release rows are outcome evidence only. Because they do not contain simulation-step traces, the goal-adjacent timeout predicate is not recomputed here and `unavailable` is not counted as false. The separately pinned 0.0.7 worked-example trace bundle supplies the bounded head-on/group-crossing diagnostic traces.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Run the release comparator CLI and write JSON and Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-archive", type=Path, required=True)
    parser.add_argument("--successor-root", type=Path, required=True)
    parser.add_argument("--successor-bundle", type=Path, required=True)
    parser.add_argument("--predecessor-sha256", required=True)
    parser.add_argument("--predecessor-source-sha", required=True)
    parser.add_argument("--successor-source-sha", required=True)
    parser.add_argument("--successor-bundle-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(
            args.predecessor_archive,
            args.successor_root,
            args.successor_bundle,
            predecessor_sha256=args.predecessor_sha256,
            predecessor_source_sha=args.predecessor_source_sha,
            successor_source_sha=args.successor_source_sha,
            successor_bundle_sha256=args.successor_bundle_sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "paired_rows": report["paired_rows"],
                "changed_episode_count": report["changed_episode_count"],
                "output_json": str(args.output_json),
                "output_markdown": str(args.output_markdown),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
