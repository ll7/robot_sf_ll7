#!/usr/bin/env python3
"""Fail-closed, descriptive 0.0.7→0.0.8 episode comparison for #9668.

The accepted 0.0.7 bundle is read in place, never extracted or modified. A
candidate identity document binds the corrected source, inputs, arm mapping,
and row bytes. An attribution ledger binds each changed field to a versioned
change and a checksummed, reviewed causal receipt. Structural acceptance by
this command does not replace independent scientific review of those receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tarfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.policy_search_manifest import resolve_candidate_manifest_runtime
from robot_sf.benchmark.release_acceptance import _status_markers
from scripts.analysis.compare_issue_9431_release import (
    EXPECTED_ARM_KEYS,
    EXPECTED_SCENARIO_IDS,
    EXPECTED_SEEDS,
    _execution_audit,
)

REPORT_SCHEMA = "release_007_008_comparison.v1"
CANDIDATE_SCHEMA = "release_007_008_candidate_identity.v1"
ATTRIBUTION_SCHEMA = "release_007_008_attribution_ledger.v1"
RECEIPT_SCHEMA = "release_007_008_causal_receipt.v1"
HISTORICAL_BUNDLE_SHA256 = "684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f"
HISTORICAL_SOURCE_SHA = "07f7e8d43084de748915e1b1eb8b2a1603357c6e"
HISTORICAL_EFFECTIVE_CONFIG_SHA256 = (
    "095331329b06673dc165109c8523579549f769c98542b207a712f6e2bf9ed6ad"
)
HISTORICAL_MATRIX_PATH = (
    "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml"
)
HISTORICAL_MATRIX_SHA256 = "03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c"
HYBRID_SLOTS = frozenset(
    {
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield",
        "scenario_adaptive_hybrid_orca_v2_collision_guard",
        "hybrid_rule_v3_fast_progress_static_escape",
        "hybrid_rule_v3_fast_progress_static_escape_continuous",
    }
)
ADAPTIVE_HYBRID_SLOTS = frozenset(
    old for old in HYBRID_SLOTS if old.startswith("scenario_adaptive_hybrid_orca_v2_")
)
V4_BASE_CONFIG_PATH = "configs/algos/hybrid_rule_v4_clearance_braking.yaml"
V4_PLANNER_VARIANT = "hybrid_rule_v4_clearance_braking"
APPROVED_ORCA_HANDOFF_SCENARIO = "francis2023_leave_group"
APPROVED_ORCA_CONFIG_PATH = "configs/algos/issue707_orca_tuned.yaml"
TOLERANCE = Decimal("1e-12")
RATE_FIELDS = (
    ("success_rate", "route_complete", "descending"),
    ("collision_rate", "collision_event", "ascending"),
    ("timeout_rate", "timeout_event", "ascending"),
)
LEGACY_NONFINITE_METRIC_FIELDS = frozenset(
    {
        "metrics.min_separation_corrupted_m",
        "metric_values.min_predicted_separation_m",
        "metrics.metric_values.min_predicted_separation_m",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _is_sha(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _bound_file(source_root: Path, relative: Any, expected_sha: Any, *, label: str) -> str:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} needs a relative path")
    path = (source_root / relative).resolve()
    if not path.is_relative_to(source_root.resolve()) or not path.is_file():
        raise ValueError(f"{label} path is missing or escapes candidate source root")
    tracked = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "--error-unmatch", "--", relative],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError(f"{label} is not tracked by candidate source commit")
    if not _is_sha(expected_sha, 64) or _sha256(path) != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch")
    return expected_sha


def _source_head(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    if dirty.stdout.strip():
        raise ValueError("candidate source checkout has tracked edits")
    return result.stdout.strip()


def load_candidate_identity(  # noqa: C901, PLR0912 - keep identity gates in one audit
    path: Path, source_root: Path
) -> dict[str, Any]:
    """Validate the candidate's source/input identity and 14 mapped slots.

    Returns:
        The checked candidate identity document.
    """
    identity = _json_object(path)
    if identity.get("schema_version") != CANDIDATE_SCHEMA or identity.get("release") != "0.0.8":
        raise ValueError("candidate comparison identity schema/release mismatch")
    source_sha = identity.get("source_sha")
    if not _is_sha(source_sha, 40) or _source_head(source_root) != source_sha:
        raise ValueError("candidate source commit does not match clean checkout")
    _bound_file(
        source_root,
        identity.get("effective_config_path"),
        identity.get("effective_config_sha256"),
        label="candidate effective config",
    )
    matrix = identity.get("scenario_matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("candidate scenario_matrix is missing")
    _bound_file(source_root, matrix.get("path"), matrix.get("sha256"), label="candidate matrix")
    slots = identity.get("arm_slots")
    if not isinstance(slots, list) or len(slots) != len(EXPECTED_ARM_KEYS):
        raise ValueError("candidate must map exactly 14 arm slots")
    old_keys: set[str] = set()
    new_keys: set[str] = set()
    for slot in slots:
        if not isinstance(slot, Mapping):
            raise ValueError("arm slot must be an object")
        old_key, new_key = slot.get("old_key"), slot.get("new_key")
        if not isinstance(old_key, str) or not isinstance(new_key, str) or not new_key:
            raise ValueError("arm slot needs old_key/new_key")
        if old_key in old_keys or new_key in new_keys:
            raise ValueError("duplicate historical slot or candidate arm key")
        old_keys.add(old_key)
        new_keys.add(new_key)
        replaced = slot.get("implementation_replaced")
        if type(replaced) is not bool:
            raise ValueError(f"{old_key}: implementation_replaced must be boolean")
        if old_key in HYBRID_SLOTS:
            if replaced != (new_key != old_key) or (replaced and "v4" not in new_key):
                raise ValueError(f"{old_key}: v4 replacement must use a new v4-named key")
        elif replaced or new_key != old_key:
            raise ValueError(f"{old_key}: only the four hybrid slots may be replaced")
        if (
            not isinstance(slot.get("implementation_version"), str)
            or not slot["implementation_version"]
        ):
            raise ValueError(f"{old_key}: implementation_version is missing")
        row_algos = slot.get("row_algos")
        if (
            not isinstance(row_algos, list)
            or not row_algos
            or any(not isinstance(item, str) or not item for item in row_algos)
            or len(set(row_algos)) != len(row_algos)
        ):
            raise ValueError(f"{old_key}: row_algos must bind runtime implementation keys")
        if old_key not in HYBRID_SLOTS and row_algos != [new_key]:
            raise ValueError(f"{old_key}: non-hybrid row_algos must match new arm key")
        if slot.get("config_path") is None:
            if slot.get("config_sha256") is not None:
                raise ValueError(f"{old_key}: config SHA requires a config path")
        else:
            _bound_file(
                source_root,
                slot.get("config_path"),
                slot.get("config_sha256"),
                label=f"{old_key} arm config",
            )
    if old_keys != EXPECTED_ARM_KEYS:
        raise ValueError("candidate slot map differs from accepted 0.0.7 arm roster")
    if identity.get("scenario_ids") != sorted(EXPECTED_SCENARIO_IDS):
        raise ValueError("candidate must bind the 48 canonical scenario identities")
    if identity.get("seeds") != list(EXPECTED_SEEDS):
        raise ValueError("candidate must bind seeds 111–140")
    if not isinstance(identity.get("episode_files"), Mapping) or not identity["episode_files"]:
        raise ValueError("candidate episode_files checksums are missing")
    _validate_changes(identity, source_root)
    identity["_effective_algorithms"] = _validate_effective_algorithms(identity, source_root)
    return identity


def _validate_changes(  # noqa: C901 - independent change-custody checks
    identity: Mapping[str, Any], source_root: Path
) -> None:
    changes = identity.get("versioned_changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("candidate needs named versioned_changes")
    bound_identities = {
        identity["source_sha"],
        identity["effective_config_sha256"],
        identity["scenario_matrix"]["sha256"],
        *(slot["config_sha256"] for slot in identity["arm_slots"] if slot.get("config_sha256")),
    }
    inputs = identity.get("versioned_inputs", [])
    if not isinstance(inputs, list):
        raise ValueError("versioned_inputs must be a list")
    for item in inputs:
        if not isinstance(item, Mapping):
            raise ValueError("versioned input must be an object")
        bound_identities.add(
            _bound_file(source_root, item.get("path"), item.get("sha256"), label="versioned input")
        )
    seen: set[str] = set()
    for change in changes:
        if not isinstance(change, Mapping):
            raise ValueError("versioned change must be an object")
        change_id = change.get("id")
        if not isinstance(change_id, str) or not change_id or change_id in seen:
            raise ValueError("versioned change IDs must be unique and nonempty")
        seen.add(change_id)
        kind = change.get("kind")
        if not isinstance(kind, str) or kind not in {"source", "config", "map", "model", "planner"}:
            raise ValueError(f"{change_id}: unsupported change kind")
        if not isinstance(change.get("version"), str) or not change["version"]:
            raise ValueError(f"{change_id}: version is missing")
        if not isinstance(change.get("old_identity"), str) or not change["old_identity"]:
            raise ValueError(f"{change_id}: old_identity is missing")
        new_identity = change.get("new_identity")
        if (
            not isinstance(new_identity, str)
            or new_identity not in bound_identities
            or new_identity == change["old_identity"]
        ):
            raise ValueError(f"{change_id}: new identity is unbound or unchanged")


def _yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    return data


def _validate_effective_algorithms(  # noqa: C901, PLR0912, PLR0915 - source-bound dispatch gate
    identity: Mapping[str, Any], source_root: Path
) -> dict[str, dict[str, str]]:
    """Resolve every 0.0.8 arm/scenario algorithm from the bound campaign config."""
    campaign = _yaml_mapping(
        source_root / identity["effective_config_path"], label="candidate campaign config"
    )
    raw_planners = campaign.get("planners")
    if not isinstance(raw_planners, list) or len(raw_planners) != len(EXPECTED_ARM_KEYS):
        raise ValueError("candidate campaign config must declare exactly 14 planners")
    planners: dict[str, Mapping[str, Any]] = {}
    for planner in raw_planners:
        if not isinstance(planner, Mapping) or not isinstance(planner.get("key"), str):
            raise ValueError("candidate campaign planner entry is malformed")
        key = planner["key"]
        if key in planners:
            raise ValueError(f"duplicate candidate campaign planner key {key}")
        planners[key] = planner
    slots = identity["arm_slots"]
    if set(planners) != {slot["new_key"] for slot in slots}:
        raise ValueError("candidate campaign planner roster differs from 14 mapped slots")

    expected: dict[str, dict[str, str]] = {}
    for slot in slots:
        old_key, new_key = slot["old_key"], slot["new_key"]
        planner = planners[new_key]
        default_algo = planner.get("algo")
        if not isinstance(default_algo, str) or not default_algo:
            raise ValueError(f"{new_key}: campaign planner algo is missing")
        config_path = slot.get("config_path")
        if planner.get("algo_config") != config_path:
            raise ValueError(f"{new_key}: campaign algo_config differs from bound slot config")
        if old_key not in HYBRID_SLOTS:
            if default_algo != new_key or set(slot["row_algos"]) != {default_algo}:
                raise ValueError(f"{new_key}: non-hybrid runtime algorithm differs from arm key")
            expected[new_key] = dict.fromkeys(identity["scenario_ids"], default_algo)
            continue
        if default_algo != "hybrid_rule_local_planner" or not isinstance(config_path, str):
            raise ValueError(f"{new_key}: hybrid campaign must use a candidate manifest")
        candidate = _yaml_mapping(source_root / config_path, label=f"{new_key} candidate config")
        if candidate.get("algo") != default_algo or not str(candidate.get("name", "")).startswith(
            new_key
        ):
            raise ValueError(f"{new_key}: candidate manifest name/algorithm mismatch")
        base_path = candidate.get("base_config_path")
        if not isinstance(base_path, str):
            raise ValueError(f"{new_key}: candidate base_config_path is missing")
        if slot["implementation_replaced"] and base_path != V4_BASE_CONFIG_PATH:
            raise ValueError(f"{new_key}: v4 replacement does not use the approved v4 base")
        base_sha = slot.get("base_config_sha256")
        _bound_file(source_root, base_path, base_sha, label=f"{new_key} base config")
        base_config = _yaml_mapping(source_root / base_path, label=f"{new_key} base config")
        if (
            slot["implementation_replaced"]
            and base_config.get("planner_variant") != V4_PLANNER_VARIANT
        ):
            raise ValueError(f"{new_key}: effective base planner variant is not v4")
        overrides = candidate.get("scenario_algo_overrides") or {}
        approved_scenarios = (
            {APPROVED_ORCA_HANDOFF_SCENARIO} if old_key in ADAPTIVE_HYBRID_SLOTS else set()
        )
        if not isinstance(overrides, Mapping) or set(overrides) != approved_scenarios:
            raise ValueError(f"{new_key}: unapproved scenario algorithm override")
        config_cache = {base_path: base_config}
        if approved_scenarios:
            handoff = overrides[APPROVED_ORCA_HANDOFF_SCENARIO]
            if (
                not isinstance(handoff, Mapping)
                or handoff.get("algo") != "orca"
                or handoff.get("base_config_path") != APPROVED_ORCA_CONFIG_PATH
            ):
                raise ValueError(f"{new_key}: ORCA hand-off differs from approved scenario")
            _bound_file(
                source_root,
                APPROVED_ORCA_CONFIG_PATH,
                slot.get("handoff_config_sha256"),
                label=f"{new_key} ORCA hand-off config",
            )
            config_cache[APPROVED_ORCA_CONFIG_PATH] = _yaml_mapping(
                source_root / APPROVED_ORCA_CONFIG_PATH, label=f"{new_key} ORCA hand-off config"
            )

        def load_config(raw_path: object) -> dict[str, Any]:
            if not isinstance(raw_path, str) or raw_path not in config_cache:
                raise ValueError(f"{new_key}: policy resolver requested unbound base config")
            return dict(config_cache[raw_path])

        by_scenario: dict[str, str] = {}
        for scenario in identity["scenario_ids"]:
            effective_algo, effective_config = resolve_candidate_manifest_runtime(
                default_algo=default_algo,
                manifest=candidate,
                scenario={"name": scenario},
                load_config=load_config,
            )
            if effective_algo not in {"hybrid_rule_local_planner", "orca"}:
                raise ValueError(f"{new_key}/{scenario}: unsupported effective algorithm")
            if effective_algo == "orca" and scenario not in approved_scenarios:
                raise ValueError(f"{new_key}/{scenario}: unapproved ORCA hand-off")
            if (
                slot["implementation_replaced"]
                and effective_algo == "hybrid_rule_local_planner"
                and effective_config.get("planner_variant") != V4_PLANNER_VARIANT
            ):
                raise ValueError(f"{new_key}/{scenario}: effective planner variant is not v4")
            by_scenario[scenario] = effective_algo
        if set(slot["row_algos"]) != set(by_scenario.values()):
            raise ValueError(
                f"{new_key}: row_algos differs from source-resolved scenario algorithms"
            )
        expected[new_key] = by_scenario
    return expected


def _compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "scenario_id",
            "seed",
            "algo",
            "outcome",
            "metrics",
            "metric_values",
            "status",
            "termination_reason",
            "steps",
            "git_hash",
            "result_provenance",
            "algorithm_metadata",
            "spawn_validity",
            "scenario_params",
            "integrity",
            "planner_runtime",
            "readiness_status",
            "row_status",
            "availability_status",
            "fallback_used",
            "degraded",
            "fallback",
            "fallback_triggered",
            "fallback_or_degraded",
        )
        if key in row
    }


def _candidate_execution_issues(row: Mapping[str, Any], *, expected_algo: str) -> list[str]:
    """Audit candidate execution eligibility on the raw row before compaction."""
    issues = [
        f"{path.removeprefix('candidate_row.')}={marker}"
        for path, marker in _status_markers(row, "candidate_row", expected_algorithm=expected_algo)
    ]
    issues.extend(_execution_audit(row, expected_algorithm=expected_algo))
    integrity = row.get("integrity")
    if not isinstance(integrity, Mapping):
        issues.append("integrity block missing or malformed")
    elif integrity.get("contradictions") != []:
        issues.append("integrity.contradictions must be an empty list")
    runtime = row.get("planner_runtime")
    if runtime is not None and not isinstance(runtime, Mapping):
        issues.append("planner_runtime must be an object when present")
    validity = row.get("spawn_validity")
    if isinstance(validity, Mapping) and validity.get("invalid_run") is True:
        issues.append("spawn_validity.invalid_run=true")
    return list(dict.fromkeys(issues))


def _read_jsonl_stream(  # noqa: C901 - all raw-row gates share one read
    lines: Any,
    *,
    arm: str,
    expected_algos: set[str],
    effective_algorithms: Mapping[str, str] | None = None,
    source: str,
    expected_commit: str,
    rows: dict[tuple[str, str, int], dict[str, Any]],
    anomalies: list[dict[str, Any]],
) -> None:
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            # The accepted 0.0.7 archive contains legacy Infinity/NaN metric
            # sentinels. Preserve and classify them at the metric-field gate.
            row = json.loads(raw)
        except (ValueError, TypeError) as exc:
            anomalies.append(
                {
                    "kind": "malformed_json",
                    "source": source,
                    "line": line_number,
                    "detail": str(exc),
                }
            )
            continue
        if not isinstance(row, Mapping):
            anomalies.append({"kind": "malformed_row", "source": source, "line": line_number})
            continue
        scenario, seed = row.get("scenario_id"), row.get("seed")
        if not isinstance(scenario, str) or not scenario or type(seed) is not int:
            anomalies.append({"kind": "invalid_identity", "source": source, "line": line_number})
            continue
        key = (arm, scenario, seed)
        if key in rows:
            anomalies.append(
                {
                    "kind": "duplicate_identity",
                    "source": source,
                    "line": line_number,
                    "identity": list(key),
                }
            )
            continue
        provenance = row.get("result_provenance")
        provenance_sha = provenance.get("repo_commit") if isinstance(provenance, Mapping) else None
        git_sha = row.get("git_hash")
        if provenance_sha and git_sha and provenance_sha != git_sha:
            anomalies.append(
                {
                    "kind": "conflicting_source",
                    "source": source,
                    "line": line_number,
                    "identity": list(key),
                }
            )
        commit = provenance_sha or git_sha
        if commit != expected_commit:
            anomalies.append(
                {
                    "kind": "source_mismatch",
                    "source": source,
                    "line": line_number,
                    "identity": list(key),
                }
            )
        row_algo = row.get("algo")
        expected_scenario_algo = (
            effective_algorithms.get(scenario) if effective_algorithms is not None else None
        )
        if (
            not isinstance(row_algo, str)
            or row_algo not in expected_algos
            or (effective_algorithms is not None and row_algo != expected_scenario_algo)
        ):
            anomalies.append(
                {
                    "kind": "arm_mismatch",
                    "source": source,
                    "line": line_number,
                    "identity": list(key),
                }
            )
        if effective_algorithms is not None:
            expected_algo = expected_scenario_algo or (
                row_algo if isinstance(row_algo, str) else arm
            )
            issues = _candidate_execution_issues(row, expected_algo=expected_algo)
            if issues:
                anomalies.append(
                    {
                        "kind": "ineligible_candidate_row",
                        "source": source,
                        "line": line_number,
                        "identity": list(key),
                        "reasons": issues,
                    }
                )
        rows[key] = _compact_row(row)


def read_historical_bundle(  # noqa: C901 - archive identity checks stay together
    path: Path,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[dict[str, Any]]]:
    """Read only the accepted 0.0.7 archive members after exact SHA binding.

    Returns:
        Compact episode rows and malformed/duplicate/source findings.
    """
    if _sha256(path) != HISTORICAL_BUNDLE_SHA256:
        raise ValueError("historical bundle differs from accepted 0.0.7 SHA-256")
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    anomalies: list[dict[str, Any]] = []
    with tarfile.open(path, "r:gz") as archive:
        manifests = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and member.name.endswith("/payload/release/release_manifest.resolved.json")
        ]
        if len(manifests) != 1:
            raise ValueError("accepted bundle must have one resolved release manifest")
        stream = archive.extractfile(manifests[0])
        if stream is None:
            raise ValueError("cannot read historical resolved manifest")
        manifest = json.load(stream)
        if not isinstance(manifest, Mapping):
            raise ValueError("historical resolved manifest must be an object")
        scenario = manifest.get("scenario")
        if (
            (manifest.get("source_sha") or manifest.get("source_commit")) != HISTORICAL_SOURCE_SHA
            or not isinstance(scenario, Mapping)
            or scenario.get("matrix_path") != HISTORICAL_MATRIX_PATH
            or str(scenario.get("matrix_sha256", "")).removeprefix("sha256:")
            != HISTORICAL_MATRIX_SHA256
        ):
            raise ValueError("accepted bundle source/v1 matrix identity mismatch")
        config_sha = manifest.get("canonical_campaign_config_sha256")
        if str(config_sha or "").removeprefix("sha256:") != HISTORICAL_EFFECTIVE_CONFIG_SHA256:
            raise ValueError("accepted bundle effective config identity mismatch")
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and member.name.endswith("/episodes.jsonl")
            and "/payload/runs/" in member.name
        ]
        if not members:
            raise ValueError("accepted bundle has no payload run rows")
        seen_arms: set[str] = set()
        for member in sorted(members, key=lambda item: item.name):
            arm = Path(member.name).parent.name.removesuffix("__differential_drive")
            if arm in seen_arms:
                anomalies.append({"kind": "duplicate_arm_file", "source": member.name, "arm": arm})
            seen_arms.add(arm)
            stream = archive.extractfile(member)
            if stream is None:
                anomalies.append({"kind": "unreadable_arm_file", "source": member.name})
                continue
            _read_jsonl_stream(
                stream,
                arm=arm,
                expected_algos=(
                    {"hybrid_rule_local_planner", "orca"}
                    if arm.startswith("scenario_adaptive_hybrid_orca_v2_")
                    else ({"hybrid_rule_local_planner"} if arm in HYBRID_SLOTS else {arm})
                ),
                source=member.name,
                expected_commit=HISTORICAL_SOURCE_SHA,
                rows=rows,
                anomalies=anomalies,
            )
    return rows, anomalies


def read_candidate_rows(
    root: Path, identity: Mapping[str, Any]
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[dict[str, Any]]]:
    """Check candidate row-byte custody and read available compact rows.

    Returns:
        Candidate rows and file/row anomalies; anomalies always block admission.
    """
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    anomalies: list[dict[str, Any]] = []
    paths = sorted((root / "runs").glob("*/episodes.jsonl"))
    expected = identity["episode_files"]
    observed = {path.relative_to(root).as_posix(): path for path in paths}
    row_algos = {slot["new_key"]: set(slot["row_algos"]) for slot in identity["arm_slots"]}
    effective = identity["_effective_algorithms"]
    for missing in sorted(set(expected) - set(observed)):
        anomalies.append({"kind": "missing_episode_file", "source": missing})
    for extra in sorted(set(observed) - set(expected)):
        anomalies.append({"kind": "extra_episode_file", "source": extra})
    for relative, path in sorted(observed.items()):
        if not _is_sha(expected.get(relative), 64) or _sha256(path) != expected[relative]:
            anomalies.append({"kind": "episode_file_sha_mismatch", "source": relative})
        arm = path.parent.name.removesuffix("__differential_drive")
        with path.open("rb") as stream:
            _read_jsonl_stream(
                stream,
                arm=arm,
                expected_algos=row_algos.get(arm, {arm}),
                effective_algorithms=effective.get(arm, {}),
                source=relative,
                expected_commit=identity["source_sha"],
                rows=rows,
                anomalies=anomalies,
            )
    return rows, anomalies


def _flatten(value: Mapping[str, Any] | list[Any], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    items = enumerate(value) if isinstance(value, list) else value.items()
    for key, item in items:
        name = f"{prefix}.{key}"
        if isinstance(item, (Mapping, list)):
            result.update(_flatten(item, name))
        else:
            result[name] = item
    return result


def _numeric(value: Any) -> Decimal | None:
    if type(value) not in (int, float, Decimal):
        return None
    try:
        number = Decimal(str(value))
    except ArithmeticError:
        return None
    return number if number.is_finite() else None


def _json_safe(value: Any) -> Any:
    if type(value) is float and not math.isfinite(value):
        token = "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
        return {"nonfinite": token}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _different(old: Any, new: Any) -> bool:
    old_number, new_number = _numeric(old), _numeric(new)
    if old_number is not None and new_number is not None:
        return abs(old_number - new_number) > TOLERANCE
    return type(old) is not type(new) or _json_safe(old) != _json_safe(new)


def _finding(
    slot: str, scenario: str, seed: int, field: str, old: Any, new: Any, replaced: bool
) -> dict[str, Any]:
    identity = [slot, scenario, seed]
    old, new = _json_safe(old), _json_safe(new)
    finding_id = _digest([identity, field, old, new])
    return {
        "finding_id": finding_id,
        "identity": identity,
        "field": field,
        "old": old,
        "new": new,
        "comparison": "implementation replaced" if replaced else "paired correction",
    }


def _row_fields(row: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    fields: dict[str, Any] = {}
    anomalies: list[str] = []
    outcome = row.get("outcome")
    if not isinstance(outcome, Mapping):
        anomalies.append("outcome_missing")
    else:
        fields.update(_flatten(outcome, "outcome"))
        for key in ("route_complete", "collision_event", "timeout_event"):
            if type(outcome.get(key)) is not bool:
                anomalies.append(f"outcome_{key}_invalid")
    for group in ("metrics", "metric_values"):
        value = row.get(group)
        if not isinstance(value, Mapping):
            anomalies.append(f"{group}_missing")
        else:
            fields.update(_flatten(value, group))
    for name in ("status", "termination_reason", "steps"):
        if name in row:
            fields[name] = row[name]
    for name, value in fields.items():
        if (
            type(value) is float
            and not math.isfinite(value)
            and name not in LEGACY_NONFINITE_METRIC_FIELDS
        ):
            anomalies.append(f"nonfinite_{name}")
    return fields, anomalies


def _rank(values: Mapping[str, float], *, direction: str) -> dict[str, int]:
    reverse = direction == "descending"
    # Same values share a competition rank; arm-key ordering only makes output stable.
    ordered = sorted(values, key=lambda key: ((-values[key] if reverse else values[key]), key))
    rank: dict[str, int] = {}
    last: float | None = None
    for position, key in enumerate(ordered, 1):
        value = values[key]
        if last is None or value != last:
            current = position
        rank[key] = current
        last = value
    return rank


def _rate_impacts(
    counts: Mapping[str, Mapping[str, Counter[str]]], pairs: Mapping[str, int]
) -> dict[str, Any]:
    impacts: dict[str, Any] = {}
    for name, outcome_key, direction in RATE_FIELDS:
        old_values = {
            slot: counts[slot]["old"][outcome_key] / pairs[slot] for slot in pairs if pairs[slot]
        }
        new_values = {
            slot: counts[slot]["new"][outcome_key] / pairs[slot] for slot in pairs if pairs[slot]
        }
        old_rank = _rank(old_values, direction=direction)
        new_rank = _rank(new_values, direction=direction)
        impacts[name] = {
            slot: {
                "paired_denominator": pairs[slot],
                "old": old_values[slot],
                "new": new_values[slot],
                "delta": new_values[slot] - old_values[slot],
                "old_rank": old_rank[slot],
                "new_rank": new_rank[slot],
                "rank_shift": new_rank[slot] - old_rank[slot],
            }
            for slot in old_values
        }
    return impacts


def _snqi_impacts(
    sums: Mapping[str, Mapping[str, Decimal]], counts: Mapping[str, int]
) -> dict[str, Any]:
    old_values = {
        slot: float(values["old"] / counts[slot]) for slot, values in sums.items() if counts[slot]
    }
    new_values = {
        slot: float(values["new"] / counts[slot]) for slot, values in sums.items() if counts[slot]
    }
    old_rank = _rank(old_values, direction="descending")
    new_rank = _rank(new_values, direction="descending")
    return {
        slot: {
            "finite_paired_denominator": counts[slot],
            "old_mean": old_values[slot],
            "new_mean": new_values[slot],
            "delta": new_values[slot] - old_values[slot],
            "old_rank": old_rank[slot],
            "new_rank": new_rank[slot],
            "rank_shift": new_rank[slot] - old_rank[slot],
        }
        for slot in old_values
    }


def compare_episode_maps(  # noqa: C901, PLR0912 - all per-pair checks share one pass
    old_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    new_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    slots: Sequence[Mapping[str, Any]],
    *,
    scenarios: Sequence[str] = tuple(sorted(EXPECTED_SCENARIO_IDS)),
    seeds: Sequence[int] = tuple(EXPECTED_SEEDS),
    emit: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Pair all available canonical identities and emit every changed common field.

    Returns:
        Diagnostic inventory and paired rate/rank effects; attribution is checked later.
    """
    mapping = {slot["old_key"]: slot for slot in slots}
    expected = {
        (arm, scenario, seed) for arm in mapping for scenario in scenarios for seed in seeds
    }
    old_present = set(old_rows)
    new_inverse = {slot["new_key"]: old for old, slot in mapping.items()}
    new_mapped = {
        (new_inverse[arm], scenario, seed) for arm, scenario, seed in new_rows if arm in new_inverse
    }
    inventory = {
        "missing_old": sorted(map(list, expected - old_present)),
        "missing_new": sorted(map(list, expected - new_mapped)),
        "extra_old": sorted(map(list, old_present - expected)),
        "extra_new": sorted(
            [list(key) for key in new_rows if key[0] not in new_inverse]
            + [
                [new_inverse[arm], scenario, seed]
                for arm, scenario, seed in new_rows
                if arm in new_inverse and (new_inverse[arm], scenario, seed) not in expected
            ]
        ),
    }
    anomalies: list[dict[str, Any]] = []
    counts: dict[str, dict[str, Counter[str]]] = {
        arm: {"old": Counter(), "new": Counter()} for arm in mapping
    }
    snqi_sums = {arm: {"old": Decimal(0), "new": Decimal(0)} for arm in mapping}
    snqi_counts: Counter[str] = Counter()
    nonfinite_common: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    all_pair_count = 0
    changed = 0
    changed_episodes: set[tuple[str, str, int]] = set()
    # Compare mapped extras too: they block the matrix but still receive an
    # outcome/metric diff. Rate denominators remain canonical-matrix-only.
    pair_keys = {key for key in old_present & new_mapped if key[0] in mapping}
    for old_key, scenario, seed in sorted(pair_keys):
        canonical = (old_key, scenario, seed) in expected
        slot = mapping[old_key]
        old = old_rows[(old_key, scenario, seed)]
        new = new_rows[(slot["new_key"], scenario, seed)]
        identity = [old_key, scenario, seed]
        for side, row in (("old", old), ("new", new)):
            _, problems = _row_fields(row)
            for problem in problems:
                anomalies.append(
                    {
                        "kind": "invalid_row_field",
                        "side": side,
                        "identity": identity,
                        "detail": problem,
                    }
                )
            if side == "new":
                problems = _candidate_execution_issues(
                    row, expected_algo=str(row.get("algo") or slot["new_key"])
                )
                if problems:
                    anomalies.append(
                        {
                            "kind": "degraded_candidate_row",
                            "identity": identity,
                            "reasons": problems,
                        }
                    )
            outcome = row.get("outcome")
            if canonical and isinstance(outcome, Mapping):
                for _, outcome_key, _ in RATE_FIELDS:
                    if type(outcome.get(outcome_key)) is bool:
                        counts[old_key][side][outcome_key] += int(outcome[outcome_key])
        old_fields, _ = _row_fields(old)
        new_fields, _ = _row_fields(new)
        old_snqi = _numeric(old_fields.get("metrics.snqi"))
        new_snqi = _numeric(new_fields.get("metrics.snqi"))
        if old_snqi is None or new_snqi is None:
            anomalies.append({"kind": "snqi_not_finite_or_missing", "identity": identity})
        elif canonical:
            snqi_sums[old_key]["old"] += old_snqi
            snqi_sums[old_key]["new"] += new_snqi
            snqi_counts[old_key] += 1
        for field in sorted(old_fields.keys() - new_fields.keys()):
            anomalies.append({"kind": "missing_common_field", "identity": identity, "field": field})
        for field in sorted(new_fields.keys() - old_fields.keys()):
            anomalies.append({"kind": "added_field", "identity": identity, "field": field})
        for field in sorted(old_fields.keys() & new_fields.keys()):
            if _numeric(old_fields[field]) is None and type(old_fields[field]) is float:
                nonfinite_common[f"old:{field}"] += 1
            if _numeric(new_fields[field]) is None and type(new_fields[field]) is float:
                nonfinite_common[f"new:{field}"] += 1
            if _different(old_fields[field], new_fields[field]):
                emit(
                    _finding(
                        old_key,
                        scenario,
                        seed,
                        field,
                        old_fields[field],
                        new_fields[field],
                        slot["implementation_replaced"],
                    )
                )
                changed += 1
                changed_episodes.add((old_key, scenario, seed))
        all_pair_count += 1
        if canonical:
            pairs[old_key] += 1
    return {
        "inventory": inventory,
        "row_anomalies": anomalies,
        "paired_rows": all_pair_count,
        "paired_contract_rows": sum(pairs.values()),
        "pairs_by_arm": dict(sorted(pairs.items())),
        "changed_field_count": changed,
        "changed_episode_count": len(changed_episodes),
        "rate_impacts": _rate_impacts(counts, pairs),
        "snqi_mean_impacts": _snqi_impacts(snqi_sums, snqi_counts),
        "nonfinite_common_metric_counts": dict(sorted(nonfinite_common.items())),
    }


def _receipt_path(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} must have a relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"{label} is missing or escapes receipt root")
    return path


def load_attribution_ledger(  # noqa: C901 - independent receipt checks stay together
    path: Path, identity: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], str]:
    """Check every ledger entry's declared versioned change and causal receipt.

    Returns:
        Entries keyed by finding ID, ledger anomalies, and ledger-byte SHA-256.
    """
    ledger = _json_object(path)
    if ledger.get("schema_version") != ATTRIBUTION_SCHEMA:
        raise ValueError("attribution ledger schema mismatch")
    if ledger.get("candidate_source_sha") != identity["source_sha"]:
        raise ValueError("attribution ledger candidate source mismatch")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("attribution ledger entries must be a list")
    declared = {change["id"] for change in identity["versioned_changes"]}
    indexed: dict[str, dict[str, Any]] = {}
    anomalies: list[dict[str, Any]] = []
    receipt_cache: dict[str, Mapping[str, Any]] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            anomalies.append({"kind": "invalid_attribution_entry", "index": position})
            continue
        finding_id, change_id = entry.get("finding_id"), entry.get("change_id")
        if not _is_sha(finding_id, 64) or finding_id in indexed:
            anomalies.append({"kind": "invalid_or_duplicate_finding_id", "index": position})
            continue
        indexed[finding_id] = dict(entry)
        if not isinstance(change_id, str) or change_id not in declared:
            anomalies.append({"kind": "undeclared_versioned_change", "finding_id": finding_id})
            continue
        relative, expected_sha = entry.get("receipt_path"), entry.get("receipt_sha256")
        try:
            receipt_path = _receipt_path(path.parent, relative, label="causal receipt")
            if not _is_sha(expected_sha, 64) or _sha256(receipt_path) != expected_sha:
                raise ValueError("causal receipt SHA-256 mismatch")
            if relative not in receipt_cache:
                receipt_cache[relative] = _json_object(receipt_path)
            receipt = receipt_cache[relative]
            finding_ids = receipt.get("finding_ids")
            if (
                receipt.get("schema_version") != RECEIPT_SCHEMA
                or receipt.get("change_id") != change_id
                or not isinstance(finding_ids, list)
                or not finding_ids
                or any(not _is_sha(item, 64) for item in finding_ids)
                or len(set(finding_ids)) != len(finding_ids)
                or finding_id not in finding_ids
                or not isinstance(receipt.get("mechanism"), str)
                or not receipt["mechanism"].strip()
            ):
                raise ValueError("causal receipt does not bind finding/change/mechanism")
            evidence_sha = receipt.get("evidence_sha256")
            evidence = _receipt_path(
                path.parent, receipt.get("evidence_path"), label="causal evidence"
            )
            if not _is_sha(evidence_sha, 64) or _sha256(evidence) != evidence_sha:
                raise ValueError("causal evidence SHA-256 mismatch")
            review = receipt.get("review")
            if (
                not isinstance(review, Mapping)
                or review.get("decision") != "accepted"
                or not all(
                    isinstance(review.get(key), str) and review[key].strip()
                    for key in ("reviewer", "reviewed_at_utc")
                )
            ):
                raise ValueError("causal receipt lacks explicit accepted scientific review")
        except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            anomalies.append(
                {"kind": "invalid_causal_receipt", "finding_id": finding_id, "detail": str(exc)}
            )
    return indexed, anomalies, _sha256(path)


def compare_with_attribution(
    old_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    new_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    slots: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Mapping[str, Any]],
    *,
    scenarios: Sequence[str] = tuple(sorted(EXPECTED_SCENARIO_IDS)),
    seeds: Sequence[int] = tuple(EXPECTED_SEEDS),
    emit: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Check field-level findings against one attributed entry each.

    Returns:
        Full diagnostic summary; ``comparison_passed`` is structural only.
    """
    observed: set[str] = set()
    unexplained: list[str] = []

    def record(finding: dict[str, Any]) -> None:
        finding_id = finding["finding_id"]
        observed.add(finding_id)
        attribution = ledger.get(finding_id)
        finding["attribution"] = (
            {"change_id": attribution["change_id"], "receipt_sha256": attribution["receipt_sha256"]}
            if attribution is not None
            else None
        )
        if attribution is None:
            unexplained.append(finding_id)
        emit(finding)

    summary = compare_episode_maps(
        old_rows, new_rows, slots, scenarios=scenarios, seeds=seeds, emit=record
    )
    orphaned = sorted(set(ledger) - observed)
    summary["unexplained_findings"] = unexplained
    summary["orphaned_attributions"] = orphaned
    summary["comparison_passed"] = (
        not any(summary["inventory"].values())
        and not summary["row_anomalies"]
        and not unexplained
        and not orphaned
    )
    return summary


def _render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["comparison"]
    lines = [
        "# 0.0.7 → 0.0.8 episode comparison",
        "",
        f"Status: **{'structurally passed' if report['comparison_passed'] else 'blocked'}**. "
        "Scientific receipt review and the other release gates remain separate.",
        "",
        f"Historical bundle SHA-256: `{HISTORICAL_BUNDLE_SHA256}`; executed v1 matrix "
        f"SHA-256: `{HISTORICAL_MATRIX_SHA256}`.",
        f"Candidate source: `{report['candidate_source_sha']}`; identity SHA-256: "
        f"`{report['candidate_identity_sha256']}`.",
        f"Paired rows: **{summary['paired_rows']}**; changed episodes: "
        f"**{summary['changed_episode_count']}**; changed common fields: "
        f"**{summary['changed_field_count']}**.",
        f"Unexplained findings: **{len(summary['unexplained_findings'])}**; "
        f"orphaned attributions: **{len(summary['orphaned_attributions'])}**.",
        "",
        "## Inventory and gate findings",
        "",
    ]
    for key, value in summary["inventory"].items():
        lines.append(f"- {key}: {len(value)}")
    lines.extend(
        [
            f"- row anomalies: {len(summary['row_anomalies'])}",
            f"- read anomalies: {len(report['read_anomalies'])}",
            f"- attribution anomalies: {len(report['attribution_anomalies'])}",
            "",
            "## Rate and ranking effects on paired identities",
            "",
            "Rates below use each arm's available paired rows. Partial inventories are diagnostic only.",
            "",
            "| Rate | Historical slot | N | Old | New | Δ | Rank old→new |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rate, arms in summary["rate_impacts"].items():
        for slot, item in sorted(arms.items()):
            lines.append(
                f"| {rate} | `{slot}` | {item['paired_denominator']} | {item['old']:.6f} | "
                f"{item['new']:.6f} | {item['delta']:+.6f} | "
                f"{item['old_rank']}→{item['new_rank']} |"
            )
    lines.extend(
        [
            "",
            "SNQI mean and rank (higher is better; finite paired values only):",
            "",
            "| Historical slot | N | Old mean | New mean | Δ | Rank old→new |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for slot, item in sorted(summary["snqi_mean_impacts"].items()):
        lines.append(
            f"| `{slot}` | {item['finite_paired_denominator']} | {item['old_mean']:.6f} | "
            f"{item['new_mean']:.6f} | {item['delta']:+.6f} | "
            f"{item['old_rank']}→{item['new_rank']} |"
        )
    lines.extend(
        [
            "",
            "Collision rates use observed `outcome.collision_event`, not a planner-caused "
            "contact attribution. Legacy non-finite metric sentinels are counted in JSON; "
            "they are not treated as numeric equality evidence.",
            "",
            "Field-level old/new values, v4 implementation labels, and attribution "
            "identities are in the checksummed findings JSONL. A causal receipt's accepted "
            "decision is a recorded claim; this tool verifies its binding and bytes, "
            "not the scientific truth of its mechanism.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bound comparison and write diagnostic artifacts; return 2 on a blocked gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-bundle", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-source-root", type=Path, required=True)
    parser.add_argument("--candidate-identity", type=Path, required=True)
    parser.add_argument("--attribution-ledger", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--findings-jsonl", type=Path, required=True)
    parser.add_argument("--report-md", type=Path)
    args = parser.parse_args(argv)
    identity = load_candidate_identity(args.candidate_identity, args.candidate_source_root)
    ledger, attribution_anomalies, ledger_sha = load_attribution_ledger(
        args.attribution_ledger, identity
    )
    historical, historical_anomalies = read_historical_bundle(args.historical_bundle)
    candidate, candidate_anomalies = read_candidate_rows(args.candidate_root, identity)
    args.findings_jsonl.parent.mkdir(parents=True, exist_ok=True)
    findings_tmp = args.findings_jsonl.with_suffix(args.findings_jsonl.suffix + ".tmp")
    try:
        with findings_tmp.open("w", encoding="utf-8") as handle:

            def emit(finding: dict[str, Any]) -> None:
                handle.write(json.dumps(finding, sort_keys=True, allow_nan=False) + "\n")

            comparison = compare_with_attribution(
                historical, candidate, identity["arm_slots"], ledger, emit=emit
            )
        findings_tmp.replace(args.findings_jsonl)
    finally:
        findings_tmp.unlink(missing_ok=True)
    report = {
        "schema_version": REPORT_SCHEMA,
        "historical_bundle_sha256": HISTORICAL_BUNDLE_SHA256,
        "historical_source_sha": HISTORICAL_SOURCE_SHA,
        "historical_effective_config_sha256": HISTORICAL_EFFECTIVE_CONFIG_SHA256,
        "historical_matrix_sha256": HISTORICAL_MATRIX_SHA256,
        "candidate_source_sha": identity["source_sha"],
        "candidate_identity_sha256": _sha256(args.candidate_identity),
        "attribution_ledger_sha256": ledger_sha,
        "findings_sha256": _sha256(args.findings_jsonl),
        "comparison": comparison,
        "read_anomalies": historical_anomalies + candidate_anomalies,
        "attribution_anomalies": attribution_anomalies,
        "comparison_passed": (
            comparison["comparison_passed"]
            and not historical_anomalies
            and not candidate_anomalies
            and not attribution_anomalies
        ),
        "claim_boundary": "structural comparison only; no row, release, or Chapter 7 admission",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    if args.report_md:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {"comparison_passed": report["comparison_passed"], "report": str(args.report_json)}
        )
    )
    return 0 if report["comparison_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
