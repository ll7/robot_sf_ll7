"""Compose the issue #6642 campaign arms into the Gate 3 sweep-summary contract.

The composer deliberately reads the canonical per-episode JSONL files instead of the
derived seed CSV: typed pedestrian/obstacle collision counts are present only in the
episode records.  It fails before writing output when an arm is incomplete, duplicated,
fallback/degraded, provenance-inconsistent, or lacks an authoritative family-feasibility
mapping.  In particular, outcome rates and route-clearance warnings are not silently
reinterpreted as family feasibility.
"""

from __future__ import annotations

import json
import math
import posixpath
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from robot_sf.benchmark.algorithm_metadata import canonical_algorithm_name
from robot_sf.benchmark.policy_search_manifest import resolve_candidate_manifest_runtime
from robot_sf.benchmark.radius_rank_stability import (
    SWEEP_SUMMARY_SCHEMA,
    _gate1_canary_receipt_is_passing,
)
from robot_sf.benchmark.radius_sweep_manifest import (
    EXPECTED_ARM_CAMPAIGN_CONFIG_SHA256,
    EXPECTED_ARM_CAMPAIGN_CONFIGS,
    EXPECTED_CAMPAIGN_GIT_COMMIT,
    EXPECTED_GATE1_RECEIPT_SHA256,
    EXPECTED_ROWS_PER_ARM,
    EXPECTED_SCENARIO_MATRIX,
    EXPECTED_SCENARIO_NAMES,
    EXPECTED_SEEDS,
    PRODUCTION_RADII,
    RELEASE_PLANNER_KEYS,
)
from robot_sf.benchmark.utils import _config_hash

CAMPAIGN_SCHEMA = "benchmark-camera-ready-campaign.v1"
FAMILY_FEASIBILITY_SCHEMA = "issue_6642_family_feasibility.v1"
EXPECTED_KINEMATICS = "differential_drive"
# These remain unset until a separately reviewed owner decision pins the exact
# family rule. A self-declared receipt cannot authorize its own interpretation.
EXPECTED_FAMILY_FEASIBILITY_DEFINITION_ID: str | None = None
EXPECTED_FAMILY_FEASIBILITY_AUTHORITY_SHA256: str | None = None
SOURCE_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RADIUS_TO_ARM_KEY = dict(zip(PRODUCTION_RADII, ("r0p5", "r0p8", "r1p0"), strict=True))


class RadiusSweepSummaryError(ValueError):
    """Raised when campaign artifacts cannot support a Gate 3 sweep summary."""


@dataclass(frozen=True, slots=True)
class _Episode:
    """One validated episode reduced to the fields used by the summary."""

    planner: str
    scenario: str
    seed: int
    success: float
    pedestrian_collisions: float
    obstacle_collisions: float
    typed_collisions: float
    snqi: float


@dataclass(frozen=True, slots=True)
class _PlannerIdentity:
    """Frozen algorithm and config identity expected for one campaign planner key."""

    algorithm: str
    algo_config_hash: str
    planner_key_required: bool = False


_FamilyFeasibilityEvaluator = Callable[
    [Path, float, Sequence[_Episode], str, str], tuple[str, Mapping[str, str]]
]
# No trusted evaluator is implemented in this source revision. Rule identity strings alone
# cannot admit a family receipt; a separately reviewed change must add an evaluator that
# recomputes its output from the exact admitted episode rows.
_FAMILY_FEASIBILITY_EVALUATOR: _FamilyFeasibilityEvaluator | None = None


@dataclass(frozen=True, slots=True)
class _Arm:
    """One validated radius arm and its authoritative provenance."""

    radius: float
    root: Path
    campaign_id: str
    campaign_commit: str
    config_path: str
    config_sha256: str
    gate1_receipt_sha256: str
    family_feasibility_definition_id: str
    family_feasibility_authority_sha256: str
    family_feasibility_definition: str
    family_feasibility: dict[str, str]
    family_feasibility_sha256: str
    episodes: tuple[_Episode, ...]


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RadiusSweepSummaryError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RadiusSweepSummaryError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RadiusSweepSummaryError(f"{label} must be a JSON object: {path}")
    return payload


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RadiusSweepSummaryError(f"{label} must be an object")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise RadiusSweepSummaryError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RadiusSweepSummaryError(f"{label} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise RadiusSweepSummaryError(f"{label} must be a finite number")
    return parsed


def _hex(value: object, length: int, label: str) -> str:
    text = str(value or "")
    if len(text) != length or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise RadiusSweepSummaryError(f"{label} must be a {length}-character hexadecimal digest")
    return text.lower()


def _validate_campaign_status(summary: Mapping[str, Any], radius: float) -> None:
    campaign = _mapping(summary.get("campaign"), f"radius {radius:g} campaign summary")
    row_status = _mapping(
        campaign.get("row_status_summary"), f"radius {radius:g} row status summary"
    )
    required = {
        "campaign_execution_status": "completed",
        "evidence_status": "valid",
        "benchmark_success": True,
        "total_episodes": EXPECTED_ROWS_PER_ARM,
        "total_runs": len(RELEASE_PLANNER_KEYS),
        "successful_runs": len(RELEASE_PLANNER_KEYS),
        "non_success_runs": 0,
        "unexpected_failed_runs": 0,
    }
    for field, expected in required.items():
        if campaign.get(field) != expected:
            raise RadiusSweepSummaryError(
                f"radius {radius:g} campaign {field}={campaign.get(field)!r}; expected {expected!r}"
            )
    if row_status.get("fallback_or_degraded_rows") != 0:
        raise RadiusSweepSummaryError(f"radius {radius:g} contains fallback/degraded rows")
    integrity = _mapping(summary.get("campaign_integrity"), f"radius {radius:g} integrity")
    if integrity.get("status") != "valid" or integrity.get("benchmark_success_allowed") is not True:
        raise RadiusSweepSummaryError(f"radius {radius:g} campaign integrity is not valid")


def _validate_planner_rows(
    summary: Mapping[str, Any], radius: float
) -> dict[str, Mapping[str, Any]]:
    rows = summary.get("planner_rows")
    if not isinstance(rows, list):
        raise RadiusSweepSummaryError(f"radius {radius:g} planner_rows must be a list")
    by_key: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, f"radius {radius:g} planner row")
        key = str(row.get("planner_key") or "")
        if not key or key in by_key:
            raise RadiusSweepSummaryError(f"radius {radius:g} has invalid/duplicate planner row")
        by_key[key] = row
    if set(by_key) != set(RELEASE_PLANNER_KEYS):
        raise RadiusSweepSummaryError(f"radius {radius:g} planner roster mismatch")
    for key, row in by_key.items():
        if (
            row.get("status") != "ok"
            or row.get("availability_status") != "available"
            or str(row.get("benchmark_success")).lower() != "true"
            or row.get("readiness_status") in {"fallback", "degraded"}
            or row.get("failed_jobs") != 0
            or row.get("episodes") != len(EXPECTED_SCENARIO_NAMES) * len(EXPECTED_SEEDS)
        ):
            raise RadiusSweepSummaryError(
                f"radius {radius:g} planner {key!r} is not complete benchmark evidence"
            )
    return by_key


def _camera_ready_mean(values: Sequence[float]) -> str:
    """Return the canonical camera-ready four-decimal serialization of a mean."""
    return f"{sum(values) / len(values):.4f}"


def _validate_planner_aggregates(
    planner_rows: Mapping[str, Mapping[str, Any]],
    episodes: Sequence[_Episode],
    radius: float,
) -> None:
    """Reconcile summary means against the exact admitted episode population."""
    by_planner: dict[str, list[_Episode]] = defaultdict(list)
    for episode in episodes:
        by_planner[episode.planner].append(episode)
    metric_sources = {
        "success_mean": "success",
        "ped_collision_count_mean": "pedestrian_collisions",
        "obstacle_collision_count_mean": "obstacle_collisions",
        "total_collision_count_mean": "typed_collisions",
        "snqi_mean": "snqi",
    }
    for planner in RELEASE_PLANNER_KEYS:
        row = planner_rows[planner]
        planner_episodes = by_planner[planner]
        for field, attribute in metric_sources.items():
            expected = _camera_ready_mean(
                [getattr(episode, attribute) for episode in planner_episodes]
            )
            if row.get(field) != expected:
                raise RadiusSweepSummaryError(
                    f"radius {radius:g} planner {planner!r} aggregate mismatch:{field}; "
                    f"reported={row.get(field)!r}, episode-derived={expected!r}"
                )


def _canonical_algorithm_carrier(label: str, value: object, radius: float) -> tuple[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise RadiusSweepSummaryError(
            f"radius {radius:g} episode {label} must be a non-empty algorithm identity"
        )
    return label, canonical_algorithm_name(value)


def _episode_algorithm_carriers(record: Mapping[str, Any], radius: float) -> list[tuple[str, str]]:
    carriers: list[tuple[str, str]] = []
    if "algo" in record:
        carriers.append(_canonical_algorithm_carrier("algo", record["algo"], radius))

    if "algorithm_metadata" in record:
        metadata = _mapping(record["algorithm_metadata"], "episode algorithm_metadata")
        for field in ("algorithm", "canonical_algorithm"):
            if field in metadata:
                carriers.append(
                    _canonical_algorithm_carrier(
                        f"algorithm_metadata.{field}", metadata[field], radius
                    )
                )

    if "scenario_params" in record:
        scenario_params = _mapping(record["scenario_params"], "episode scenario_params")
        if "algo" in scenario_params:
            carriers.append(
                _canonical_algorithm_carrier(
                    "scenario_params.algo", scenario_params["algo"], radius
                )
            )

    return carriers


def _episode_planner_key_carriers(
    record: Mapping[str, Any], radius: float
) -> list[tuple[str, str]]:
    carriers: list[tuple[str, str]] = []
    if "planner_key" in record:
        value = record["planner_key"]
        if not isinstance(value, str) or not value.strip():
            raise RadiusSweepSummaryError(
                f"radius {radius:g} episode planner_key must be a non-empty planner identity"
            )
        carriers.append(("planner_key", value))

    if "scenario_params" in record:
        scenario_params = _mapping(record["scenario_params"], "episode scenario_params")
        if "planner_key" in scenario_params:
            value = scenario_params["planner_key"]
            if not isinstance(value, str) or not value.strip():
                raise RadiusSweepSummaryError(
                    "radius "
                    f"{radius:g} episode scenario_params.planner_key must be a non-empty "
                    "planner identity"
                )
            carriers.append(("scenario_params.planner_key", value))

    if "result_provenance" in record:
        provenance = _mapping(record["result_provenance"], "episode result_provenance")
        if "planner_key" in provenance:
            value = provenance["planner_key"]
            if not isinstance(value, str) or not value.strip():
                raise RadiusSweepSummaryError(
                    f"radius {radius:g} episode result_provenance.planner_key must be "
                    "a non-empty planner identity"
                )
            carriers.append(("result_provenance.planner_key", value))
    return carriers


def _validate_episode_planner_identity(
    record: Mapping[str, Any],
    *,
    planner: str,
    expected: _PlannerIdentity,
    radius: float,
) -> None:
    """Bind episode algorithm, per-arm config, and any explicit planner key."""
    algorithm_carriers = _episode_algorithm_carriers(record, radius)
    if not algorithm_carriers:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} episode has no embedded planner identity carrier; "
            f"cannot validate it against directory planner {planner!r}"
        )

    distinct = {canonical for _, canonical in algorithm_carriers}
    if len(distinct) > 1:
        details = dict(algorithm_carriers)
        raise RadiusSweepSummaryError(
            f"radius {radius:g} episode planner identity carriers conflict: {details}"
        )
    mismatched = {
        label: canonical
        for label, canonical in algorithm_carriers
        if canonical != expected.algorithm
    }
    if mismatched:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} episode planner identity mismatch for directory planner "
            f"{planner!r}; expected algorithm {expected.algorithm!r}: {mismatched}"
        )

    params = _mapping(record.get("scenario_params"), "episode scenario_params")
    config_hash = _hex(
        params.get("algo_config_hash"), 16, "episode scenario_params.algo_config_hash"
    )
    if config_hash != expected.algo_config_hash:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} episode planner identity mismatch for directory planner "
            f"{planner!r}: algo_config_hash={config_hash!r}; "
            f"expected={expected.algo_config_hash!r}"
        )

    key_carriers = _episode_planner_key_carriers(record, radius)
    if expected.planner_key_required and not key_carriers:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} episode has no embedded planner_key carrier for directory "
            f"planner {planner!r}; algorithm/config identity is shared by multiple roster keys"
        )
    mismatched_keys = {label: value for label, value in key_carriers if value != planner}
    if mismatched_keys:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} episode planner identity mismatch for directory planner "
            f"{planner!r}: planner-key carriers={mismatched_keys}"
        )


def _family_receipt_rows(raw: Mapping[str, Any], radius: float) -> tuple[str, dict[str, str]]:
    definition = raw.get("definition")
    if not isinstance(definition, str) or not definition.strip():
        raise RadiusSweepSummaryError(
            f"radius {radius:g} family_feasibility requires an explicit definition"
        )
    families = _mapping(raw.get("families"), f"radius {radius:g} family feasibility rows")
    normalized: dict[str, str] = {}
    for name, status in families.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(status, str):
            raise RadiusSweepSummaryError(
                f"radius {radius:g} family_feasibility has invalid family rows"
            )
        normalized[name] = status
    if not normalized or "narrow_doorway" not in normalized:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} family_feasibility must include narrow_doorway"
        )
    invalid = {
        name: status
        for name, status in normalized.items()
        if status not in {"feasible", "infeasible"}
    }
    if invalid:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} family_feasibility has invalid statuses: {invalid}"
        )
    return definition.strip(), normalized


def _family_feasibility(
    root: Path,
    *,
    radius: float,
    campaign_id: str,
    campaign_commit: str,
    config_sha256: str,
    episodes: Sequence[_Episode],
) -> tuple[str, str, str, dict[str, str], str]:
    definition_id = EXPECTED_FAMILY_FEASIBILITY_DEFINITION_ID
    authority_sha256 = EXPECTED_FAMILY_FEASIBILITY_AUTHORITY_SHA256
    if (
        not isinstance(definition_id, str)
        or not definition_id.strip()
        or not isinstance(authority_sha256, str)
        or len(authority_sha256) != 64
        or any(char not in "0123456789abcdef" for char in authority_sha256)
    ):
        raise RadiusSweepSummaryError(
            "no owner-approved family-feasibility rule identity is pinned in this source revision"
        )
    path = root / "reports/radius_family_feasibility.json"
    if not path.is_file():
        raise RadiusSweepSummaryError(
            f"radius {radius:g} lacks authoritative family_feasibility; "
            "outcome rates and route-clearance warnings are not an approved substitute"
        )
    raw = _read_object(path, "family feasibility receipt")
    if raw.get("schema_version") != FAMILY_FEASIBILITY_SCHEMA:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} family_feasibility schema must be {FAMILY_FEASIBILITY_SCHEMA}"
        )
    approved_rule = _mapping(
        raw.get("approved_rule"), f"radius {radius:g} approved family-feasibility rule"
    )
    if (
        approved_rule.get("definition_id") != definition_id
        or approved_rule.get("authority_sha256") != authority_sha256
    ):
        raise RadiusSweepSummaryError(
            f"radius {radius:g} family_feasibility does not match the pinned approved rule identity"
        )
    if _finite(raw.get("radius_m"), "family_feasibility radius_m") != radius:
        raise RadiusSweepSummaryError(f"radius {radius:g} family_feasibility radius mismatch")
    if (
        raw.get("source_campaign_id") != campaign_id
        or raw.get("source_campaign_commit") != campaign_commit
        or raw.get("source_config_sha256") != config_sha256
    ):
        raise RadiusSweepSummaryError(
            f"radius {radius:g} family_feasibility source provenance mismatch"
        )
    definition, normalized = _family_receipt_rows(raw, radius)
    _require_family_feasibility_evaluator_match(
        root,
        radius=radius,
        episodes=episodes,
        definition_id=definition_id,
        authority_sha256=authority_sha256,
        definition=definition,
        families=normalized,
    )
    return (
        definition_id,
        authority_sha256,
        definition,
        dict(sorted(normalized.items())),
        sha256(path.read_bytes()).hexdigest(),
    )


def _require_family_feasibility_evaluator_match(
    root: Path,
    *,
    radius: float,
    episodes: Sequence[_Episode],
    definition_id: str,
    authority_sha256: str,
    definition: str,
    families: Mapping[str, str],
) -> None:
    evaluator = _FAMILY_FEASIBILITY_EVALUATOR
    if not callable(evaluator):
        raise RadiusSweepSummaryError(
            "family_feasibility cannot be admitted without an in-tree evaluator that "
            "independently recomputes family results from the exact source episodes"
        )
    try:
        evaluated_definition, evaluated_families = evaluator(
            root, radius, episodes, definition_id, authority_sha256
        )
    except RadiusSweepSummaryError as exc:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} family-feasibility evaluator failed: {exc}"
        ) from exc
    if (
        not isinstance(evaluated_definition, str)
        or not evaluated_definition.strip()
        or not isinstance(evaluated_families, Mapping)
    ):
        raise RadiusSweepSummaryError(
            f"radius {radius:g} family-feasibility evaluator returned an invalid result"
        )
    expected_families: dict[str, str] = {}
    for name, status in evaluated_families.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(status, str)
            or status not in {"feasible", "infeasible"}
        ):
            raise RadiusSweepSummaryError(
                f"radius {radius:g} family-feasibility evaluator returned invalid rows"
            )
        expected_families[name] = status
    if definition.strip() != evaluated_definition.strip() or dict(families) != expected_families:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} family_feasibility receipt does not match independently "
            "evaluated source-row results"
        )


def _episode_from_record(
    record: Mapping[str, Any],
    *,
    planner: str,
    planner_identities: Mapping[str, Mapping[str, _PlannerIdentity]],
    radius: float,
    commit: str,
) -> _Episode:
    scenario = str(record.get("scenario_id") or "")
    if scenario not in EXPECTED_SCENARIO_NAMES:
        raise RadiusSweepSummaryError(f"radius {radius:g} has unexpected scenario {scenario!r}")
    planner_identities_for_scenario = planner_identities.get(planner)
    if planner_identities_for_scenario is None or scenario not in planner_identities_for_scenario:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} has no frozen planner identity for {planner!r}/{scenario!r}"
        )
    _validate_episode_planner_identity(
        record,
        planner=planner,
        expected=planner_identities_for_scenario[scenario],
        radius=radius,
    )
    seed_raw = record.get("seed")
    if (
        isinstance(seed_raw, bool)
        or not isinstance(seed_raw, int)
        or seed_raw not in EXPECTED_SEEDS
    ):
        raise RadiusSweepSummaryError(f"radius {radius:g} has unexpected seed {seed_raw!r}")
    if record.get("git_hash") != commit:
        raise RadiusSweepSummaryError(f"radius {radius:g} episode commit mismatch")
    integrity = _mapping(record.get("integrity"), "episode integrity")
    if integrity.get("contradictions") != []:
        raise RadiusSweepSummaryError(f"radius {radius:g} episode has integrity contradictions")
    effective = _mapping(integrity.get("effective_view"), "episode effective integrity view")
    if effective.get("degraded") is not False:
        raise RadiusSweepSummaryError(f"radius {radius:g} episode is degraded")

    params = _mapping(record.get("scenario_params"), "episode scenario_params")
    robot = _mapping(params.get("robot_config"), "episode robot_config")
    if _finite(robot.get("radius"), "episode robot radius") != radius:
        raise RadiusSweepSummaryError(f"radius {radius:g} episode robot radius mismatch")
    metrics = _mapping(record.get("metrics"), "episode metrics")
    success_raw = metrics.get("success")
    if not isinstance(success_raw, bool):
        raise RadiusSweepSummaryError("episode metrics.success must be boolean")
    pedestrian = _finite(metrics.get("ped_collision_count"), "ped_collision_count")
    obstacle = _finite(metrics.get("obstacle_collision_count"), "obstacle_collision_count")
    total = _finite(metrics.get("total_collision_count"), "total_collision_count")
    if (
        pedestrian < 0
        or obstacle < 0
        or total < 0
        or not math.isclose(total, pedestrian + obstacle, abs_tol=1e-12)
    ):
        raise RadiusSweepSummaryError("typed collision fields are negative or inconsistent")
    return _Episode(
        planner=planner,
        scenario=scenario,
        seed=seed_raw,
        success=float(success_raw),
        pedestrian_collisions=pedestrian,
        obstacle_collisions=obstacle,
        typed_collisions=total,
        snqi=_finite(metrics.get("snqi"), "episode SNQI"),
    )


def _load_episodes(
    root: Path,
    radius: float,
    commit: str,
    planner_identities: Mapping[str, Mapping[str, _PlannerIdentity]],
) -> tuple[_Episode, ...]:
    episodes: list[_Episode] = []
    identities: set[tuple[str, str, int]] = set()
    runs_root = root / "runs"
    if not runs_root.is_dir():
        raise RadiusSweepSummaryError(f"missing runs directory: {runs_root}")
    expected_paths = {
        runs_root / f"{planner}__{EXPECTED_KINEMATICS}" / "episodes.jsonl"
        for planner in RELEASE_PLANNER_KEYS
    }
    actual_paths = set(runs_root.glob("*/episodes.jsonl"))
    if actual_paths != expected_paths:
        missing = sorted(str(path.relative_to(root)) for path in expected_paths - actual_paths)
        unexpected = sorted(str(path.relative_to(root)) for path in actual_paths - expected_paths)
        raise RadiusSweepSummaryError(
            f"radius {radius:g} run scope mismatch: missing={missing}, unexpected={unexpected}"
        )
    for planner in RELEASE_PLANNER_KEYS:
        episodes_path = runs_root / f"{planner}__{EXPECTED_KINEMATICS}" / "episodes.jsonl"
        try:
            lines = episodes_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RadiusSweepSummaryError(f"cannot read {episodes_path}: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RadiusSweepSummaryError(
                    f"invalid JSON at {episodes_path}:{line_number}: {exc}"
                ) from exc
            record = _mapping(raw, f"episode at {episodes_path}:{line_number}")
            episode = _episode_from_record(
                record,
                planner=planner,
                planner_identities=planner_identities,
                radius=radius,
                commit=commit,
            )
            identity = (planner, episode.scenario, episode.seed)
            if identity in identities:
                raise RadiusSweepSummaryError(
                    f"radius {radius:g} duplicate row identity {identity}"
                )
            identities.add(identity)
            episodes.append(episode)
    expected_identities = {
        (planner, scenario, seed)
        for planner in RELEASE_PLANNER_KEYS
        for scenario in EXPECTED_SCENARIO_NAMES
        for seed in EXPECTED_SEEDS
    }
    if identities != expected_identities or len(episodes) != EXPECTED_ROWS_PER_ARM:
        missing = len(expected_identities - identities)
        extra = len(identities - expected_identities)
        raise RadiusSweepSummaryError(
            f"radius {radius:g} row identity mismatch: present={len(episodes)}, "
            f"missing={missing}, extra={extra}"
        )
    return tuple(episodes)


def _committed_config_blob(commit: str, config_path: str) -> bytes:
    """Read config bytes from the source commit, never from a mutable worktree file.

    Returns:
        Exact config bytes committed at the requested path.
    """
    parsed_path = PurePosixPath(config_path)
    if parsed_path.is_absolute() or ".." in parsed_path.parts or not parsed_path.parts:
        raise RadiusSweepSummaryError(f"invalid committed config path: {config_path!r}")
    try:
        result = subprocess.run(
            ["git", "-C", str(SOURCE_REPOSITORY_ROOT), "show", f"{commit}:{config_path}"],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RadiusSweepSummaryError(
            f"frozen campaign config blob is unavailable: {commit}:{config_path}"
        ) from exc
    return result.stdout


def _committed_blob_exists(commit: str, config_path: str) -> bool:
    """Return whether a relative path is a blob in the pinned Git tree."""
    parsed_path = PurePosixPath(config_path)
    if parsed_path.is_absolute() or ".." in parsed_path.parts or not parsed_path.parts:
        raise RadiusSweepSummaryError(f"invalid committed config path: {config_path!r}")
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(SOURCE_REPOSITORY_ROOT),
                "cat-file",
                "-e",
                f"{commit}:{config_path}",
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RadiusSweepSummaryError(
            f"cannot inspect frozen config path: {commit}:{config_path}"
        ) from exc
    return result.returncode == 0


def _committed_manifest_reference_path(
    commit: str, manifest_path: str, raw_path: object
) -> str | None:
    """Resolve a policy-manifest reference using map-runner path precedence.

    Returns:
        A repository-relative committed path, or ``None`` for an absent reference.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    parsed_raw = PurePosixPath(raw_path)
    if parsed_raw.is_absolute():
        raise RadiusSweepSummaryError(
            f"absolute policy config paths cannot be resolved from a frozen commit: {raw_path!r}"
        )

    candidates = (
        posixpath.normpath(str(PurePosixPath(manifest_path).parent / parsed_raw)),
        posixpath.normpath(raw_path),
    )
    for candidate in candidates:
        if candidate in {".", ".."} or candidate.startswith("../"):
            continue
        if _committed_blob_exists(commit, candidate):
            return candidate
    raise RadiusSweepSummaryError(
        f"frozen policy config reference is unavailable from {commit}:{manifest_path}: {raw_path!r}"
    )


def _committed_manifest_reference_yaml(
    commit: str,
    manifest_path: str,
    raw_path: object,
    *,
    label: str,
) -> Mapping[str, Any]:
    """Read one policy-manifest reference from the same immutable source tree.

    Returns:
        Parsed mapping from the resolved Git blob.
    """
    resolved_path = _committed_manifest_reference_path(commit, manifest_path, raw_path)
    if resolved_path is None:
        return {}
    return _committed_yaml_mapping(commit, resolved_path, label)


def _committed_config_sha256(commit: str, config_path: str) -> str:
    """Hash config bytes from the source commit, never from a mutable worktree file.

    Returns:
        The SHA-256 digest of the committed config bytes.
    """
    return sha256(_committed_config_blob(commit, config_path)).hexdigest()


def _committed_yaml_mapping(commit: str, config_path: str, label: str) -> Mapping[str, Any]:
    """Load one mapping-valued YAML config directly from an immutable Git commit.

    Returns:
        Parsed YAML mapping, or an empty mapping for an empty config.
    """
    raw = _committed_config_blob(commit, config_path)
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RadiusSweepSummaryError(
            f"invalid committed {label} YAML: {commit}:{config_path}"
        ) from exc
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise RadiusSweepSummaryError(
            f"committed {label} must be a YAML mapping: {commit}:{config_path}"
        )
    return payload


def _committed_planner_identities(  # noqa: C901
    commit: str, campaign_config_path: str
) -> dict[str, dict[str, _PlannerIdentity]]:
    """Bind roster keys and scenarios to map-runner's resolved algorithm/config.

    Returns:
        Frozen per-scenario planner identities in campaign roster order.
    """
    campaign_config = _committed_yaml_mapping(
        commit, campaign_config_path, "radius-sweep campaign config"
    )
    raw_planners = campaign_config.get("planners")
    if not isinstance(raw_planners, list):
        raise RadiusSweepSummaryError(
            f"committed radius-sweep config has no planner roster: {campaign_config_path}"
        )

    resolved: dict[str, dict[str, tuple[str, str]]] = {}
    for index, raw_planner in enumerate(raw_planners):
        planner = _mapping(raw_planner, f"committed planner roster row {index}")
        key = planner.get("key")
        algorithm = planner.get("algo")
        if not isinstance(key, str) or not key.strip() or key in resolved:
            raise RadiusSweepSummaryError(
                f"committed planner roster row {index} has an invalid or duplicate key"
            )
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise RadiusSweepSummaryError(f"committed planner {key!r} has no configured algorithm")

        algo_config_path = planner.get("algo_config")
        manifest_path: str | None
        if algo_config_path is None:
            manifest: Mapping[str, Any] = {}
            manifest_path = None
        elif isinstance(algo_config_path, str) and algo_config_path.strip():
            manifest_path = algo_config_path
            manifest = _committed_yaml_mapping(
                commit, algo_config_path, f"planner {key!r} algorithm config"
            )
        else:
            raise RadiusSweepSummaryError(
                f"committed planner {key!r} has an invalid algo_config path"
            )
        reference_cache: dict[str, Mapping[str, Any]] = {}

        def load_config(raw_path: object) -> dict[str, Any]:
            if manifest_path is None:
                return {}
            if isinstance(raw_path, str) and raw_path.strip() in reference_cache:
                return dict(reference_cache[raw_path.strip()])
            loaded = _committed_manifest_reference_yaml(
                commit,
                manifest_path,
                raw_path,
                label=f"planner {key!r} referenced algorithm config",
            )
            if isinstance(raw_path, str) and raw_path.strip():
                reference_cache[raw_path.strip()] = loaded
            return dict(loaded)

        resolved[key] = {}
        for scenario in EXPECTED_SCENARIO_NAMES:
            try:
                effective_algorithm, effective_config = resolve_candidate_manifest_runtime(
                    default_algo=algorithm,
                    manifest=dict(manifest),
                    scenario={"name": scenario},
                    load_config=load_config,
                )
                canonical_algorithm = canonical_algorithm_name(effective_algorithm)
                config_hash = _config_hash(effective_config)
            except (TypeError, ValueError) as exc:
                raise RadiusSweepSummaryError(
                    "cannot resolve committed algorithm/config identity for planner "
                    f"{key!r}/{scenario!r}"
                ) from exc
            resolved[key][scenario] = (canonical_algorithm, config_hash)

    if tuple(resolved) != RELEASE_PLANNER_KEYS:
        raise RadiusSweepSummaryError(
            "committed planner algorithm/config roster does not match the frozen release keys"
        )

    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for planner_identities in resolved.values():
        for scenario, (algorithm, config_hash) in planner_identities.items():
            counts[(scenario, algorithm, config_hash)] += 1
    return {
        key: {
            scenario: _PlannerIdentity(
                algorithm=algorithm,
                algo_config_hash=config_hash,
                planner_key_required=counts[(scenario, algorithm, config_hash)] > 1,
            )
            for scenario, (algorithm, config_hash) in planner_identities.items()
        }
        for key, planner_identities in resolved.items()
    }


def _load_arm(root: Path) -> _Arm:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise RadiusSweepSummaryError(f"campaign root does not exist: {root}")
    manifest = _read_object(root / "campaign_manifest.json", "campaign manifest")
    preflight = _read_object(root / "preflight/validate_config.json", "preflight config receipt")
    summary = _read_object(root / "reports/campaign_summary.json", "campaign summary")
    if manifest.get("schema_version") != CAMPAIGN_SCHEMA:
        raise RadiusSweepSummaryError(f"unsupported campaign schema in {root}")
    binding = _mapping(manifest.get("radius_binding"), "campaign radius_binding")
    radius = _finite(binding.get("radius_m"), "campaign radius")
    if radius not in PRODUCTION_RADII or binding.get("arm_key") != RADIUS_TO_ARM_KEY.get(radius):
        raise RadiusSweepSummaryError(f"unexpected radius/arm identity in {root}")
    expected_config = EXPECTED_ARM_CAMPAIGN_CONFIGS[RADIUS_TO_ARM_KEY[radius]]
    config_path = str(preflight.get("config_path") or "")
    if config_path != expected_config:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} config path {config_path!r}; expected {expected_config!r}"
        )
    preflight_binding = _mapping(preflight.get("radius_binding"), "preflight radius_binding")
    if preflight_binding != binding:
        raise RadiusSweepSummaryError(f"radius {radius:g} manifest/preflight binding mismatch")
    commit = _hex(_mapping(manifest.get("git"), "manifest git").get("commit"), 40, "commit")
    if commit != EXPECTED_CAMPAIGN_GIT_COMMIT:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} campaign commit does not match the frozen #6642 source commit"
        )
    config_sha = _hex(preflight.get("config_sha256"), 64, "config_sha256")
    expected_config_sha = EXPECTED_ARM_CAMPAIGN_CONFIG_SHA256[RADIUS_TO_ARM_KEY[radius]]
    committed_config_sha = _committed_config_sha256(commit, config_path)
    if committed_config_sha != expected_config_sha:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} frozen config digest does not match the Git blob at the campaign commit"
        )
    if config_sha != committed_config_sha:
        raise RadiusSweepSummaryError(
            f"radius {radius:g} preflight config digest does not match config bytes at the campaign commit"
        )
    receipt_sha = _hex(binding.get("gate1_receipt_sha256"), 64, "Gate 1 receipt digest")
    _validate_campaign_status(summary, radius)
    planner_rows = _validate_planner_rows(summary, radius)
    campaign = _mapping(summary.get("campaign"), "campaign summary header")
    campaign_id = str(manifest.get("campaign_id") or "")
    if (
        campaign.get("campaign_id") != campaign_id
        or campaign.get("git_hash") != commit
        or campaign.get("scenario_matrix") != EXPECTED_SCENARIO_MATRIX
        or manifest.get("scenario_matrix") != EXPECTED_SCENARIO_MATRIX
        or tuple(_mapping(manifest.get("seed_policy"), "seed policy").get("resolved_seeds", ()))
        != EXPECTED_SEEDS
    ):
        raise RadiusSweepSummaryError(f"radius {radius:g} campaign identity mismatch")
    planner_identities = _committed_planner_identities(commit, config_path)
    episodes = _load_episodes(root, radius, commit, planner_identities)
    _validate_planner_aggregates(planner_rows, episodes, radius)
    (
        family_definition_id,
        family_authority_sha256,
        family_definition,
        family_feasibility,
        family_sha256,
    ) = _family_feasibility(
        root,
        radius=radius,
        campaign_id=campaign_id,
        campaign_commit=commit,
        config_sha256=config_sha,
        episodes=episodes,
    )
    return _Arm(
        radius=radius,
        root=root,
        campaign_id=campaign_id,
        campaign_commit=commit,
        config_path=config_path,
        config_sha256=config_sha,
        gate1_receipt_sha256=receipt_sha,
        family_feasibility_definition_id=family_definition_id,
        family_feasibility_authority_sha256=family_authority_sha256,
        family_feasibility_definition=family_definition,
        family_feasibility=family_feasibility,
        family_feasibility_sha256=family_sha256,
        episodes=episodes,
    )


def _arm_metrics(arm: _Arm) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    by_planner: dict[str, list[_Episode]] = defaultdict(list)
    by_planner_seed: dict[tuple[str, int], list[_Episode]] = defaultdict(list)
    for episode in arm.episodes:
        by_planner[episode.planner].append(episode)
        by_planner_seed[(episode.planner, episode.seed)].append(episode)
    table: dict[str, dict[str, float]] = {}
    paired: dict[str, Any] = {}
    for planner in RELEASE_PLANNER_KEYS:
        rows = by_planner[planner]
        table[planner] = {
            "success": sum(row.success for row in rows) / len(rows),
            "typed_collisions": sum(row.typed_collisions for row in rows) / len(rows),
            "snqi": sum(row.snqi for row in rows) / len(rows),
        }
        paired[planner] = {
            metric: {
                str(seed): sum(getattr(row, metric) for row in by_planner_seed[(planner, seed)])
                / len(by_planner_seed[(planner, seed)])
                for seed in EXPECTED_SEEDS
            }
            for metric in ("success", "typed_collisions", "snqi")
        }
    return table, paired


def _gate1_receipt_digest(gate1_canary_receipt: str | Path) -> str:
    receipt_path = Path(gate1_canary_receipt).expanduser().resolve()
    if not receipt_path.is_file():
        raise RadiusSweepSummaryError(f"missing original Gate 1 receipt: {receipt_path}")
    if not _gate1_canary_receipt_is_passing(receipt_path):
        raise RadiusSweepSummaryError(
            f"Gate 1 receipt is not a complete passing receipt: {receipt_path}"
        )
    digest = sha256(receipt_path.read_bytes()).hexdigest()
    if digest != EXPECTED_GATE1_RECEIPT_SHA256:
        raise RadiusSweepSummaryError(
            "Gate 1 receipt bytes do not match the frozen Gate 2 receipt digest"
        )
    return digest


def _validate_arm_set(arms: Sequence[_Arm], receipt_sha256: str) -> None:
    if tuple(arm.radius for arm in arms) != PRODUCTION_RADII:
        raise RadiusSweepSummaryError("campaign roots must cover exactly radii 0.5, 0.8, and 1.0")
    if len({arm.root for arm in arms}) != len(arms):
        raise RadiusSweepSummaryError("campaign roots must be distinct")
    if len({arm.campaign_commit for arm in arms}) != 1:
        raise RadiusSweepSummaryError("campaign roots use mixed commits")
    if len({arm.gate1_receipt_sha256 for arm in arms}) != 1:
        raise RadiusSweepSummaryError("campaign roots use mixed Gate 1 receipts")
    if any(arm.gate1_receipt_sha256 != receipt_sha256 for arm in arms):
        raise RadiusSweepSummaryError(
            "campaign roots do not match the supplied original Gate 1 receipt bytes"
        )
    if len({arm.family_feasibility_definition for arm in arms}) != 1:
        raise RadiusSweepSummaryError("campaign roots use mixed family-feasibility definitions")
    if len({arm.family_feasibility_definition_id for arm in arms}) != 1:
        raise RadiusSweepSummaryError("campaign roots use mixed family-feasibility rule identities")
    if len({arm.family_feasibility_authority_sha256 for arm in arms}) != 1:
        raise RadiusSweepSummaryError(
            "campaign roots use mixed family-feasibility authority digests"
        )
    family_sets = {tuple(arm.family_feasibility) for arm in arms}
    if len(family_sets) != 1:
        raise RadiusSweepSummaryError("campaign roots have mismatched family-feasibility rosters")


def compose_radius_sweep_summary(
    campaign_roots: Sequence[str | Path], *, gate1_canary_receipt: str | Path
) -> dict[str, Any]:
    """Validate three campaign roots and return a deterministic Gate 3 summary.

    Returns:
        A JSON-safe ``issue_6642_radius_sweep_summary.v1`` payload.
    """
    receipt_sha256 = _gate1_receipt_digest(gate1_canary_receipt)
    if len(campaign_roots) != len(PRODUCTION_RADII):
        raise RadiusSweepSummaryError("exactly three campaign roots are required")
    arms = sorted((_load_arm(Path(root)) for root in campaign_roots), key=lambda arm: arm.radius)
    _validate_arm_set(arms, receipt_sha256)

    tables: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    accounting: dict[str, Any] = {}
    families: dict[str, Any] = {}
    family_provenance: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for arm in arms:
        radius_key = f"{arm.radius:g}"
        tables[radius_key], paired[radius_key] = _arm_metrics(arm)
        accounting[radius_key] = {
            "declared": EXPECTED_ROWS_PER_ARM,
            "present": EXPECTED_ROWS_PER_ARM,
            "excluded_by_reason": {},
        }
        families[radius_key] = arm.family_feasibility
        family_provenance[radius_key] = {
            "definition": arm.family_feasibility_definition,
            "definition_id": arm.family_feasibility_definition_id,
            "authority_sha256": arm.family_feasibility_authority_sha256,
            "receipt_sha256": arm.family_feasibility_sha256,
        }
        provenance[radius_key] = {
            "campaign_commit": arm.campaign_commit,
            "campaign_id": arm.campaign_id,
            "config_path": arm.config_path,
            "config_sha256": arm.config_sha256,
            "gate1_canary_receipt_sha256": arm.gate1_receipt_sha256,
        }
    return {
        "schema_version": SWEEP_SUMMARY_SCHEMA,
        "radii_m": list(PRODUCTION_RADII),
        "planners": list(RELEASE_PLANNER_KEYS),
        "scenario_matrix": EXPECTED_SCENARIO_MATRIX,
        "scenario_cells": list(EXPECTED_SCENARIO_NAMES),
        "seeds": list(EXPECTED_SEEDS),
        "metric_tables": tables,
        "row_accounting": accounting,
        "paired_observations": paired,
        "family_feasibility": families,
        "family_feasibility_provenance": family_provenance,
        "campaign_provenance": provenance,
    }


def write_radius_sweep_summary(summary: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write canonical JSON bytes for a composed summary.

    Returns:
        The written output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
