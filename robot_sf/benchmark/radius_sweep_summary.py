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
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from robot_sf.benchmark.radius_rank_stability import (
    SWEEP_SUMMARY_SCHEMA,
    _gate1_canary_receipt_is_passing,
)
from robot_sf.benchmark.radius_sweep_manifest import (
    EXPECTED_ARM_CAMPAIGN_CONFIGS,
    EXPECTED_GATE1_RECEIPT_SHA256,
    EXPECTED_ROWS_PER_ARM,
    EXPECTED_SCENARIO_MATRIX,
    EXPECTED_SCENARIO_NAMES,
    EXPECTED_SEEDS,
    PRODUCTION_RADII,
    RELEASE_PLANNER_KEYS,
)

CAMPAIGN_SCHEMA = "benchmark-camera-ready-campaign.v1"
FAMILY_FEASIBILITY_SCHEMA = "issue_6642_family_feasibility.v1"
EXPECTED_KINEMATICS = "differential_drive"
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
    typed_collisions: float
    snqi: float


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


def _validate_planner_rows(summary: Mapping[str, Any], radius: float) -> None:
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


def _family_feasibility(
    root: Path,
    *,
    radius: float,
    campaign_id: str,
    campaign_commit: str,
    config_sha256: str,
) -> tuple[str, dict[str, str], str]:
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
    definition = raw.get("definition")
    if not isinstance(definition, str) or not definition.strip():
        raise RadiusSweepSummaryError(
            f"radius {radius:g} family_feasibility requires an explicit definition"
        )
    families = _mapping(raw.get("families"), f"radius {radius:g} family feasibility rows")
    normalized = {str(name): str(status) for name, status in families.items()}
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
    return (
        definition.strip(),
        dict(sorted(normalized.items())),
        sha256(path.read_bytes()).hexdigest(),
    )


def _episode_from_record(
    record: Mapping[str, Any], *, planner: str, radius: float, commit: str
) -> _Episode:
    scenario = str(record.get("scenario_id") or "")
    if scenario not in EXPECTED_SCENARIO_NAMES:
        raise RadiusSweepSummaryError(f"radius {radius:g} has unexpected scenario {scenario!r}")
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
        typed_collisions=total,
        snqi=_finite(metrics.get("snqi"), "episode SNQI"),
    )


def _load_episodes(root: Path, radius: float, commit: str) -> tuple[_Episode, ...]:
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
            episode = _episode_from_record(record, planner=planner, radius=radius, commit=commit)
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
    config_sha = _hex(preflight.get("config_sha256"), 64, "config_sha256")
    receipt_sha = _hex(binding.get("gate1_receipt_sha256"), 64, "Gate 1 receipt digest")
    _validate_campaign_status(summary, radius)
    _validate_planner_rows(summary, radius)
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
    family_definition, family_feasibility, family_sha256 = _family_feasibility(
        root,
        radius=radius,
        campaign_id=campaign_id,
        campaign_commit=commit,
        config_sha256=config_sha,
    )
    return _Arm(
        radius=radius,
        root=root,
        campaign_id=campaign_id,
        campaign_commit=commit,
        config_path=config_path,
        config_sha256=config_sha,
        gate1_receipt_sha256=receipt_sha,
        family_feasibility_definition=family_definition,
        family_feasibility=family_feasibility,
        family_feasibility_sha256=family_sha256,
        episodes=_load_episodes(root, radius, commit),
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
