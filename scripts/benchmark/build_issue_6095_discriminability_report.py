#!/usr/bin/env python3
"""Build the issue #6095 nominal-versus-stress discriminability report.

The report deliberately reads the retained episode JSONL files instead of copying
the aggregate values from ``campaign_summary.json``.  It validates the frozen
matrix contract, reports seed-aware uncertainty, classifies every stress scenario,
and keeps checkpoint/provenance blockers separate from valid campaign execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.utils import (
    episode_collision_value,
    episode_metric_value,
    episode_success_value,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_COMMIT = "fcc495b955c9eab00bc60842b5cae63f74cf2e2c"
DEFAULT_MODEL_ID = "ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417"
DEFAULT_MODEL_SHA256 = "2b30df812bfcc737924b126b0763d69c567fe20716dc1c1eba8f56f926b49c1d"
DEFAULT_SEEDS = tuple(range(111, 121))
DEFAULT_S3_SEEDS = (111, 112, 113)
EXPECTED_PLANNERS = ("orca", "ppo")
EXPECTED_KINEMATICS = "differential_drive"
EXPECTED_HORIZON = 100
EXPECTED_DT = 0.1
BOOTSTRAP_SAMPLES = 2_000


class ReportContractError(ValueError):
    """Raised when a campaign cannot satisfy the issue #6095 report contract."""


@dataclass(frozen=True)
class EpisodeRow:
    """Validated metrics and provenance for one planner/scenario/seed cell."""

    planner_key: str
    scenario_id: str
    seed: int
    success: float
    collision: float
    near_misses: float
    near_miss_any: float
    execution_mode: str
    observation_level: str
    model_id: str | None
    horizon: int | None
    dt: float | None


@dataclass
class RegimeData:
    """Validated data and diagnostics for one campaign regime."""

    name: str
    root: Path
    campaign_id: str
    scenario_matrix: str
    scenario_matrix_hash: str
    git_commit: str
    scenario_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    rows: dict[tuple[str, str, int], EpisodeRow]
    blockers: list[str]
    warnings: list[str]
    checkpoint: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReportOptions:
    """Frozen inputs for one issue #6095 report build."""

    expected_commit: str = DEFAULT_COMMIT
    expected_model_id: str = DEFAULT_MODEL_ID
    expected_model_sha256: str = DEFAULT_MODEL_SHA256
    expected_seeds: tuple[int, ...] = DEFAULT_SEEDS
    s3_seeds: tuple[int, ...] = DEFAULT_S3_SEEDS
    bootstrap_seed: int = 6095
    bootstrap_samples: int = BOOTSTRAP_SAMPLES


def _read_json(path: Path) -> Any:
    """Read a UTF-8 JSON artifact."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportContractError(f"Cannot read JSON artifact {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records, rejecting malformed rows."""
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReportContractError(f"Cannot read episode artifact {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReportContractError(f"Malformed JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ReportContractError(f"JSONL record is not an object: {path}:{line_number}")
        records.append(record)
    return records


def _finite_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` for missing/non-numeric values."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_bool(value: Any) -> bool:
    """Parse common JSON/string boolean spellings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "ok", "pass"}
    return bool(value)


def _nested(mapping: Any, *keys: str) -> Any:
    """Read a nested mapping value without raising on malformed artifacts."""
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _observation_level(record: dict[str, Any]) -> str:
    """Return the most specific observation-level value in an episode record."""
    metadata = record.get("algorithm_metadata")
    learned = _nested(metadata, "learned_checkpoint_observation_contract")
    candidates = (
        _nested(learned, "observation_level"),
        _nested(metadata, "observation_level", "key"),
        record.get("observation_level"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "missing"


def _execution_mode(record: dict[str, Any]) -> str:
    """Return the planner execution mode recorded for an episode."""
    value = _nested(record, "algorithm_metadata", "planner_kinematics", "execution_mode")
    return str(value or "missing").strip().lower()


def _model_id(record: dict[str, Any]) -> str | None:
    """Return the learned-model identifier recorded in an episode, when present."""
    value = _nested(record, "algorithm_metadata", "config", "model_id")
    if value is None:
        return None
    return str(value).strip() or None


def _record_horizon(record: dict[str, Any]) -> int | None:
    """Read the effective episode horizon from primary or provenance fields."""
    value = record.get("horizon")
    if value is None:
        value = _nested(record, "result_provenance", "simulator_settings", "horizon")
    parsed = _finite_float(value)
    return int(parsed) if parsed is not None and parsed.is_integer() else None


def _record_dt(record: dict[str, Any]) -> float | None:
    """Read the effective episode time step from provenance fields."""
    value = _nested(record, "result_provenance", "simulator_settings", "dt")
    return _finite_float(value)


def _record_kinematics(record: dict[str, Any]) -> str:
    """Return the effective robot kinematics recorded for an episode."""
    value = _nested(record, "algorithm_metadata", "planner_kinematics", "robot_kinematics")
    if value is None:
        value = _nested(record, "scenario_params", "robot_config", "type")
    return str(value or "missing").strip()


def _episode_row(record: dict[str, Any], *, planner_key: str, source: Path) -> EpisodeRow:
    """Normalize one episode record into the report's metric contract."""
    scenario_id = str(record.get("scenario_id") or "").strip()
    if not scenario_id:
        raise ReportContractError(f"Missing scenario_id in {source}")
    try:
        seed = int(record["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportContractError(f"Missing integer seed for {scenario_id} in {source}") from exc

    near_misses = episode_metric_value(record, "near_misses")
    if near_misses is None or not math.isfinite(near_misses) or near_misses < 0.0:
        raise ReportContractError(
            f"Missing or invalid near_misses for {scenario_id} seed {seed} in {source}"
        )
    return EpisodeRow(
        planner_key=planner_key,
        scenario_id=scenario_id,
        seed=seed,
        success=episode_success_value(record),
        collision=episode_collision_value(record),
        near_misses=float(near_misses),
        near_miss_any=1.0 if near_misses > 0.0 else 0.0,
        execution_mode=_execution_mode(record),
        observation_level=_observation_level(record),
        model_id=_model_id(record),
        horizon=_record_horizon(record),
        dt=_record_dt(record),
    )


def _preview_scenarios(root: Path) -> tuple[str, ...]:
    """Read the preflight scenario expansion and preserve its declared order."""
    payload = _read_json(root / "preflight" / "preview_scenarios.json")
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(scenarios, list):
        raise ReportContractError(f"Preview artifact has no scenario list: {root}")
    names = tuple(
        str(item.get("name") or "").strip()
        for item in scenarios
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    )
    if len(names) != len(scenarios) or len(set(names)) != len(names):
        raise ReportContractError(f"Preview scenario identities are invalid: {root}")
    return names


def _manifest_seeds(manifest: dict[str, Any]) -> tuple[int, ...]:
    """Read resolved seeds from the campaign manifest."""
    values = _nested(manifest, "seed_policy", "resolved_seeds")
    if not isinstance(values, list):
        raise ReportContractError("Campaign manifest has no resolved seed list")
    try:
        return tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ReportContractError("Campaign manifest seed list is invalid") from exc


def _planner_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index campaign summary planner rows by planner key."""
    rows = summary.get("planner_rows")
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("planner_key") or "").strip():
            result[str(row["planner_key"]).strip()] = row
    return result


def _checkpoint_receipt(
    root: Path, *, expected_model_id: str, expected_sha256: str
) -> dict[str, Any]:
    """Summarize the strongest checkpoint receipt available in a campaign root."""
    staging_path = root / "preflight" / "checkpoint_staging.json"
    resolvability_path = root / "preflight" / "checkpoint_resolvability.json"
    path = staging_path if staging_path.exists() else resolvability_path
    payload = _read_json(path)
    arms = payload.get("arms") if isinstance(payload, dict) else None
    ppo_arms = [
        arm
        for arm in (arms if isinstance(arms, list) else [])
        if isinstance(arm, dict) and str(arm.get("planner_key") or "") == "ppo"
    ]
    if len(ppo_arms) != 1:
        return {
            "status": "invalid",
            "path": str(path),
            "blocker": "checkpoint receipt does not contain exactly one PPO arm",
        }
    arm = ppo_arms[0]
    model_id = str(arm.get("model_id") or arm.get("value") or "").strip()
    sha256 = str(arm.get("checkpoint_sha256") or "").strip()
    identity_matches = model_id == expected_model_id and sha256 == expected_sha256
    mode = str(payload.get("mode") or "").strip()
    staged = bool(payload.get("stage")) and _as_bool(payload.get("submit_safe"))
    if not identity_matches:
        status = "identity_mismatch"
    elif staged and str(arm.get("status") or "") == "staged":
        status = "staged_receipt"
    elif mode == "metadata_only":
        status = "metadata_only"
    else:
        status = "unresolved"
    return {
        "status": status,
        "path": str(path),
        "mode": mode,
        "stage": bool(payload.get("stage")),
        "submit_safe": _as_bool(payload.get("submit_safe")),
        "arm_status": arm.get("status"),
        "model_id": model_id,
        "checkpoint_sha256": sha256,
        "hash_source": arm.get("hash_source"),
        "resolved_path": arm.get("resolved_path"),
        "load_status": arm.get("load_status"),
        "identity_matches_expected": identity_matches,
    }


def _validate_campaign_receipts(  # noqa: C901
    *,
    name: str,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    integrity: dict[str, Any],
    expected_matrix: str,
    expected_seeds: tuple[int, ...],
    expected_commit: str,
) -> tuple[dict[str, Any], list[str]]:
    """Validate campaign, manifest, and integrity receipts."""
    campaign = summary.get("campaign")
    campaign = campaign if isinstance(campaign, dict) else {}
    blockers: list[str] = []
    manifest_commit = str(_nested(manifest, "git", "commit") or "").strip()
    summary_commit = str(campaign.get("git_hash") or "").strip()
    manifest_matrix = str(manifest.get("scenario_matrix") or "").strip()
    summary_matrix = str(campaign.get("scenario_matrix") or "").strip()
    if manifest_matrix != expected_matrix or summary_matrix != expected_matrix:
        blockers.append(
            f"{name}: scenario matrix mismatch (manifest={manifest_matrix!r}, "
            f"summary={summary_matrix!r}, expected={expected_matrix!r})"
        )
    if manifest_commit != expected_commit or summary_commit != expected_commit:
        blockers.append(
            f"{name}: repository commit mismatch (manifest={manifest_commit!r}, "
            f"summary={summary_commit!r}, expected={expected_commit!r})"
        )
    if _manifest_seeds(manifest) != expected_seeds:
        blockers.append(f"{name}: manifest seeds do not match expected S10 seeds")
    if not _as_bool(campaign.get("benchmark_success")):
        blockers.append(f"{name}: campaign benchmark_success is not true")
    if str(campaign.get("evidence_status") or "") != "valid":
        blockers.append(f"{name}: campaign evidence_status is not valid")
    if str(campaign.get("campaign_execution_status") or "") != "completed":
        blockers.append(f"{name}: campaign execution did not complete")
    row_status = campaign.get("row_status_summary")
    if isinstance(row_status, dict):
        for field, description in (
            ("fallback_or_degraded_rows", "fallback/degraded rows"),
            ("unexpected_failed_rows", "unexpected failed rows"),
        ):
            if int(row_status.get(field) or 0) != 0:
                blockers.append(f"{name}: {description} are present")
    if str(integrity.get("status") or "") != "valid" or not _as_bool(
        integrity.get("benchmark_success_allowed")
    ):
        blockers.append(f"{name}: campaign integrity does not allow benchmark success")
    if integrity.get("blockers"):
        blockers.append(f"{name}: campaign integrity reports blockers")
    return campaign, blockers


def _validate_campaign_metadata(
    *,
    name: str,
    root: Path,
    expected_matrix: str,
    expected_seeds: tuple[int, ...],
    expected_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], str, list[str], list[str]]:
    """Validate campaign-level receipts and return the normalized metadata."""
    summary = _read_json(root / "reports" / "campaign_summary.json")
    manifest = _read_json(root / "campaign_manifest.json")
    integrity = _read_json(root / "reports" / "campaign_integrity.json")
    preview = _read_json(root / "preflight" / "preview_scenarios.json")
    scenario_ids = _preview_scenarios(root)
    campaign, blockers = _validate_campaign_receipts(
        name=name,
        summary=summary,
        manifest=manifest,
        integrity=integrity,
        expected_matrix=expected_matrix,
        expected_seeds=expected_seeds,
        expected_commit=expected_commit,
    )
    warnings: list[str] = []
    if len(scenario_ids) not in {4, 48}:
        blockers.append(f"{name}: unexpected scenario count {len(scenario_ids)}")
    preview_scenarios = preview.get("scenarios") if isinstance(preview, dict) else []
    preview_seed_sets = {
        tuple(int(seed) for seed in item.get("seeds", []))
        for item in (preview_scenarios or [])
        if isinstance(item, dict)
    }
    if preview_seed_sets != {expected_seeds}:
        blockers.append(f"{name}: preview seed schedule does not match expected S10 seeds")
    summary_rows = _planner_rows(summary)
    row_blockers, row_warnings = _validate_summary_rows(
        name=name,
        summary_rows=summary_rows,
        expected_episode_count=len(scenario_ids) * len(expected_seeds),
    )
    blockers.extend(row_blockers)
    warnings.extend(row_warnings)
    return (
        summary,
        manifest,
        scenario_ids,
        str(campaign.get("scenario_matrix_hash") or manifest.get("scenario_matrix_hash") or ""),
        blockers,
        warnings,
    )


def _validate_summary_rows(
    *, name: str, summary_rows: dict[str, dict[str, Any]], expected_episode_count: int
) -> tuple[list[str], list[str]]:
    """Validate planner-level campaign summary rows."""
    blockers: list[str] = []
    warnings: list[str] = []
    if set(summary_rows) != set(EXPECTED_PLANNERS):
        blockers.append(f"{name}: planner roster is {sorted(summary_rows)!r}, expected ORCA/PPO")
    for planner_key in EXPECTED_PLANNERS:
        row = summary_rows.get(planner_key)
        if row is None:
            continue
        if int(float(row.get("episodes") or 0)) != expected_episode_count:
            blockers.append(f"{name}/{planner_key}: unexpected episode count")
        if str(row.get("status") or "") != "ok" or not _as_bool(row.get("benchmark_success")):
            blockers.append(f"{name}/{planner_key}: summary row is not benchmark-success")
        if str(row.get("availability_status") or "") != "available":
            blockers.append(f"{name}/{planner_key}: availability is not available")
        if str(row.get("execution_mode") or "").lower() in {
            "fallback",
            "degraded",
            "mixed",
            "unavailable",
        }:
            blockers.append(f"{name}/{planner_key}: summary execution mode is degraded/fallback")
        fairness_flags = row.get("fairness_mismatch_flags")
        if fairness_flags:
            warnings.append(
                f"{name}/{planner_key}: campaign fairness summary reports {len(fairness_flags)} "
                "mismatch flag(s); this report does not use planner ranking."
            )
    return blockers, warnings


def _load_episode_rows(
    *,
    name: str,
    root: Path,
    scenario_ids: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    expected_commit: str,
    expected_model_id: str,
) -> tuple[dict[tuple[str, str, int], EpisodeRow], list[str], list[str]]:
    """Load raw episode rows and validate identity/provenance invariants."""
    rows: dict[tuple[str, str, int], EpisodeRow] = {}
    blockers: list[str] = []
    warnings: list[str] = []
    expected_episode_count = len(scenario_ids) * len(expected_seeds)
    for planner_key in EXPECTED_PLANNERS:
        path = root / "runs" / f"{planner_key}__{EXPECTED_KINEMATICS}" / "episodes.jsonl"
        records = _read_jsonl(path)
        if len(records) != expected_episode_count:
            blockers.append(
                f"{name}/{planner_key}: raw episode count {len(records)} != {expected_episode_count}"
            )
        for record in records:
            row = _episode_row(record, planner_key=planner_key, source=path)
            key = (planner_key, row.scenario_id, row.seed)
            if key in rows:
                blockers.append(f"{name}: duplicate planner/scenario/seed identity {key!r}")
                continue
            rows[key] = row
            blockers.extend(
                _validate_episode_row(
                    name=name,
                    key=key,
                    record=record,
                    row=row,
                    scenario_ids=scenario_ids,
                    expected_seeds=expected_seeds,
                    expected_commit=expected_commit,
                    expected_model_id=expected_model_id,
                )
            )
            if row.model_id is not None and planner_key == "orca":
                warnings.append(f"{name}: ORCA row unexpectedly carries a model id at {key!r}")
    expected_keys = {
        (planner, scenario, seed)
        for planner in EXPECTED_PLANNERS
        for scenario in scenario_ids
        for seed in expected_seeds
    }
    if set(rows) != expected_keys:
        blockers.append(
            f"{name}: raw identity set has {len(rows)} rows; expected {len(expected_keys)}"
        )
    return rows, blockers, warnings


def _validate_episode_row(
    *,
    name: str,
    key: tuple[str, str, int],
    record: dict[str, Any],
    row: EpisodeRow,
    scenario_ids: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    expected_commit: str,
    expected_model_id: str,
) -> list[str]:
    """Validate one raw episode row against the frozen contract."""
    blockers: list[str] = []
    planner_key = key[0]
    if row.scenario_id not in scenario_ids or row.seed not in expected_seeds:
        blockers.append(f"{name}: episode identity outside frozen matrix: {key!r}")
    if row.execution_mode not in {"native", "adapter"}:
        blockers.append(f"{name}: unsupported execution mode {row.execution_mode!r} at {key!r}")
    if row.horizon != EXPECTED_HORIZON:
        blockers.append(f"{name}: horizon mismatch at {key!r}: {row.horizon!r}")
    if row.dt is None or not math.isclose(row.dt, EXPECTED_DT, abs_tol=1e-12):
        blockers.append(f"{name}: dt mismatch at {key!r}: {row.dt!r}")
    if planner_key == "ppo" and row.model_id != expected_model_id:
        blockers.append(f"{name}: PPO model id mismatch at {key!r}: {row.model_id!r}")
    if _record_kinematics(record) != EXPECTED_KINEMATICS:
        blockers.append(f"{name}: kinematics mismatch at {key!r}")
    record_commit = str(record.get("git_hash") or "").strip()
    if record_commit != expected_commit:
        blockers.append(f"{name}: episode commit mismatch at {key!r}")
    return blockers


def _regime_metadata(
    *,
    root: Path,
    campaign: dict[str, Any],
    manifest: dict[str, Any],
    scenario_ids: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    rows: dict[tuple[str, str, int], EpisodeRow],
) -> dict[str, Any]:
    """Build compact campaign metadata from validated artifacts."""
    raw_modes = {
        planner: sorted({row.execution_mode for key, row in rows.items() if key[0] == planner})
        for planner in EXPECTED_PLANNERS
    }
    raw_observations = {
        planner: sorted({row.observation_level for key, row in rows.items() if key[0] == planner})
        for planner in EXPECTED_PLANNERS
    }
    matrix_summary = _read_json(root / "reports" / "matrix_summary.json")
    return {
        "campaign_id": str(campaign.get("campaign_id") or manifest.get("campaign_id") or ""),
        "scenario_count": len(scenario_ids),
        "episode_count_per_planner": len(scenario_ids) * len(expected_seeds),
        "total_episodes": int(campaign.get("total_episodes") or 0),
        "raw_execution_modes": raw_modes,
        "raw_observation_levels": raw_observations,
        "config_hashes": [
            str(row.get("config_hash"))
            for row in (matrix_summary.get("rows") or [])
            if isinstance(row, dict) and row.get("config_hash")
        ],
        "checkpoint_provenance_enforcement": manifest.get("checkpoint_provenance_enforcement"),
    }


def _load_regime(
    *,
    name: str,
    root: Path,
    expected_matrix: str,
    expected_seeds: tuple[int, ...],
    expected_commit: str,
    expected_model_id: str,
    expected_model_sha256: str,
) -> RegimeData:
    """Validate and load one nominal or stress campaign."""
    root = root.resolve()
    summary, manifest, scenario_ids, matrix_hash, blockers, warnings = _validate_campaign_metadata(
        name=name,
        root=root,
        expected_matrix=expected_matrix,
        expected_seeds=expected_seeds,
        expected_commit=expected_commit,
    )
    campaign = summary.get("campaign") if isinstance(summary, dict) else {}
    campaign = campaign if isinstance(campaign, dict) else {}
    checkpoint = _checkpoint_receipt(
        root,
        expected_model_id=expected_model_id,
        expected_sha256=expected_model_sha256,
    )
    if checkpoint.get("status") == "identity_mismatch":
        blockers.append(f"{name}: PPO checkpoint identity/hash does not match frozen provenance")
    elif checkpoint.get("status") in {"invalid", "unresolved"}:
        blockers.append(f"{name}: PPO checkpoint receipt is unresolved")
    rows, row_blockers, row_warnings = _load_episode_rows(
        name=name,
        root=root,
        scenario_ids=scenario_ids,
        expected_seeds=expected_seeds,
        expected_commit=expected_commit,
        expected_model_id=expected_model_id,
    )
    blockers.extend(row_blockers)
    warnings.extend(row_warnings)
    metadata = _regime_metadata(
        root=root,
        campaign=campaign,
        manifest=manifest,
        scenario_ids=scenario_ids,
        expected_seeds=expected_seeds,
        rows=rows,
    )
    return RegimeData(
        name=name,
        root=root,
        campaign_id=metadata["campaign_id"],
        scenario_matrix=expected_matrix,
        scenario_matrix_hash=matrix_hash,
        git_commit=expected_commit,
        scenario_ids=scenario_ids,
        seeds=expected_seeds,
        rows=rows,
        blockers=sorted(set(blockers)),
        warnings=sorted(set(warnings)),
        checkpoint=checkpoint,
        metadata=metadata,
    )


def _stable_seed(*parts: str, base: int) -> int:
    """Derive a deterministic NumPy seed without relying on Python hash randomization."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:4], "big")) % (2**32 - 1)


def bootstrap_mean_ci(
    matrix: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float, float]:
    """Estimate a mean and two-way scenario/seed bootstrap 95% interval.

    Each resample draws the complete seed cluster and the complete scenario
    cluster with replacement, then evaluates their cross-product.  This keeps
    the declared scenario and seed structure visible instead of treating all
    episode rows as independent Bernoulli observations.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    seed_count, scenario_count = values.shape
    observed = float(values.mean())
    rng = np.random.default_rng(bootstrap_seed)
    seed_indices = rng.integers(0, seed_count, size=(bootstrap_samples, seed_count))
    scenario_indices = rng.integers(0, scenario_count, size=(bootstrap_samples, scenario_count))
    sampled = values[seed_indices[:, :, None], scenario_indices[:, None, :]]
    estimates = sampled.mean(axis=(1, 2))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return observed, float(low), float(high)


def _metric_value(row: EpisodeRow, metric: str) -> float:
    """Read one normalized metric from an episode row."""
    if metric == "success":
        return row.success
    if metric == "collision":
        return row.collision
    if metric == "near_miss_any":
        return row.near_miss_any
    if metric == "near_misses":
        return row.near_misses
    raise KeyError(metric)


def _matrix_for(
    regime: RegimeData,
    *,
    planner_key: str,
    metric: str,
    seeds: tuple[int, ...],
) -> np.ndarray:
    """Build a seed-by-scenario matrix for one planner and metric."""
    return np.asarray(
        [
            [
                _metric_value(regime.rows[(planner_key, scenario_id, seed)], metric)
                for scenario_id in regime.scenario_ids
            ]
            for seed in seeds
        ],
        dtype=float,
    )


def summarize_regime(
    regime: RegimeData,
    *,
    seeds: tuple[int, ...],
    label: str,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Summarize one regime for one seed schedule."""
    metrics = ("success", "collision", "near_miss_any", "near_misses")
    planners: dict[str, Any] = {}
    for planner_key in EXPECTED_PLANNERS:
        planner_metrics: dict[str, Any] = {}
        for metric in metrics:
            matrix = _matrix_for(regime, planner_key=planner_key, metric=metric, seeds=seeds)
            observed, ci_low, ci_high = bootstrap_mean_ci(
                matrix,
                bootstrap_seed=_stable_seed(
                    regime.name, label, planner_key, metric, base=bootstrap_seed
                ),
                bootstrap_samples=bootstrap_samples,
            )
            planner_metrics[metric] = {
                "mean": observed,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "confidence": 0.95,
                "bootstrap_method": "two_way_scenario_seed_cluster",
                "bootstrap_samples": bootstrap_samples,
            }
        planners[planner_key] = planner_metrics
    return {
        "label": label,
        "seed_list": list(seeds),
        "seed_count": len(seeds),
        "scenario_count": len(regime.scenario_ids),
        "episode_count_per_planner": len(seeds) * len(regime.scenario_ids),
        "planners": planners,
    }


def _success_floor_class(success_values: Iterable[float]) -> str:
    """Classify one planner's scenario success support."""
    return "some_success" if any(value > 0.0 for value in success_values) else "zero_success"


def classify_stress_floor(
    regime: RegimeData,
    *,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Classify every stress scenario and quantify both-zero metric differences."""
    rows: list[dict[str, Any]] = []
    for scenario_id in regime.scenario_ids:
        planner_values: dict[str, dict[str, float]] = {}
        for planner_key in EXPECTED_PLANNERS:
            cells = [regime.rows[(planner_key, scenario_id, seed)] for seed in seeds]
            planner_values[planner_key] = {
                "success_count": float(sum(cell.success > 0.0 for cell in cells)),
                "success_rate": float(np.mean([cell.success for cell in cells])),
                "collision_rate": float(np.mean([cell.collision for cell in cells])),
                "near_miss_any_rate": float(np.mean([cell.near_miss_any for cell in cells])),
                "near_misses_mean": float(np.mean([cell.near_misses for cell in cells])),
            }
        classes = {
            planner: _success_floor_class(
                [regime.rows[(planner, scenario_id, seed)].success for seed in seeds]
            )
            for planner in EXPECTED_PLANNERS
        }
        if all(value == "some_success" for value in classes.values()):
            floor_class = "both_planners_some_success"
        elif all(value == "zero_success" for value in classes.values()):
            floor_class = "both_planners_zero_success"
        else:
            floor_class = "exactly_one_planner_some_success"
        collision_delta = (
            planner_values["ppo"]["collision_rate"] - planner_values["orca"]["collision_rate"]
        )
        near_any_delta = (
            planner_values["ppo"]["near_miss_any_rate"]
            - planner_values["orca"]["near_miss_any_rate"]
        )
        near_count_delta = (
            planner_values["ppo"]["near_misses_mean"] - planner_values["orca"]["near_misses_mean"]
        )
        distinguished_by_collision = not math.isclose(collision_delta, 0.0, abs_tol=1e-12)
        distinguished_by_near_miss = not (
            math.isclose(near_any_delta, 0.0, abs_tol=1e-12)
            and math.isclose(near_count_delta, 0.0, abs_tol=1e-12)
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "floor_class": floor_class,
                "planners": planner_values,
                "both_zero_discriminability": {
                    "applicable": floor_class == "both_planners_zero_success",
                    "distinguished_by_collision": distinguished_by_collision,
                    "distinguished_by_near_miss": distinguished_by_near_miss,
                    "collision_rate_delta_ppo_minus_orca": collision_delta,
                    "near_miss_any_rate_delta_ppo_minus_orca": near_any_delta,
                    "near_misses_mean_delta_ppo_minus_orca": near_count_delta,
                },
            }
        )
    both_zero = [row for row in rows if row["floor_class"] == "both_planners_zero_success"]
    distinguished = [
        row
        for row in both_zero
        if row["both_zero_discriminability"]["distinguished_by_collision"]
        or row["both_zero_discriminability"]["distinguished_by_near_miss"]
    ]
    counts = defaultdict(int)
    for row in rows:
        counts[row["floor_class"]] += 1
    return {
        "seed_list": list(seeds),
        "scenario_count": len(rows),
        "class_counts": dict(sorted(counts.items())),
        "both_zero_count": len(both_zero),
        "both_zero_distinguished_count": len(distinguished),
        "both_zero_distinguished_by_collision_count": sum(
            row["both_zero_discriminability"]["distinguished_by_collision"] for row in both_zero
        ),
        "both_zero_distinguished_by_near_miss_count": sum(
            row["both_zero_discriminability"]["distinguished_by_near_miss"] for row in both_zero
        ),
        "scenarios": rows,
    }


def _regime_comparison(
    nominal_summaries: dict[str, Any],
    stress_summaries: dict[str, Any],
) -> dict[str, Any]:
    """Compare nominal and stress estimates without making a planner ranking claim."""
    result: dict[str, Any] = {}
    for label in ("s3", "s10"):
        nominal = nominal_summaries[label]["planners"]
        stress = stress_summaries[label]["planners"]
        result[label] = {}
        for planner_key in EXPECTED_PLANNERS:
            success_gap = (
                nominal[planner_key]["success"]["mean"] - stress[planner_key]["success"]["mean"]
            )
            collision_gap = (
                nominal[planner_key]["collision"]["mean"] - stress[planner_key]["collision"]["mean"]
            )
            result[label][planner_key] = {
                "nominal_success_mean": nominal[planner_key]["success"]["mean"],
                "stress_success_mean": stress[planner_key]["success"]["mean"],
                "nominal_minus_stress_success": success_gap,
                "nominal_success_higher": success_gap > 0.0,
                "nominal_minus_stress_collision": collision_gap,
            }
    for planner_key in EXPECTED_PLANNERS:
        s3_gap = result["s3"][planner_key]["nominal_minus_stress_success"]
        s10_gap = result["s10"][planner_key]["nominal_minus_stress_success"]
        result.setdefault("direction", {})[planner_key] = {
            "s3_nominal_higher": s3_gap > 0.0,
            "s10_nominal_higher": s10_gap > 0.0,
            "direction_stable": (s3_gap > 0.0) == (s10_gap > 0.0),
        }
    result["both_planners_nominal_higher_s10"] = all(
        result["s10"][planner]["nominal_success_higher"] for planner in EXPECTED_PLANNERS
    )
    result["both_planners_direction_stable"] = all(
        result["direction"][planner]["direction_stable"] for planner in EXPECTED_PLANNERS
    )
    return result


def build_report(
    *,
    nominal_root: Path,
    stress_root: Path,
    options: ReportOptions | None = None,
) -> dict[str, Any]:
    """Build the machine-readable issue #6095 report."""
    options = options or ReportOptions()
    if not set(options.s3_seeds).issubset(options.expected_seeds) or not options.s3_seeds:
        raise ValueError("s3_seeds must be a non-empty subset of expected_seeds")
    nominal = _load_regime(
        name="nominal",
        root=nominal_root,
        expected_matrix="configs/scenarios/nominal_v1.yaml",
        expected_seeds=options.expected_seeds,
        expected_commit=options.expected_commit,
        expected_model_id=options.expected_model_id,
        expected_model_sha256=options.expected_model_sha256,
    )
    stress = _load_regime(
        name="stress",
        root=stress_root,
        expected_matrix="configs/scenarios/classic_interactions_francis2023.yaml",
        expected_seeds=options.expected_seeds,
        expected_commit=options.expected_commit,
        expected_model_id=options.expected_model_id,
        expected_model_sha256=options.expected_model_sha256,
    )
    validation_blockers = sorted(set(nominal.blockers + stress.blockers))
    if nominal.scenario_ids == stress.scenario_ids:
        validation_blockers.append("nominal and stress scenario matrices unexpectedly match")
    if nominal.scenario_matrix_hash == stress.scenario_matrix_hash:
        validation_blockers.append("nominal and stress scenario matrix hashes unexpectedly match")
    nominal_summaries = {
        "s3": summarize_regime(
            nominal,
            seeds=options.s3_seeds,
            label="s3",
            bootstrap_seed=options.bootstrap_seed,
            bootstrap_samples=options.bootstrap_samples,
        ),
        "s10": summarize_regime(
            nominal,
            seeds=options.expected_seeds,
            label="s10",
            bootstrap_seed=options.bootstrap_seed,
            bootstrap_samples=options.bootstrap_samples,
        ),
    }
    stress_summaries = {
        "s3": summarize_regime(
            stress,
            seeds=options.s3_seeds,
            label="s3",
            bootstrap_seed=options.bootstrap_seed,
            bootstrap_samples=options.bootstrap_samples,
        ),
        "s10": summarize_regime(
            stress,
            seeds=options.expected_seeds,
            label="s10",
            bootstrap_seed=options.bootstrap_seed,
            bootstrap_samples=options.bootstrap_samples,
        ),
    }
    floor = classify_stress_floor(stress, seeds=options.expected_seeds)
    comparison = _regime_comparison(nominal_summaries, stress_summaries)
    provenance_blockers = [
        f"{regime.name}: checkpoint receipt is metadata-only; runtime checkpoint use is not hash-verified"
        for regime in (nominal, stress)
        if regime.checkpoint.get("status") == "metadata_only"
    ]
    blockers = sorted(set(validation_blockers + provenance_blockers))
    if validation_blockers:
        status = "blocked_validation"
    elif provenance_blockers:
        status = "blocked_ppo_provenance"
    elif not comparison["both_planners_nominal_higher_s10"]:
        status = "revise_nominal_stress_direction"
    elif not comparison["both_planners_direction_stable"]:
        status = "revise_s3_s10_direction"
    else:
        status = "diagnostic_ready_for_domain_review"
    ppo_provenance = {
        "model_id": options.expected_model_id,
        "checkpoint_sha256": options.expected_model_sha256,
        "nominal": nominal.checkpoint,
        "stress": stress.checkpoint,
        "interpretation_status": "blocked" if provenance_blockers else "declared_and_receipted",
        "overlap_caveat": {
            "stress": "documented_in_distribution_in_ppo_full_maintained_eval_v1",
            "nominal": "not_in_documented_eval-set_components; possible atomic-archetype overlap is unresolved",
            "claim_impact": "no unseen-scenario generalization or planner-family claim is allowed",
        },
    }
    return {
        "schema_version": "issue-6095-discriminability-report.v1",
        "issue": 6095,
        "status": status,
        "benchmark_success_allowed": not validation_blockers,
        "interpretation_allowed": not blockers,
        "claim_boundary": (
            "Configured-matrix nominal-versus-stress diagnostics only; no planner-family "
            "superiority, transfer, unseen-scenario generalization, safety, or paper-grade claim."
        ),
        "frozen_contract": {
            "commit": options.expected_commit,
            "planners": list(EXPECTED_PLANNERS),
            "seeds": list(options.expected_seeds),
            "s3_seeds": list(options.s3_seeds),
            "horizon": EXPECTED_HORIZON,
            "dt": EXPECTED_DT,
            "kinematics": EXPECTED_KINEMATICS,
            "model_id": options.expected_model_id,
            "checkpoint_sha256": options.expected_model_sha256,
        },
        "regimes": {
            "nominal": {
                "campaign_id": nominal.campaign_id,
                "root": str(nominal.root),
                "scenario_matrix": nominal.scenario_matrix,
                "scenario_matrix_hash": nominal.scenario_matrix_hash,
                "scenario_ids": list(nominal.scenario_ids),
                "metadata": nominal.metadata,
                "s3": nominal_summaries["s3"],
                "s10": nominal_summaries["s10"],
            },
            "stress": {
                "campaign_id": stress.campaign_id,
                "root": str(stress.root),
                "scenario_matrix": stress.scenario_matrix,
                "scenario_matrix_hash": stress.scenario_matrix_hash,
                "scenario_ids": list(stress.scenario_ids),
                "metadata": stress.metadata,
                "s3": stress_summaries["s3"],
                "s10": stress_summaries["s10"],
            },
        },
        "nominal_vs_stress": comparison,
        "stress_floor": floor,
        "ppo_provenance": ppo_provenance,
        "warnings": sorted(set(nominal.warnings + stress.warnings)),
        "blockers": blockers,
        "analysis_method": {
            "confidence": 0.95,
            "bootstrap": "two_way_scenario_seed_cluster",
            "bootstrap_samples": options.bootstrap_samples,
            "bootstrap_seed": options.bootstrap_seed,
            "near_miss_outcome": "episode has near_misses > 0",
            "both_zero_discriminability": "exact observed rate/count inequality; descriptive, not a significance test",
        },
    }


def _fmt_interval(summary: dict[str, Any]) -> str:
    """Format a mean and 95% interval for Markdown."""
    return f"{summary['mean']:.4f} [{summary['ci_low']:.4f}, {summary['ci_high']:.4f}]"


def _provenance_limitation_lines(ppo_provenance: dict[str, Any]) -> list[str]:
    """Render receipt-aware PPO provenance caveats without stale status claims."""
    receipts = {
        regime_name: ppo_provenance.get(regime_name) or {} for regime_name in ("nominal", "stress")
    }
    statuses = {name: str(receipt.get("status") or "unknown") for name, receipt in receipts.items()}
    metadata_only = [name for name, status in statuses.items() if status == "metadata_only"]
    if metadata_only:
        regimes = ", ".join(metadata_only)
        return [
            f"- The {regimes} checkpoint receipt(s) are metadata-only; runtime checkpoint use "
            "is not hash-verified, so the fail-closed provenance status blocks interpretive "
            "promotion under the issue stop condition."
        ]

    staged_receipts = all(
        receipt.get("status") == "staged_receipt"
        and receipt.get("identity_matches_expected") is True
        and receipt.get("hash_source") == "computed_file"
        and receipt.get("submit_safe") is True
        for receipt in receipts.values()
    )
    if staged_receipts:
        load_statuses = ", ".join(
            f"{name}={receipt.get('load_status') or 'unknown'}"
            for name, receipt in receipts.items()
        )
        return [
            "- Nominal and stress checkpoint preflight receipts are staged, identity-matched, "
            "and checksum-verified from the computed file with submit-safe status.",
            f"- Runtime checkpoint load status remains `{load_statuses}`; no runtime load or "
            "runtime hash-verification claim is made.",
        ]

    receipt_statuses = ", ".join(f"{name}={status}" for name, status in statuses.items())
    return [
        f"- Checkpoint receipt statuses: `{receipt_statuses}`; interpretive promotion follows the fail-closed status above."
    ]


def _markdown_report(report: dict[str, Any]) -> str:
    """Render a compact human-readable report."""
    lines = [
        "# Issue #6095 S10 nominal-versus-stress discriminability report",
        "",
        f"- Status: **{report['status']}**",
        f"- Interpretation allowed: **{report['interpretation_allowed']}**",
        f"- Benchmark execution success allowed: **{report['benchmark_success_allowed']}**",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Frozen contract and evidence counts",
        "",
        "| regime | scenarios | episodes/planner | campaign | scenario matrix hash |",
        "|---|---:|---:|---|---|",
    ]
    for regime_name in ("nominal", "stress"):
        regime = report["regimes"][regime_name]
        lines.append(
            f"| {regime_name} | {len(regime['scenario_ids'])} | "
            f"{regime['metadata']['episode_count_per_planner']} | {regime['campaign_id']} | "
            f"`{regime['scenario_matrix_hash']}` |"
        )
    lines.extend(
        [
            "",
            "## Success and collision estimates",
            "",
            "Intervals are two-way scenario/seed cluster bootstrap 95% percentile intervals; "
            "they are descriptive and do not establish planner superiority.",
            "",
            "| seed schedule | regime | planner | success | collision | near-miss-any |",
            "|---|---|---|---|---|---|",
        ]
    )
    for label in ("s3", "s10"):
        for regime_name in ("nominal", "stress"):
            for planner in EXPECTED_PLANNERS:
                metrics = report["regimes"][regime_name][label]["planners"][planner]
                lines.append(
                    f"| {label.upper()} | {regime_name} | {planner} | "
                    f"{_fmt_interval(metrics['success'])} | {_fmt_interval(metrics['collision'])} | "
                    f"{_fmt_interval(metrics['near_miss_any'])} |"
                )
    lines.extend(
        [
            "",
            "## Nominal-versus-stress decision rule",
            "",
            f"- Both planners have a higher observed nominal success estimate in S10: "
            f"**{report['nominal_vs_stress']['both_planners_nominal_higher_s10']}**.",
            f"- The nominal-higher direction is stable between S3 and S10: "
            f"**{report['nominal_vs_stress']['both_planners_direction_stable']}**.",
            "- These observations are conditional diagnostics; the report does not promote the "
            "stress-floor interpretation when a blocker below is present.",
            "",
            "## Stress success-floor classification",
            "",
            f"- Both planners have some success: **{report['stress_floor']['class_counts'].get('both_planners_some_success', 0)}**.",
            f"- Exactly one planner has some success: **{report['stress_floor']['class_counts'].get('exactly_one_planner_some_success', 0)}**.",
            f"- Both planners have zero success: **{report['stress_floor']['both_zero_count']}**.",
            f"- Both-zero scenarios with a non-equal collision or near-miss outcome: **{report['stress_floor']['both_zero_distinguished_count']}**.",
            f"  Collision-only count: **{report['stress_floor']['both_zero_distinguished_by_collision_count']}**; "
            f"near-miss count: **{report['stress_floor']['both_zero_distinguished_by_near_miss_count']}**.",
            "",
            "Near-miss discriminability uses both the episode-level any-near-miss rate and the "
            "mean near-miss count; differences are descriptive exact observed differences.",
            "",
            "## PPO provenance and limitations",
            "",
            f"- Model ID: `{report['ppo_provenance']['model_id']}`",
            f"- Declared checkpoint SHA-256: `{report['ppo_provenance']['checkpoint_sha256']}`",
            f"- Provenance interpretation status: **{report['ppo_provenance']['interpretation_status']}**",
            *_provenance_limitation_lines(report["ppo_provenance"]),
            "- Raw episode modes are retained separately (`orca=adapter`, `ppo=native` in the "
            "validated campaign artifacts); adapter/native and observation-contract caveats prevent "
            "planner ranking claims.",
            "- The stress PPO evaluation set is documented as in-distribution; nominal overlap is "
            "not fully resolved. No generalization claim is permitted.",
        ]
    )
    blockers = report.get("blockers") or []
    lines.extend(["", "## Blockers", ""])
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- None.")
    warnings = report.get("warnings") or []
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Source commit: `{report['frozen_contract']['commit']}`",
            "- Raw episode inputs and campaign receipts remain outside Git; promote them with "
            "retrieval metadata before treating this as durable benchmark evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    """Build the report CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-root", type=Path, required=True)
    parser.add_argument("--stress-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", default=DEFAULT_COMMIT)
    parser.add_argument("--expected-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--expected-model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--expected-seed", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--s3-seed", type=int, nargs="+", default=list(DEFAULT_S3_SEEDS))
    parser.add_argument("--bootstrap-seed", type=int, default=6095)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser


def main() -> int:
    """Build JSON and Markdown report artifacts."""
    args = _build_parser().parse_args()
    report = build_report(
        nominal_root=args.nominal_root,
        stress_root=args.stress_root,
        options=ReportOptions(
            expected_commit=args.expected_commit,
            expected_model_id=args.expected_model_id,
            expected_model_sha256=args.expected_model_sha256,
            expected_seeds=tuple(args.expected_seed),
            s3_seeds=tuple(args.s3_seed),
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_samples=args.bootstrap_samples,
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "issue6095_discriminability_report.json"
    markdown_path = args.output_dir / "issue6095_discriminability_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {"status": report["status"], "json": str(json_path), "markdown": str(markdown_path)}
        )
    )
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
