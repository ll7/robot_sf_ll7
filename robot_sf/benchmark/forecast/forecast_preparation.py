"""Matched oracle/ego forecast-preparation contracts and fail-closed checks.

This module prepares a small, provenance-bound design packet from existing
``simulation_trace_export.v1`` files.  It intentionally does not run a
predictor, simulator, training job, or forecast campaign.  The canonical trace
export currently contains oracle pedestrian state and robot/planner metadata,
but no tracked-agent observation tier; that absence is represented explicitly
instead of being inferred from oracle state.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    load_simulation_trace_export,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file, stable_hash

FORECAST_PREPARATION_SCHEMA_VERSION = "forecast_preparation.v1"
FORECAST_PREPARATION_ROW_SCHEMA_VERSION = "forecast_preparation_row.v1"
OBSERVATION_TIERS = ("oracle_full_state", "ego_observation")
SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_EGO_SOURCE_KEY = "tracked_agents"
DEFAULT_HORIZONS_S = (1.0,)
_CLAIM_BOUNDARY = (
    "preparation-only matched oracle-versus-ego design evidence; not forecasting performance, "
    "training, planner integration, scientific claim, or real-world forecasting"
)
_SOURCE_OWNER = "robot_sf/benchmark/forecast/forecast_preparation.py"
_SOURCE_SCHEMA = "simulation_trace_export.v1"
_FALSE_REASSURANCE_CASE_ID = "stationary_zero_ade_fde_but_robot_clearance_is_close"
_FALSE_REASSURANCE_RISK_REFERENCE_M = 0.8
_FALSE_REASSURANCE_INTERPRETATION = (
    "A stationary forecast is exactly right for this target, so ADE/FDE are zero, while "
    "the robot is within the 0.8 m diagnostic clearance reference. ADE/FDE alone do not "
    "measure robot clearance or collision relevance. This is a counterexample, not a safety claim."
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_EGO_INPUT_PARTS = frozenset(
    {
        "future",
        "future_position_m",
        "future_positions_m",
        "label",
        "target",
        "target_position_m",
        "target_time_s",
    }
)
_SPLIT_GROUP_FIELDS = ("scenario_family", "scenario_id", "seed", "episode_id")
_NEAR_DUPLICATE_POLICY = {
    "rule": "round trace positions/velocities to 2 decimals and hash the normalized trajectory",
    "scope": "an exact normalized fingerprint cannot occur in more than one split",
    "future_expansion": (
        "before training, add an RMS-aligned trajectory threshold review; this packet "
        "does not perform a forecast campaign"
    ),
}
_SOURCE_ARTIFACT_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "trace_id",
        "episode_id",
        "scenario_id",
        "scenario_family",
        "seed",
        "planner_id",
        "frame_count",
        "lineage_group_id",
        "near_duplicate_fingerprint",
        "split",
        "ego_observation_source_key",
        "ego_observation_status",
        "ego_observation_reason",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "source_lineage_id",
        "frame_step",
        "cutoff_time_s",
        "target_frame_step",
        "target_time_s",
        "actor_id",
        "horizon_s",
    }
)
_LINEAGE_FIELDS = frozenset(
    {
        "source_path",
        "source_sha256",
        "trace_id",
        "episode_id",
        "scenario_id",
        "scenario_family",
        "seed",
        "planner_id",
        "lineage_group_id",
        "split",
    }
)
_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "pair_id",
        "observation_tier",
        "availability_status",
        "identity",
        "lineage",
        "input",
        "target",
        "field_leakage_ledger",
    }
)

_BASELINE_ESTIMATES: tuple[dict[str, Any], ...] = (
    {
        "baseline_id": "stationary",
        "family": "zero_order_hold",
        "sample_size_assumption": "one actor over one forecast horizon; no empirical sample was run",
        "hardware_assumption": "CPU-only analytic estimate; no hardware benchmark or runtime measurement",
        "runtime_estimate": {
            "complexity": "O(A * H)",
            "estimated_scalar_operations_per_actor_horizon": 2,
            "estimate_unit": "scalar operations, analytic only",
        },
        "memory_estimate": {
            "complexity": "O(A * H)",
            "estimated_peak_working_bytes_per_actor": 96,
            "estimate_unit": "bytes, analytic only",
        },
        "dependency": "numpy (already a core dependency)",
        "estimate_status": "preparation_estimate_not_measured",
    },
    {
        "baseline_id": "constant_velocity",
        "family": "first_order_kinematic",
        "sample_size_assumption": "one actor over one forecast horizon; no empirical sample was run",
        "hardware_assumption": "CPU-only analytic estimate; no hardware benchmark or runtime measurement",
        "runtime_estimate": {
            "complexity": "O(A * H)",
            "estimated_scalar_operations_per_actor_horizon": 8,
            "estimate_unit": "scalar operations, analytic only",
        },
        "memory_estimate": {
            "complexity": "O(A * H)",
            "estimated_peak_working_bytes_per_actor": 128,
            "estimate_unit": "bytes, analytic only",
        },
        "dependency": "numpy (already a core dependency)",
        "estimate_status": "preparation_estimate_not_measured",
    },
    {
        "baseline_id": "constant_acceleration",
        "family": "second_order_kinematic",
        "sample_size_assumption": "one actor over one forecast horizon; no empirical sample was run",
        "hardware_assumption": "CPU-only analytic estimate; no hardware benchmark or runtime measurement",
        "runtime_estimate": {
            "complexity": "O(A * H)",
            "estimated_scalar_operations_per_actor_horizon": 18,
            "estimate_unit": "scalar operations, analytic only",
        },
        "memory_estimate": {
            "complexity": "O(A * H)",
            "estimated_peak_working_bytes_per_actor": 160,
            "estimate_unit": "bytes, analytic only",
        },
        "dependency": "numpy (already a core dependency)",
        "estimate_status": "preparation_estimate_not_measured",
    },
    {
        "baseline_id": "kalman",
        "family": "linear_gaussian_state_space",
        "sample_size_assumption": "one actor over one forecast horizon; no empirical sample was run",
        "hardware_assumption": "CPU-only analytic estimate; no hardware benchmark or runtime measurement",
        "runtime_estimate": {
            "complexity": "O(A * H * d^3), d=4 state dimensions",
            "estimated_scalar_operations_per_actor_horizon": 256,
            "estimate_unit": "scalar operations, analytic only",
        },
        "memory_estimate": {
            "complexity": "O(A * d^2)",
            "estimated_peak_working_bytes_per_actor": 2048,
            "estimate_unit": "bytes, analytic only",
        },
        "dependency": "numpy; scipy is optional and not required for the small design filter",
        "estimate_status": "preparation_estimate_not_measured",
    },
    {
        "baseline_id": "social_force",
        "family": "interaction_aware_force",
        "sample_size_assumption": "one actor over one forecast horizon with symbolic N_neighbors; no empirical sample was run",
        "hardware_assumption": "CPU-only analytic estimate; no hardware benchmark or runtime measurement",
        "runtime_estimate": {
            "complexity": "O(H * A * (A - 1))",
            "estimated_scalar_operations_per_actor_horizon": "320 + 80 * N_neighbors",
            "estimate_unit": "scalar operations, analytic only",
        },
        "memory_estimate": {
            "complexity": "O(A * H + A * N_neighbors)",
            "estimated_peak_working_bytes_per_actor": "4096 + 512 * N_neighbors",
            "estimate_unit": "bytes, analytic only",
        },
        "dependency": "local fast-pysf/pysocialforce; no new dependency is proposed",
        "estimate_status": "preparation_estimate_not_measured",
    },
)

_DEPENDENCY_LICENSE_COMPARISON: tuple[dict[str, Any], ...] = (
    {
        "component": "numpy",
        "role": "array arithmetic for stationary/CV/CA/Kalman preparation",
        "availability": "core_dependency",
        "license_or_rights": "BSD-3-Clause upstream package metadata; not a legal opinion",
        "evidence_paths": ["pyproject.toml", "docs/context/dependency_license_inventory.md"],
        "decision": "reuse_existing_dependency",
    },
    {
        "component": "scipy",
        "role": "optional linear algebra implementation for a future Kalman baseline",
        "availability": "optional_benchmark_extra",
        "license_or_rights": "BSD-3-Clause upstream package metadata; not required here",
        "evidence_paths": ["pyproject.toml", "docs/context/dependency_license_inventory.md"],
        "decision": "do_not_add_or_require",
    },
    {
        "component": "fast-pysf/pysocialforce",
        "role": "existing in-repository Social Force implementation",
        "availability": "vendored_existing",
        "license_or_rights": "MIT",
        "evidence_paths": ["fast-pysf/LICENSE", "THIRD_PARTY_NOTICES.md"],
        "decision": "reuse_only_if_future_baseline_is_separately_proven",
    },
    {
        "component": "pyrvo2",
        "role": "planner provenance for the selected ORCA source stratum",
        "availability": "optional_orca_extra",
        "license_or_rights": "Apache-2.0 for the vendored companion",
        "evidence_paths": ["third_party/python-rvo2/LICENSE", "THIRD_PARTY_NOTICES.md"],
        "decision": "record_provenance_only; no planner_integration_change",
    },
    {
        "component": "external socialforce package",
        "role": "external comparison point for a Social Force dependency",
        "availability": "not_declared",
        "license_or_rights": "not adopted; exact package/license would require a separate review",
        "evidence_paths": [
            "docs/context/issue_653_social_navigation_pyenvs_socialforce_runtime.md"
        ],
        "decision": "do_not_add_dependency",
    },
)

_FALSE_REASSURANCE_REFERENCE = (
    "docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/"
    "trace_failure_predicate_tables.json"
)


@dataclass(frozen=True)
class ForecastPreparationSourceSpec:
    """One existing trace and its explicit scenario-family/cutoff declaration."""

    path: Path | str
    scenario_family: str
    cutoff_frame_step: int
    actor_id: str | None = None


def build_forecast_preparation_packet(
    source_specs: Sequence[ForecastPreparationSourceSpec],
    *,
    repo_root: Path | str | None = None,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
    ego_source_key: str = DEFAULT_EGO_SOURCE_KEY,
    unavailable_strata: Sequence[Mapping[str, Any]] | None = None,
    checksum_paths: Sequence[Path | str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, matched oracle/ego preparation packet.

    The function consumes only existing trace exports.  When the source does
    not contain the declared ego source key, a paired ``ego_observation`` row
    is emitted with ``availability_status=not_available`` and robot context
    only.  Oracle pedestrian state is never copied into that row.

    Returns:
        JSON-serializable preparation packet.
    """
    root = Path(repo_root) if repo_root is not None else _repository_root()
    root = root.resolve()
    specs = tuple(source_specs)
    if not specs:
        raise ValueError("source_specs must be non-empty")
    horizons = _validate_horizons(horizons_s)
    ego_key = _require_text(ego_source_key, "ego_source_key")
    unavailable = _validate_unavailable_strata(unavailable_strata)
    if not _has_unavailable_dimension(unavailable, "observation_tier"):
        unavailable.append(
            {
                "dimension": "observation_tier",
                "value": "ego_observation",
                "status": "not_available",
                "reason": (
                    "simulation_trace_export.v1 has no tracked-agent observation field; "
                    "no oracle-to-ego inference is allowed"
                ),
            }
        )

    loaded: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for spec in specs:
        family = _require_text(spec.scenario_family, "scenario_family")
        if isinstance(spec.cutoff_frame_step, bool) or spec.cutoff_frame_step < 0:
            raise ValueError("cutoff_frame_step must be a non-negative integer")
        path = _resolve_repo_path(spec.path, root)
        relative_path = path.relative_to(root).as_posix()
        if relative_path in seen_paths:
            raise ValueError(f"duplicate source path: {relative_path}")
        seen_paths.add(relative_path)
        raw_payload = _load_json_object(path)
        trace = load_simulation_trace_export(path)
        cutoff = _frame_for_step(trace, spec.cutoff_frame_step)
        actor = _actor_for_frame(cutoff.pedestrians, spec.actor_id)
        dt_s = _trace_dt_s(trace)
        source_sha = sha256_file(path)
        ego_status, ego_reason = _ego_source_status(raw_payload, ego_key)
        if ego_status == "available":
            raise ValueError(
                f"{relative_path}: {ego_key} is present but no canonical ego adapter is declared; "
                "fail closed instead of changing observation semantics"
            )
        loaded.append(
            {
                "spec": spec,
                "family": family,
                "path": path,
                "relative_path": relative_path,
                "raw_payload": raw_payload,
                "trace": trace,
                "cutoff": cutoff,
                "actor": actor,
                "dt_s": dt_s,
                "source_sha256": source_sha,
                "ego_status": ego_status,
                "ego_reason": ego_reason,
                "lineage_id": stable_hash(
                    {
                        "source_sha256": source_sha,
                        "trace_id": trace.trace_id,
                        "episode_id": trace.source.episode_id,
                    }
                ),
                "near_duplicate_fingerprint": _near_duplicate_fingerprint(trace),
            }
        )

    _validate_coverage(
        loaded,
        unavailable,
        required_families=3,
        required_planners=2,
    )
    split_by_group = _assign_group_splits(loaded)
    rows: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    for item in sorted(loaded, key=lambda entry: entry["relative_path"]):
        trace: SimulationTraceExport = item["trace"]
        group_id = _lineage_group_id(item)
        split = split_by_group[group_id]
        source_artifacts.append(
            {
                "path": item["relative_path"],
                "sha256": item["source_sha256"],
                "size_bytes": item["path"].stat().st_size,
                "trace_id": trace.trace_id,
                "episode_id": trace.source.episode_id,
                "scenario_id": trace.source.scenario_id,
                "scenario_family": item["family"],
                "seed": trace.source.seed,
                "planner_id": trace.source.planner_id,
                "frame_count": len(trace.frames),
                "lineage_group_id": group_id,
                "near_duplicate_fingerprint": item["near_duplicate_fingerprint"],
                "split": split,
                "ego_observation_source_key": ego_key,
                "ego_observation_status": item["ego_status"],
                "ego_observation_reason": item["ego_reason"],
            }
        )
        for horizon_s in horizons:
            target = _target_for_horizon(
                trace,
                cutoff_step=item["cutoff"].step,
                horizon_s=horizon_s,
                dt_s=item["dt_s"],
                actor_id=item["actor"]["actor_id"],
            )
            identity = {
                "source_lineage_id": item["lineage_id"],
                "frame_step": item["cutoff"].step,
                "cutoff_time_s": item["cutoff"].time_s,
                "target_frame_step": target["frame_step"],
                "target_time_s": target["time_s"],
                "actor_id": item["actor"]["actor_id"],
                "horizon_s": float(horizon_s),
            }
            pair_id = f"pair-{stable_hash(identity)[:24]}"
            lineage = {
                "source_path": item["relative_path"],
                "source_sha256": item["source_sha256"],
                "trace_id": trace.trace_id,
                "episode_id": trace.source.episode_id,
                "scenario_id": trace.source.scenario_id,
                "scenario_family": item["family"],
                "seed": trace.source.seed,
                "planner_id": trace.source.planner_id,
                "lineage_group_id": group_id,
                "split": split,
            }
            rows.append(
                _build_row(
                    pair_id=pair_id,
                    identity=identity,
                    lineage=lineage,
                    trace=trace,
                    actor=item["actor"],
                    target=target,
                    observation_tier="oracle_full_state",
                    availability_status="available",
                    ego_source_key=ego_key,
                    ego_reason=None,
                )
            )
            rows.append(
                _build_row(
                    pair_id=pair_id,
                    identity=identity,
                    lineage=lineage,
                    trace=trace,
                    actor=item["actor"],
                    target=target,
                    observation_tier="ego_observation",
                    availability_status=item["ego_status"],
                    ego_source_key=ego_key,
                    ego_reason=item["ego_reason"],
                )
            )

    rows.sort(key=lambda row: (row["pair_id"], OBSERVATION_TIERS.index(row["observation_tier"])))
    evidence_references = _build_evidence_references(root)
    packet = {
        "schema_version": FORECAST_PREPARATION_SCHEMA_VERSION,
        "issue": 7602,
        "claim_boundary": _CLAIM_BOUNDARY,
        "evidence_status": "diagnostic-only",
        "observation_contract_changed": False,
        "source_owner": _SOURCE_OWNER,
        "source_schema": _SOURCE_SCHEMA,
        "ego_observation_source_key": ego_key,
        "row_schema_version": FORECAST_PREPARATION_ROW_SCHEMA_VERSION,
        "horizons_s": horizons,
        "pair_count": len(rows) // len(OBSERVATION_TIERS),
        "row_count": len(rows),
        "source_artifacts": source_artifacts,
        "evidence_references": evidence_references,
        "coverage": {
            "scenario_families": sorted({item["family"] for item in loaded}),
            "planners": sorted({item["trace"].source.planner_id for item in loaded}),
            "observation_tiers": list(OBSERVATION_TIERS),
            "ego_observation_status": "not_available",
            "unavailable_strata": unavailable,
        },
        "pair_identity_fields": [
            "source_lineage_id",
            "frame_step",
            "cutoff_time_s",
            "target_frame_step",
            "target_time_s",
            "actor_id",
            "horizon_s",
        ],
        "field_leakage_ledger": {
            "time_roles": {
                "cutoff": "input state at or before the declared frame time",
                "target": "supervision-only future label at cutoff_time_s + horizon_s",
            },
            "robot_available": (
                "true means the field is part of the robot/ego context at cutoff; false means "
                "privileged or future-only data unavailable to the robot"
            ),
            "future_target": (
                "true only for supervision labels; any true field under row.input fails validation"
            ),
        },
        "split_policy": {
            "strategy": "deterministic_grouped_split",
            "split_names": list(SPLIT_NAMES),
            "group_fields": list(_SPLIT_GROUP_FIELDS),
            "group_id_definition": "scenario_family:scenario_id:seed:episode_id",
            "near_duplicate_policy": deepcopy(_NEAR_DUPLICATE_POLICY),
            "assignments": {
                group_id: split_by_group[group_id] for group_id in sorted(split_by_group)
            },
        },
        "runtime_memory_estimates": [deepcopy(item) for item in _BASELINE_ESTIMATES],
        "dependency_license_comparison": [
            deepcopy(item) for item in _DEPENDENCY_LICENSE_COMPARISON
        ],
        "ade_fde_false_reassurance_case": _build_false_reassurance_case(loaded, root),
        "rows": rows,
        "sha256_coverage": _build_sha256_coverage(root, checksum_paths),
    }
    validate_forecast_preparation_packet(packet, repo_root=root, verify_checksums=False)
    return packet


def validate_forecast_preparation_packet(  # noqa: C901
    payload: Mapping[str, Any],
    *,
    repo_root: Path | str | None = None,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """Validate packet shape, pair identity, leakage, provenance, and checksums.

    Returns:
        Compact validation summary.  All violations raise ``ValueError``.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("packet must be a mapping")
    if payload.get("schema_version") != FORECAST_PREPARATION_SCHEMA_VERSION:
        raise ValueError("schema_version must be forecast_preparation.v1")
    if payload.get("issue") != 7602:
        raise ValueError("issue must be 7602")
    if payload.get("claim_boundary") != _CLAIM_BOUNDARY:
        raise ValueError("claim_boundary must preserve the preparation-only boundary")
    if payload.get("evidence_status") != "diagnostic-only":
        raise ValueError("evidence_status must be diagnostic-only")
    if payload.get("observation_contract_changed") is not False:
        raise ValueError("observation_contract_changed must be false")
    if payload.get("source_owner") != _SOURCE_OWNER:
        raise ValueError("source_owner is not canonical")
    if payload.get("source_schema") != _SOURCE_SCHEMA:
        raise ValueError("source_schema is not canonical")
    if payload.get("row_schema_version") != FORECAST_PREPARATION_ROW_SCHEMA_VERSION:
        raise ValueError("row_schema_version is not canonical")
    horizons = _validate_horizons(payload.get("horizons_s"))
    rows = payload.get("rows")
    source_artifacts = payload.get("source_artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        raise ValueError("source_artifacts must be a non-empty list")
    if payload.get("row_count") != len(rows):
        raise ValueError("row_count does not match rows")
    if payload.get("pair_count") != len(rows) // len(OBSERVATION_TIERS):
        raise ValueError("pair_count does not match rows")
    if len(rows) % len(OBSERVATION_TIERS):
        raise ValueError("rows must contain two rows per pair")

    root = (Path(repo_root) if repo_root is not None else _repository_root()).resolve()
    ego_source_key = _require_text(
        payload.get("ego_observation_source_key"), "ego_observation_source_key"
    )
    source_by_path = _validate_source_artifacts(
        source_artifacts, root, ego_source_key=ego_source_key
    )
    _validate_coverage_from_packet(payload, source_artifacts)
    _validate_rows(payload, rows, source_by_path, root)
    _validate_horizon_rows(rows, horizons)
    _validate_split_policy(payload, source_artifacts)
    _validate_deterministic_split_assignments(payload, source_artifacts)
    _validate_evidence_references(payload.get("evidence_references"), root)
    _validate_estimates(payload.get("runtime_memory_estimates"))
    _validate_dependencies(payload.get("dependency_license_comparison"), root)
    _validate_false_reassurance_case(
        payload.get("ade_fde_false_reassurance_case"),
        root,
        source_by_path,
    )
    if verify_checksums:
        _validate_sha256_coverage(payload.get("sha256_coverage"), root)
    return {
        "status": "passed",
        "evidence_status": "diagnostic-only",
        "source_trace_count": len(source_artifacts),
        "pair_count": payload["pair_count"],
        "row_count": len(rows),
        "scenario_family_count": len(payload["coverage"]["scenario_families"]),
        "planner_count": len(payload["coverage"]["planners"]),
        "ego_observation_status": payload["coverage"]["ego_observation_status"],
    }


def _build_row(  # noqa: PLR0913
    *,
    pair_id: str,
    identity: dict[str, Any],
    lineage: dict[str, Any],
    trace: SimulationTraceExport,
    actor: dict[str, Any],
    target: dict[str, Any],
    observation_tier: str,
    availability_status: str,
    ego_source_key: str,
    ego_reason: str | None,
) -> dict[str, Any]:
    robot = _frame_for_step(trace, int(identity["frame_step"])).robot
    robot_input = {
        "robot_position_m": [float(value) for value in robot["position"]],
        "robot_velocity_mps": [float(value) for value in robot["velocity"]],
        "robot_heading_rad": float(robot["heading"]),
    }
    if observation_tier == "oracle_full_state":
        input_fields = {
            "pedestrian_position_m": [float(value) for value in actor["position"]],
            "pedestrian_velocity_mps": [float(value) for value in actor["velocity"]],
            **robot_input,
        }
        pedestrian_status = "available"
        pedestrian_owner = "simulation_trace_export.pedestrians"
        pedestrian_robot_available = False
    elif observation_tier == "ego_observation":
        input_fields = robot_input
        pedestrian_status = "not_available"
        pedestrian_owner = f"declared_ego_source.{ego_source_key}"
        pedestrian_robot_available = False
    else:  # pragma: no cover - guarded by the caller's fixed tier values
        raise ValueError(f"unsupported observation tier: {observation_tier}")

    ledger = [
        _ledger_entry(
            "input.pedestrian_position_m",
            owner=pedestrian_owner,
            unit="m",
            time_role="cutoff",
            robot_available=pedestrian_robot_available,
            future_target=False,
            status=pedestrian_status,
            reason=ego_reason if pedestrian_status == "not_available" else None,
        ),
        _ledger_entry(
            "input.pedestrian_velocity_mps",
            owner=pedestrian_owner,
            unit="m/s",
            time_role="cutoff",
            robot_available=pedestrian_robot_available,
            future_target=False,
            status=pedestrian_status,
            reason=ego_reason if pedestrian_status == "not_available" else None,
        ),
        _ledger_entry(
            "input.robot_position_m",
            owner="simulation_trace_export.robot",
            unit="m",
            time_role="cutoff",
            robot_available=True,
            future_target=False,
            status="available",
        ),
        _ledger_entry(
            "input.robot_velocity_mps",
            owner="simulation_trace_export.robot",
            unit="m/s",
            time_role="cutoff",
            robot_available=True,
            future_target=False,
            status="available",
        ),
        _ledger_entry(
            "input.robot_heading_rad",
            owner="simulation_trace_export.robot",
            unit="rad",
            time_role="cutoff",
            robot_available=True,
            future_target=False,
            status="available",
        ),
        _ledger_entry(
            "target.future_position_m",
            owner="simulation_trace_export.pedestrians",
            unit="m",
            time_role="target",
            robot_available=False,
            future_target=True,
            status="available",
        ),
    ]
    return {
        "schema_version": FORECAST_PREPARATION_ROW_SCHEMA_VERSION,
        "pair_id": pair_id,
        "observation_tier": observation_tier,
        "availability_status": availability_status,
        "identity": identity,
        "lineage": lineage,
        "input": input_fields,
        "target": {
            "future_position_m": target["position"],
            "source": "simulation_trace_export.pedestrians",
        },
        "field_leakage_ledger": ledger,
    }


def _ledger_entry(
    field: str,
    *,
    owner: str,
    unit: str,
    time_role: str,
    robot_available: bool,
    future_target: bool,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    entry = {
        "field": field,
        "owner": owner,
        "unit": unit,
        "time_role": time_role,
        "robot_available": robot_available,
        "future_target": future_target,
        "status": status,
    }
    if reason:
        entry["reason"] = reason
    return entry


def _validate_rows(  # noqa: C901, PLR0912, PLR0915
    payload: Mapping[str, Any],
    rows: list[Any],
    source_by_path: dict[str, dict[str, Any]],
    root: Path,
) -> None:
    expected_fields = tuple(payload.get("pair_identity_fields", ()))
    if expected_fields != (
        "source_lineage_id",
        "frame_step",
        "cutoff_time_s",
        "target_frame_step",
        "target_time_s",
        "actor_id",
        "horizon_s",
    ):
        raise ValueError("pair_identity_fields are not the canonical preparation fields")
    rows_by_pair: dict[str, list[dict[str, Any]]] = {}
    seen_row_keys: set[tuple[str, str]] = set()
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            raise ValueError(f"rows[{index}] must be a mapping")
        if set(raw_row) != _ROW_FIELDS:
            raise ValueError(f"rows[{index}] fields are not canonical")
        if raw_row.get("schema_version") != FORECAST_PREPARATION_ROW_SCHEMA_VERSION:
            raise ValueError(f"rows[{index}] schema_version is not canonical")
        pair_id = _require_text(raw_row.get("pair_id"), f"rows[{index}].pair_id")
        tier = _require_text(raw_row.get("observation_tier"), f"rows[{index}].observation_tier")
        if tier not in OBSERVATION_TIERS:
            raise ValueError(f"rows[{index}] has unsupported observation_tier: {tier}")
        row_key = (pair_id, tier)
        if row_key in seen_row_keys:
            raise ValueError(f"duplicate row identity: {pair_id}/{tier}")
        seen_row_keys.add(row_key)
        identity = raw_row.get("identity")
        lineage = raw_row.get("lineage")
        if not isinstance(identity, dict) or not isinstance(lineage, dict):
            raise ValueError(f"rows[{index}] requires identity and lineage mappings")
        if set(identity) != _IDENTITY_FIELDS:
            raise ValueError(f"rows[{index}] identity fields are not canonical")
        if set(lineage) != _LINEAGE_FIELDS:
            raise ValueError(f"rows[{index}] lineage fields are not canonical")
        source_path = _require_text(
            lineage.get("source_path"), f"rows[{index}].lineage.source_path"
        )
        if Path(source_path).is_absolute():
            raise ValueError(f"rows[{index}] lineage source_path must be repository-relative")
        source = source_by_path.get(source_path)
        if source is None:
            raise ValueError(f"rows[{index}] references unknown source path: {source_path}")
        if lineage.get("source_sha256") != source["sha256"]:
            raise ValueError(f"rows[{index}] source SHA-256 does not match source artifact")
        if lineage.get("split") != source["split"]:
            raise ValueError(f"rows[{index}] split does not match source artifact")
        if identity.get("source_lineage_id") != _lineage_id_from_source(source):
            raise ValueError(f"rows[{index}] source_lineage_id does not match source artifact")
        for field in (
            "trace_id",
            "episode_id",
            "scenario_id",
            "scenario_family",
            "seed",
            "planner_id",
            "lineage_group_id",
        ):
            if lineage.get(field) != source.get(field):
                raise ValueError(f"rows[{index}] lineage metadata does not match source: {field}")
        expected_pair_id = f"pair-{stable_hash(identity)[:24]}"
        if pair_id != expected_pair_id:
            raise ValueError(f"rows[{index}] pair_id does not match identity")
        input_fields = raw_row.get("input")
        if not isinstance(input_fields, dict):
            raise ValueError(f"rows[{index}].input must be a mapping")
        if tier == "ego_observation":
            _validate_ego_input(input_fields, raw_row, index)
            if raw_row.get("availability_status") != "not_available":
                raise ValueError(
                    "ego_observation rows must be explicitly not_available in this packet"
                )
        elif raw_row.get("availability_status") != "available":
            raise ValueError("oracle_full_state rows must be available")
        _validate_row_ledger(raw_row, index)
        rows_by_pair.setdefault(pair_id, []).append(raw_row)

    for pair_id, pair_rows in rows_by_pair.items():
        tiers = {row["observation_tier"] for row in pair_rows}
        if tiers != set(OBSERVATION_TIERS) or len(pair_rows) != len(OBSERVATION_TIERS):
            raise ValueError(f"pair {pair_id} must contain exactly oracle and ego rows")
        first_identity = pair_rows[0]["identity"]
        first_lineage = pair_rows[0]["lineage"]
        first_target = pair_rows[0]["target"]
        for row in pair_rows[1:]:
            if row["identity"] != first_identity:
                raise ValueError(f"pair {pair_id} has mismatched pair identity")
            if row["lineage"] != first_lineage:
                raise ValueError(f"pair {pair_id} has mismatched lineage")
            if row["target"] != first_target:
                raise ValueError(f"pair {pair_id} has mismatched future target")
        source = source_by_path[first_lineage["source_path"]]
        _validate_pair_against_source(
            pair_rows,
            identity=first_identity,
            target=first_target,
            source=source,
            root=root,
            pair_id=pair_id,
        )


def _validate_horizon_rows(rows: list[Any], horizons: list[float]) -> None:
    """Require every source/actor pair to carry every declared horizon."""
    expected = set(horizons)
    horizons_by_pair: dict[tuple[str, str], set[float]] = {}
    for index, row in enumerate(rows):
        identity = row["identity"]
        lineage = row["lineage"]
        horizon = _require_finite_number(identity.get("horizon_s"), f"rows[{index}].horizon_s")
        key = (lineage["source_path"], identity["actor_id"])
        horizons_by_pair.setdefault(key, set()).add(horizon)
    for key, actual in horizons_by_pair.items():
        if actual != expected:
            raise ValueError(
                f"declared horizons do not match row coverage for source/actor {key[0]}/{key[1]}"
            )


def _validate_ego_input(input_fields: dict[str, Any], row: dict[str, Any], index: int) -> None:
    for key in _walk_mapping_keys(input_fields):
        normalized = key.lower()
        if normalized in _FORBIDDEN_EGO_INPUT_PARTS or any(
            part in normalized for part in ("future", "target", "label")
        ):
            raise ValueError(f"rows[{index}] future/target field leaked into ego input: {key}")
    if "pedestrian_position_m" in input_fields or "pedestrian_velocity_mps" in input_fields:
        raise ValueError(f"rows[{index}] ego input contains privileged pedestrian state")


def _validate_pair_against_source(
    pair_rows: list[dict[str, Any]],
    *,
    identity: dict[str, Any],
    target: Any,
    source: Mapping[str, Any],
    root: Path,
    pair_id: str,
) -> None:
    """Recompute pair identity, target, and input values from the source trace."""
    source_path = _require_text(source.get("path"), f"pair {pair_id}.source_path")
    trace = load_simulation_trace_export(_resolve_repo_path(source_path, root))
    frame_step = _require_int(identity.get("frame_step"), f"pair {pair_id}.frame_step")
    target_frame_step = _require_int(
        identity.get("target_frame_step"), f"pair {pair_id}.target_frame_step"
    )
    cutoff_time_s = _require_finite_number(
        identity.get("cutoff_time_s"), f"pair {pair_id}.cutoff_time_s"
    )
    target_time_s = _require_finite_number(
        identity.get("target_time_s"), f"pair {pair_id}.target_time_s"
    )
    horizon_s = _require_finite_number(identity.get("horizon_s"), f"pair {pair_id}.horizon_s")
    if horizon_s <= 0.0:
        raise ValueError(f"pair {pair_id}.horizon_s must be positive")
    actor_id = _require_text(identity.get("actor_id"), f"pair {pair_id}.actor_id")
    cutoff = _frame_for_step(trace, frame_step)
    actor = _actor_for_frame(cutoff.pedestrians, actor_id)
    expected_target = _target_for_horizon(
        trace,
        cutoff_step=frame_step,
        horizon_s=horizon_s,
        dt_s=_trace_dt_s(trace),
        actor_id=actor_id,
    )
    if target_frame_step != expected_target["frame_step"]:
        raise ValueError(f"pair {pair_id}.target_frame_step does not match source trace")
    if not math.isclose(cutoff_time_s, cutoff.time_s, abs_tol=1e-9):
        raise ValueError(f"pair {pair_id}.cutoff_time_s does not match source trace")
    if not math.isclose(target_time_s, expected_target["time_s"], abs_tol=1e-9):
        raise ValueError(f"pair {pair_id}.target_time_s does not match source trace")
    if not math.isclose(
        target_time_s - cutoff_time_s,
        horizon_s,
        abs_tol=1e-9,
    ):
        raise ValueError(f"pair {pair_id}.horizon_s does not match source times")
    expected_target_payload = {
        "future_position_m": expected_target["position"],
        "source": "simulation_trace_export.pedestrians",
    }
    if target != expected_target_payload:
        raise ValueError(f"pair {pair_id}.target does not match source trace")
    for index, row in enumerate(pair_rows):
        _validate_row_input_against_source(
            row,
            index=index,
            cutoff=cutoff,
            actor=actor,
            pair_id=pair_id,
        )


def _validate_row_input_against_source(
    row: dict[str, Any],
    *,
    index: int,
    cutoff: Any,
    actor: Mapping[str, Any],
    pair_id: str,
) -> None:
    input_fields = row["input"]
    robot = cutoff.robot
    expected_robot = {
        "robot_position_m": [float(value) for value in robot["position"]],
        "robot_velocity_mps": [float(value) for value in robot["velocity"]],
        "robot_heading_rad": float(robot["heading"]),
    }
    tier = row["observation_tier"]
    if tier == "oracle_full_state":
        expected = {
            "pedestrian_position_m": [float(value) for value in actor["position"]],
            "pedestrian_velocity_mps": [float(value) for value in actor["velocity"]],
            **expected_robot,
        }
    elif tier == "ego_observation":
        expected = expected_robot
    else:  # pragma: no cover - fixed-tier validation runs before this helper
        raise ValueError(f"pair {pair_id} has unsupported observation tier: {tier}")
    if set(input_fields) != set(expected):
        raise ValueError(f"rows[{index}] input fields do not match the declared observation tier")
    for field, expected_value in expected.items():
        actual_value = input_fields.get(field)
        if isinstance(expected_value, list):
            if not _vectors_close(actual_value, expected_value):
                raise ValueError(f"rows[{index}] input field is not bound to source: {field}")
        elif not isinstance(actual_value, (int, float)) or isinstance(actual_value, bool):
            raise ValueError(f"rows[{index}] input field is not numeric: {field}")
        elif not math.isclose(float(actual_value), expected_value, abs_tol=1e-9):
            raise ValueError(f"rows[{index}] input field is not bound to source: {field}")


def _vectors_close(actual: Any, expected: Sequence[float]) -> bool:
    return (
        isinstance(actual, list)
        and len(actual) == len(expected)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and math.isclose(float(value), expected_value, abs_tol=1e-9)
            for value, expected_value in zip(actual, expected, strict=True)
        )
    )


def _validate_row_ledger(row: dict[str, Any], index: int) -> None:  # noqa: C901, PLR0912
    ledger = row.get("field_leakage_ledger")
    if not isinstance(ledger, list):
        raise ValueError(f"rows[{index}].field_leakage_ledger must be a list")
    required_fields = {
        "input.pedestrian_position_m",
        "input.pedestrian_velocity_mps",
        "input.robot_position_m",
        "input.robot_velocity_mps",
        "input.robot_heading_rad",
        "target.future_position_m",
    }
    pedestrian_status = (
        "available" if row["observation_tier"] == "oracle_full_state" else "not_available"
    )
    expected_metadata = {
        "input.pedestrian_position_m": ("m", "cutoff", False, False, pedestrian_status),
        "input.pedestrian_velocity_mps": ("m/s", "cutoff", False, False, pedestrian_status),
        "input.robot_position_m": ("m", "cutoff", True, False, "available"),
        "input.robot_velocity_mps": ("m/s", "cutoff", True, False, "available"),
        "input.robot_heading_rad": ("rad", "cutoff", True, False, "available"),
        "target.future_position_m": ("m", "target", False, True, "available"),
    }
    actual_fields = set()
    for entry in ledger:
        if not isinstance(entry, dict):
            raise ValueError(f"rows[{index}] ledger entries must be mappings")
        field = _require_text(entry.get("field"), f"rows[{index}].ledger.field")
        actual_fields.add(field)
        for key in ("owner", "unit", "time_role", "status"):
            _require_text(entry.get(key), f"rows[{index}].ledger.{key}")
        if not isinstance(entry.get("robot_available"), bool):
            raise ValueError(f"rows[{index}] ledger.robot_available must be boolean")
        if not isinstance(entry.get("future_target"), bool):
            raise ValueError(f"rows[{index}] ledger.future_target must be boolean")
        if field.startswith("input.") and entry["future_target"]:
            raise ValueError(f"rows[{index}] future target marked inside input ledger")
        expected = expected_metadata.get(field)
        if expected is not None:
            unit, time_role, robot_available, future_target, status = expected
            if (
                entry["unit"] != unit
                or entry["time_role"] != time_role
                or entry["robot_available"] != robot_available
                or entry["future_target"] != future_target
                or entry["status"] != status
            ):
                raise ValueError(f"rows[{index}] ledger semantics drift: {field}")
            if field.startswith("input.pedestrian_"):
                owner = entry["owner"]
                if row["observation_tier"] == "oracle_full_state":
                    owner_matches = owner == "simulation_trace_export.pedestrians"
                else:
                    owner_matches = owner.startswith("declared_ego_source.") and len(owner) > len(
                        "declared_ego_source."
                    )
                if not owner_matches:
                    raise ValueError(f"rows[{index}] ledger owner drift: {field}")
            elif field == "target.future_position_m":
                if entry["owner"] != "simulation_trace_export.pedestrians":
                    raise ValueError(f"rows[{index}] ledger owner drift: {field}")
            elif entry["owner"] != "simulation_trace_export.robot":
                raise ValueError(f"rows[{index}] ledger owner drift: {field}")
        if entry["status"] == "not_available" and not entry.get("reason"):
            raise ValueError(f"rows[{index}] unavailable ledger field needs a reason")
    if len(actual_fields) != len(ledger):
        raise ValueError(f"rows[{index}] field ledger contains duplicate fields")
    if actual_fields != required_fields:
        raise ValueError(f"rows[{index}] field ledger coverage mismatch")


def _validate_split_policy(  # noqa: C901
    payload: Mapping[str, Any], source_artifacts: list[Any]
) -> None:
    policy = payload.get("split_policy")
    if not isinstance(policy, dict):
        raise ValueError("split_policy must be a mapping")
    if policy.get("strategy") != "deterministic_grouped_split":
        raise ValueError("split_policy strategy must be deterministic_grouped_split")
    if policy.get("split_names") != list(SPLIT_NAMES):
        raise ValueError("split_policy split_names are not canonical")
    if policy.get("group_fields") != list(_SPLIT_GROUP_FIELDS):
        raise ValueError("split_policy group_fields are not canonical")
    if policy.get("group_id_definition") != ":".join(_SPLIT_GROUP_FIELDS):
        raise ValueError("split_policy group_id_definition is not canonical")
    if policy.get("near_duplicate_policy") != _NEAR_DUPLICATE_POLICY:
        raise ValueError("split_policy near_duplicate_policy is not canonical")
    assignments = policy.get("assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("split_policy.assignments must be a non-empty mapping")
    group_splits: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    for artifact in source_artifacts:
        group_id = _require_text(artifact.get("lineage_group_id"), "lineage_group_id")
        split = _require_text(artifact.get("split"), "split")
        if split not in SPLIT_NAMES:
            raise ValueError(f"unsupported split: {split}")
        previous_split = group_splits.get(group_id)
        if previous_split is not None and previous_split != split:
            raise ValueError(f"group leakage across splits: {group_id}")
        group_splits[group_id] = split
        if assignments.get(group_id) != split:
            raise ValueError(f"split assignment drift for group: {group_id}")
        fingerprint = _require_text(
            artifact.get("near_duplicate_fingerprint"), "near_duplicate_fingerprint"
        )
        previous_group = fingerprints.get(fingerprint)
        if previous_group is not None and group_splits[previous_group] != split:
            raise ValueError(f"near-duplicate trajectory leakage across splits: {fingerprint}")
        fingerprints[fingerprint] = group_id


def _validate_deterministic_split_assignments(
    payload: Mapping[str, Any], source_artifacts: list[dict[str, Any]]
) -> None:
    """Recompute the declared grouped split instead of trusting packet assignments."""
    policy = payload["split_policy"]
    assignments = policy["assignments"]
    group_ids = sorted({artifact["lineage_group_id"] for artifact in source_artifacts})
    expected = {
        group_id: SPLIT_NAMES[index % len(SPLIT_NAMES)] for index, group_id in enumerate(group_ids)
    }
    if assignments != expected:
        raise ValueError("split assignments drift from the deterministic grouped policy")
    for artifact in source_artifacts:
        group_id = artifact["lineage_group_id"]
        if artifact["split"] != expected[group_id]:
            raise ValueError(f"source split drifts from deterministic grouped policy: {group_id}")


def _validate_source_artifacts(
    source_artifacts: list[Any],
    root: Path,
    *,
    ego_source_key: str,
) -> dict[str, dict[str, Any]]:
    source_by_path: dict[str, dict[str, Any]] = {}
    for index, raw_artifact in enumerate(source_artifacts):
        if not isinstance(raw_artifact, dict):
            raise ValueError(f"source_artifacts[{index}] must be a mapping")
        if set(raw_artifact) != _SOURCE_ARTIFACT_FIELDS:
            raise ValueError(f"source_artifacts[{index}] fields are not canonical")
        relative_path = _require_text(raw_artifact.get("path"), f"source_artifacts[{index}].path")
        if Path(relative_path).is_absolute():
            raise ValueError(f"source_artifacts[{index}].path must be repository-relative")
        path = _resolve_repo_path(relative_path, root)
        expected_sha = _require_text(
            raw_artifact.get("sha256"), f"source_artifacts[{index}].sha256"
        )
        if not _SHA256_RE.fullmatch(expected_sha):
            raise ValueError(f"invalid source SHA-256: {relative_path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"source SHA-256 mismatch: {relative_path}")
        trace = load_simulation_trace_export(path)
        _validate_source_artifact_metadata(
            raw_artifact,
            index=index,
            relative_path=relative_path,
            path=path,
            trace=trace,
            ego_source_key=ego_source_key,
        )
        source_by_path[relative_path] = raw_artifact
    if len(source_by_path) != len(source_artifacts):
        raise ValueError("duplicate source artifact path")
    return source_by_path


def _validate_source_artifact_metadata(  # noqa: C901
    raw_artifact: dict[str, Any],
    *,
    index: int,
    relative_path: str,
    path: Path,
    trace: SimulationTraceExport,
    ego_source_key: str,
) -> None:
    for field, actual in (
        ("trace_id", trace.trace_id),
        ("episode_id", trace.source.episode_id),
        ("scenario_id", trace.source.scenario_id),
        ("seed", trace.source.seed),
        ("planner_id", trace.source.planner_id),
    ):
        if raw_artifact.get(field) != actual:
            raise ValueError(f"source metadata drift for {relative_path}: {field}")
    scenario_family = _require_text(
        raw_artifact.get("scenario_family"), f"source_artifacts[{index}].scenario_family"
    )
    expected_group_id = (
        f"{scenario_family}:{trace.source.scenario_id}:{trace.source.seed}:"
        f"{trace.source.episode_id}"
    )
    if raw_artifact.get("lineage_group_id") != expected_group_id:
        raise ValueError(f"source metadata drift for {relative_path}: lineage_group_id")
    if raw_artifact.get("frame_count") != len(trace.frames):
        raise ValueError(f"source metadata drift for {relative_path}: frame_count")
    if raw_artifact.get("size_bytes") != path.stat().st_size:
        raise ValueError(f"source metadata drift for {relative_path}: size_bytes")
    if raw_artifact.get("near_duplicate_fingerprint") != _near_duplicate_fingerprint(trace):
        raise ValueError(f"source metadata drift for {relative_path}: near_duplicate_fingerprint")
    if raw_artifact.get("ego_observation_source_key") != ego_source_key:
        raise ValueError(f"source metadata drift for {relative_path}: ego_observation_source_key")
    raw_payload = _load_json_object(path)
    ego_status, ego_reason = _ego_source_status(raw_payload, ego_source_key)
    if ego_status != "not_available":
        raise ValueError(f"source ego observation field is present: {relative_path}")
    ego_status = raw_artifact.get("ego_observation_status")
    if ego_status != "not_available":
        raise ValueError("source ego_observation_status must be not_available")
    if raw_artifact.get("ego_observation_reason") != ego_reason:
        raise ValueError(f"source metadata drift for {relative_path}: ego_observation_reason")


def _validate_coverage_from_packet(
    payload: Mapping[str, Any],
    source_artifacts: list[dict[str, Any]],
) -> None:
    coverage = payload.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("coverage must be a mapping")
    families = sorted({artifact.get("scenario_family") for artifact in source_artifacts})
    planners = sorted({artifact.get("planner_id") for artifact in source_artifacts})
    if coverage.get("scenario_families") != families:
        raise ValueError("scenario family coverage drift")
    if coverage.get("planners") != planners:
        raise ValueError("planner coverage drift")
    if coverage.get("observation_tiers") != list(OBSERVATION_TIERS):
        raise ValueError("observation tier coverage drift")
    if coverage.get("ego_observation_status") != "not_available":
        raise ValueError("ego observation availability must remain explicit")
    unavailable = coverage.get("unavailable_strata")
    if not isinstance(unavailable, list):
        raise ValueError("coverage.unavailable_strata must be a list")
    _validate_unavailable_strata(unavailable)
    if len(families) < 3 and not _has_unavailable_dimension(unavailable, "scenario_family"):
        raise ValueError("fewer than three scenario families without an unavailable stratum")
    if len(planners) < 2 and not _has_unavailable_dimension(unavailable, "planner"):
        raise ValueError("fewer than two planners without an unavailable stratum")
    if not _has_unavailable_dimension(unavailable, "observation_tier"):
        raise ValueError("ego observation unavailable stratum is missing")


def _validate_estimates(estimates: Any) -> None:
    if not isinstance(estimates, list):
        raise ValueError("runtime_memory_estimates must be a list")
    expected = {item["baseline_id"] for item in _BASELINE_ESTIMATES}
    actual = set()
    expected_by_id = {item["baseline_id"]: item for item in _BASELINE_ESTIMATES}
    for item in estimates:
        if not isinstance(item, dict):
            raise ValueError("baseline estimates must be mappings")
        baseline_id = _require_text(item.get("baseline_id"), "baseline_id")
        if baseline_id not in expected_by_id or item != expected_by_id[baseline_id]:
            raise ValueError(f"baseline estimate contract drift: {baseline_id}")
        actual.add(baseline_id)
    if len(actual) != len(estimates) or actual != expected:
        raise ValueError(
            "baseline estimate coverage must include stationary, CV, CA, Kalman, Social Force"
        )


def _validate_dependencies(dependencies: Any, root: Path) -> None:
    if not isinstance(dependencies, list):
        raise ValueError("dependency_license_comparison must be a list")
    expected = {item["component"] for item in _DEPENDENCY_LICENSE_COMPARISON}
    actual = set()
    expected_by_component = {item["component"]: item for item in _DEPENDENCY_LICENSE_COMPARISON}
    for item in dependencies:
        if not isinstance(item, dict):
            raise ValueError("dependency comparison rows must be mappings")
        component = _require_text(item.get("component"), "dependency component")
        if component not in expected_by_component or item != expected_by_component[component]:
            raise ValueError(f"dependency/license comparison contract drift: {component}")
        actual.add(component)
        paths = item.get("evidence_paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"dependency {component} has no evidence paths")
        for path in paths:
            _resolve_repo_path(path, root)
    if len(actual) != len(dependencies) or actual != expected:
        raise ValueError("dependency/license comparison coverage drift")


def _validate_evidence_references(references: Any, root: Path) -> None:
    """Bind cited evidence paths and digests to the current repository bytes."""
    expected = _build_evidence_references(root)
    if references != expected:
        raise ValueError("evidence_references are not bound to the cited repository bytes")


def _validate_false_reassurance_case(  # noqa: C901, PLR0912
    case: Any,
    root: Path,
    source_by_path: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(case, dict):
        raise ValueError("ade_fde_false_reassurance_case is required")
    if case.get("status") != "analytic_trace_backed_diagnostic_only":
        raise ValueError("false-reassurance case status is not diagnostic-only")
    if case.get("case_id") != _FALSE_REASSURANCE_CASE_ID:
        raise ValueError("false-reassurance case id is not canonical")
    if case.get("interpretation") != _FALSE_REASSURANCE_INTERPRETATION:
        raise ValueError("false-reassurance case interpretation is not canonical")
    source_path = _require_text(case.get("source_path"), "false case source_path")
    source_artifact = source_by_path.get(source_path)
    if source_artifact is None:
        raise ValueError("false case source is not one of the packet source artifacts")
    if source_artifact.get("scenario_family") != "crossing_proxy":
        raise ValueError("false case source must be the crossing_proxy stratum")
    source_file = _resolve_repo_path(source_path, root)
    expected_source_sha = _require_text(case.get("source_sha256"), "false case source_sha256")
    if not _SHA256_RE.fullmatch(expected_source_sha):
        raise ValueError("false case source_sha256 is not a SHA-256 digest")
    if expected_source_sha != source_artifact.get("sha256"):
        raise ValueError("false case source SHA-256 does not match source artifact")
    if sha256_file(source_file) != expected_source_sha:
        raise ValueError("false case source SHA-256 does not match source file")
    source = load_simulation_trace_export(source_file)
    cutoff_step = case.get("cutoff_frame_step")
    target_step = case.get("target_frame_step")
    if (
        isinstance(cutoff_step, bool)
        or isinstance(target_step, bool)
        or not isinstance(cutoff_step, int)
        or not isinstance(target_step, int)
        or target_step <= cutoff_step
    ):
        raise ValueError("false case frame steps are invalid")
    cutoff = _frame_for_step(source, cutoff_step)
    target = _frame_for_step(source, target_step)
    actor_id = _require_text(case.get("actor_id"), "false case actor_id")
    cutoff_actor = _actor_for_frame(cutoff.pedestrians, actor_id)
    target_actor = _actor_for_frame(target.pedestrians, actor_id)
    expected_horizon = float(target.time_s - cutoff.time_s)
    for field, expected in (
        ("cutoff_time_s", float(cutoff.time_s)),
        ("target_time_s", float(target.time_s)),
        ("horizon_s", expected_horizon),
    ):
        value = _require_finite_number(case.get(field), f"false case {field}")
        if not math.isclose(value, expected, abs_tol=1e-9):
            raise ValueError(f"false case {field} is not trace-backed")
    for field, expected in (
        ("stationary_prediction_m", cutoff_actor["position"]),
        ("target_position_m", target_actor["position"]),
        ("robot_position_m", [float(value) for value in cutoff.robot["position"]]),
    ):
        if not _vectors_close(case.get(field), expected):
            raise ValueError(f"false case {field} is not trace-backed")
    actual_error = float(
        np.linalg.norm(np.asarray(cutoff_actor["position"]) - target_actor["position"])
    )
    actual_clearance = float(
        np.linalg.norm(np.asarray(cutoff.robot["position"]) - target_actor["position"])
    )
    for field, expected in (
        ("ade_m", actual_error),
        ("fde_m", actual_error),
        ("robot_pedestrian_clearance_m", actual_clearance),
    ):
        value = case.get(field)
        if not isinstance(value, (int, float)) or not math.isclose(
            float(value), expected, abs_tol=1e-9
        ):
            raise ValueError(f"false case {field} is not trace-backed")
    predicate_path = _require_text(
        case.get("predicate_reference"), "false case predicate_reference"
    )
    predicate_file = _resolve_repo_path(predicate_path, root)
    expected_predicate_sha = _require_text(
        case.get("predicate_reference_sha256"), "false case predicate_reference_sha256"
    )
    if not _SHA256_RE.fullmatch(expected_predicate_sha):
        raise ValueError("false case predicate_reference_sha256 is not a SHA-256 digest")
    if predicate_path != _FALSE_REASSURANCE_REFERENCE:
        raise ValueError("false case predicate reference is not canonical")
    if sha256_file(predicate_file) != expected_predicate_sha:
        raise ValueError("false case predicate reference SHA-256 does not match file")
    risk_reference = _require_finite_number(
        case.get("risk_reference_m"), "false case risk_reference_m"
    )
    if not math.isclose(risk_reference, _FALSE_REASSURANCE_RISK_REFERENCE_M, abs_tol=1e-9):
        raise ValueError("false case risk reference is not canonical")
    if risk_reference <= actual_clearance:
        raise ValueError("false case must show clearance below its risk reference")


def _validate_sha256_coverage(coverage: Any, root: Path) -> None:  # noqa: C901
    if not isinstance(coverage, dict):
        raise ValueError("sha256_coverage must be a mapping")
    if coverage.get("algorithm") != "SHA-256":
        raise ValueError("sha256_coverage algorithm must be SHA-256")
    manifest_path_value = _require_text(coverage.get("manifest_path"), "sha256 manifest_path")
    manifest_path = _resolve_repo_path(manifest_path_value, root)
    covered_paths = coverage.get("covered_paths")
    if not isinstance(covered_paths, list) or not covered_paths:
        raise ValueError("sha256_coverage.covered_paths must be non-empty")
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            digest, path_value = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"malformed checksum line: {line!r}") from exc
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"invalid checksum digest: {digest}")
        path = _resolve_repo_path(path_value, root)
        relative = path.relative_to(root).as_posix()
        if relative in entries:
            raise ValueError(f"duplicate checksum path: {relative}")
        entries[relative] = digest
        if sha256_file(path) != digest:
            raise ValueError(f"checksum mismatch: {relative}")
    expected_paths = {str(path) for path in covered_paths}
    if set(entries) != expected_paths:
        missing = sorted(expected_paths - set(entries))
        extra = sorted(set(entries) - expected_paths)
        raise ValueError(f"SHA-256 coverage mismatch; missing={missing}, extra={extra}")


def _build_sha256_coverage(
    root: Path,
    checksum_paths: Sequence[Path | str] | None,
) -> dict[str, Any]:
    if checksum_paths is None:
        return {
            "algorithm": "SHA-256",
            "manifest_path": "",
            "covered_paths": [],
            "status": "pending_sidecar_generation",
        }
    relative_paths = sorted(
        {_resolve_repo_path(path, root).relative_to(root).as_posix() for path in checksum_paths}
    )
    return {
        "algorithm": "SHA-256",
        "manifest_path": "docs/context/evidence/issue_7399_forecast_preparation/checksums.sha256",
        "covered_paths": relative_paths,
        "status": "complete_after_sidecar_generation",
    }


def _build_evidence_references(root: Path) -> list[dict[str, str]]:
    paths = {
        "docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/trace_failure_predicate_tables.json"
    }
    for item in _DEPENDENCY_LICENSE_COMPARISON:
        paths.update(str(path) for path in item["evidence_paths"])
    references: list[dict[str, str]] = []
    for relative_path in sorted(paths):
        path = _resolve_repo_path(relative_path, root)
        references.append({"path": relative_path, "sha256": sha256_file(path)})
    return references


def _build_false_reassurance_case(loaded: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    candidates = [item for item in loaded if item["family"] == "crossing_proxy"]
    if not candidates:
        raise ValueError("crossing_proxy source is required for the false-reassurance case")
    item = candidates[0]
    trace: SimulationTraceExport = item["trace"]
    cutoff_step = 2
    target_step = 3
    cutoff = _frame_for_step(trace, cutoff_step)
    target = _frame_for_step(trace, target_step)
    actor = _actor_for_frame(cutoff.pedestrians, None)
    target_actor = _actor_for_frame(target.pedestrians, actor["actor_id"])
    cutoff_position = np.asarray(actor["position"], dtype=float)
    target_position = np.asarray(target_actor["position"], dtype=float)
    robot_position = np.asarray(cutoff.robot["position"], dtype=float)
    error = float(np.linalg.norm(cutoff_position - target_position))
    clearance = float(np.linalg.norm(robot_position - target_position))
    predicate_path = _resolve_repo_path(_FALSE_REASSURANCE_REFERENCE, root)
    return {
        "case_id": _FALSE_REASSURANCE_CASE_ID,
        "status": "analytic_trace_backed_diagnostic_only",
        "source_path": item["relative_path"],
        "source_sha256": item["source_sha256"],
        "predicate_reference": _FALSE_REASSURANCE_REFERENCE,
        "predicate_reference_sha256": sha256_file(predicate_path),
        "cutoff_frame_step": cutoff_step,
        "cutoff_time_s": cutoff.time_s,
        "target_frame_step": target_step,
        "target_time_s": target.time_s,
        "actor_id": actor["actor_id"],
        "horizon_s": float(target.time_s - cutoff.time_s),
        "stationary_prediction_m": [float(value) for value in cutoff_position],
        "target_position_m": [float(value) for value in target_position],
        "ade_m": error,
        "fde_m": error,
        "robot_position_m": [float(value) for value in robot_position],
        "robot_pedestrian_clearance_m": clearance,
        "risk_reference_m": _FALSE_REASSURANCE_RISK_REFERENCE_M,
        "interpretation": (_FALSE_REASSURANCE_INTERPRETATION),
    }


def _validate_coverage(
    loaded: list[dict[str, Any]],
    unavailable: list[dict[str, Any]],
    *,
    required_families: int,
    required_planners: int,
) -> None:
    families = {item["family"] for item in loaded}
    planners = {item["trace"].source.planner_id for item in loaded}
    if len(families) < required_families and not _has_unavailable_dimension(
        unavailable, "scenario_family"
    ):
        raise ValueError(
            "source sample needs three scenario families or an explicit unavailable stratum"
        )
    if len(planners) < required_planners and not _has_unavailable_dimension(unavailable, "planner"):
        raise ValueError("source sample needs two planners or an explicit unavailable stratum")
    if not _has_unavailable_dimension(unavailable, "observation_tier"):
        raise ValueError("ego_observation unavailable stratum must be explicit")


def _assign_group_splits(loaded: list[dict[str, Any]]) -> dict[str, str]:
    groups = sorted({_lineage_group_id(item) for item in loaded})
    return {
        group_id: SPLIT_NAMES[index % len(SPLIT_NAMES)] for index, group_id in enumerate(groups)
    }


def _lineage_group_id(item: dict[str, Any]) -> str:
    trace: SimulationTraceExport = item["trace"]
    return (
        f"{item['family']}:{trace.source.scenario_id}:{trace.source.seed}:{trace.source.episode_id}"
    )


def _lineage_id_from_source(source: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "source_sha256": source["sha256"],
            "trace_id": source["trace_id"],
            "episode_id": source["episode_id"],
        }
    )


def _target_for_horizon(
    trace: SimulationTraceExport,
    *,
    cutoff_step: int,
    horizon_s: float,
    dt_s: float,
    actor_id: str,
) -> dict[str, Any]:
    cutoff = _frame_for_step(trace, cutoff_step)
    desired_time = cutoff.time_s + horizon_s
    candidates = [
        (index, frame) for index, frame in enumerate(trace.frames) if frame.step > cutoff_step
    ]
    if not candidates:
        raise ValueError(f"{trace.trace_id}: no future frames for horizon {horizon_s}")
    target_index, target = min(candidates, key=lambda item: abs(item[1].time_s - desired_time))
    if abs(target.time_s - desired_time) > dt_s * 0.5 + 1e-9:
        raise ValueError(
            f"{trace.trace_id}: no target frame within half-step for horizon {horizon_s}"
        )
    target_actor = _actor_for_frame(target.pedestrians, actor_id)
    return {
        "frame_index": target_index,
        "frame_step": target.step,
        "time_s": target.time_s,
        "position": [float(value) for value in target_actor["position"]],
    }


def _frame_for_step(trace: SimulationTraceExport, step: int):
    matches = [frame for frame in trace.frames if frame.step == step]
    if len(matches) != 1:
        raise ValueError(f"{trace.trace_id}: expected exactly one frame for step {step}")
    return matches[0]


def _actor_for_frame(
    pedestrians: Sequence[Mapping[str, Any]], actor_id: str | None
) -> dict[str, Any]:
    normalized = [dict(actor) for actor in pedestrians]
    if not normalized:
        raise ValueError("frame has no pedestrian actors")
    if actor_id is None:
        selected = sorted(normalized, key=lambda actor: str(actor.get("id", "")))[0]
    else:
        matches = [actor for actor in normalized if str(actor.get("id")) == str(actor_id)]
        if len(matches) != 1:
            raise ValueError(f"actor_id is not unique in the selected frame: {actor_id}")
        selected = matches[0]
    if "id" not in selected or "position" not in selected or "velocity" not in selected:
        raise ValueError("selected pedestrian must contain id, position, and velocity")
    return {
        "actor_id": str(selected["id"]),
        "position": [float(value) for value in selected["position"]],
        "velocity": [float(value) for value in selected["velocity"]],
    }


def _trace_dt_s(trace: SimulationTraceExport) -> float:
    if len(trace.frames) < 2:
        return 0.1
    dt_s = float(trace.frames[1].time_s - trace.frames[0].time_s)
    return dt_s if dt_s > 0.0 else 0.1


def _near_duplicate_fingerprint(trace: SimulationTraceExport) -> str:
    def rounded(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return round(float(value), 2)
        if isinstance(value, list):
            return [rounded(item) for item in value]
        if isinstance(value, dict):
            return {str(key): rounded(item) for key, item in sorted(value.items())}
        return value

    projection = {
        "frames": [
            {
                "step": frame.step,
                "time_s": rounded(frame.time_s),
                "robot": rounded(frame.robot),
                "pedestrians": rounded(frame.pedestrians),
            }
            for frame in trace.frames
        ]
    }
    return stable_hash(projection)


def _ego_source_status(raw_payload: Mapping[str, Any], source_key: str) -> tuple[str, str]:
    frames = raw_payload.get("frames")
    if isinstance(frames, list) and any(
        isinstance(frame, Mapping) and source_key in frame for frame in frames
    ):
        return (
            "available",
            f"source contains {source_key}; a canonical adapter is required before use",
        )
    return (
        "not_available",
        f"simulation_trace_export.v1 source has no {source_key} field; no oracle-to-ego inference is allowed",
    )


def _validate_unavailable_strata(
    unavailable_strata: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    rows = list(unavailable_strata or ())
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"unavailable_strata[{index}] must be a mapping")
        for field in ("dimension", "value", "status", "reason"):
            _require_text(row.get(field), f"unavailable_strata[{index}].{field}")
        if row["status"] != "not_available":
            raise ValueError("unavailable strata must use status=not_available")
        normalized.append({str(key): value for key, value in row.items()})
    return normalized


def _has_unavailable_dimension(rows: Sequence[Mapping[str, Any]], dimension: str) -> bool:
    return any(
        row.get("dimension") == dimension and row.get("status") == "not_available" for row in rows
    )


def _validate_horizons(horizons_s: Any) -> list[float]:
    if isinstance(horizons_s, (str, bytes)) or not isinstance(horizons_s, Sequence):
        raise ValueError("horizons_s must be a sequence")
    horizons = [
        _require_finite_number(value, f"horizons_s[{index}]")
        for index, value in enumerate(horizons_s)
    ]
    if not horizons or any(not math.isfinite(value) or value <= 0.0 for value in horizons):
        raise ValueError("horizons_s must contain finite positive values")
    if len(set(horizons)) != len(horizons) or any(
        later <= earlier for earlier, later in pairwise(horizons)
    ):
        raise ValueError("horizons_s must be strictly increasing")
    return horizons


def _walk_mapping_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_mapping_keys(child))
    return keys


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _require_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _resolve_repo_path(value: Path | str, root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes repository root: {value}")
    if not resolved.is_file():
        raise ValueError(f"path is not a file: {value}")
    return resolved


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


__all__ = [
    "DEFAULT_EGO_SOURCE_KEY",
    "DEFAULT_HORIZONS_S",
    "FORECAST_PREPARATION_ROW_SCHEMA_VERSION",
    "FORECAST_PREPARATION_SCHEMA_VERSION",
    "ForecastPreparationSourceSpec",
    "build_forecast_preparation_packet",
    "validate_forecast_preparation_packet",
]
