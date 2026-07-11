"""Deterministic CPU corridor acceptance harness for the opt-in Zanlungo force.

The harness is deliberately diagnostic. It labels short synthetic head-on traces and
records one-at-a-time parameter variants, but never promotes them to benchmark evidence
or a realism calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import yaml

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.sim.pedestrian_model_variants import (
    HSFM_ZANLUNGO_COLLISION_PREDICTION_V1,
    SOCIAL_FORCE_DEFAULT,
    normalize_pedestrian_model,
)
from robot_sf.sim.sim_config import (
    SimulationSettings,
    ZanlungoCollisionPredictionConfig,
)
from robot_sf.sim.simulator import Simulator

if TYPE_CHECKING:
    from collections.abc import Mapping

    from robot_sf.common.types import Rect

SCHEMA_VERSION = "zanlungo-corridor-acceptance.v1"
EVIDENCE_TIER = "diagnostic-only"
CLAIM_BOUNDARY = (
    "deterministic CPU synthetic-corridor acceptance only; benchmark_evidence=false; "
    "no Robot SF realism calibration, campaign, or paper-facing claim"
)
OutcomeLabel = Literal[
    "yielding",
    "freezing",
    "collision_proxy",
    "pass_through",
    "incomplete",
]


@dataclass(frozen=True)
class CorridorFixtureConfig:
    """Geometry and runtime controls for the two-pedestrian corridor fixture."""

    width_m: float
    height_m: float
    corridor_min_y_m: float
    corridor_max_y_m: float
    left_start: tuple[float, float]
    right_start: tuple[float, float]
    left_goal_x_m: float
    right_goal_x_m: float
    desired_speed_mps: float
    duration_s: float
    dt_s: float
    seed: int

    def __post_init__(self) -> None:
        """Fail closed on malformed or physically empty fixture geometry."""
        values = {
            "width_m": self.width_m,
            "height_m": self.height_m,
            "corridor_min_y_m": self.corridor_min_y_m,
            "corridor_max_y_m": self.corridor_max_y_m,
            "left_start[0]": self.left_start[0],
            "left_start[1]": self.left_start[1],
            "right_start[0]": self.right_start[0],
            "right_start[1]": self.right_start[1],
            "left_goal_x_m": self.left_goal_x_m,
            "right_goal_x_m": self.right_goal_x_m,
            "desired_speed_mps": self.desired_speed_mps,
            "duration_s": self.duration_s,
            "dt_s": self.dt_s,
        }
        for field_name, value in values.items():
            _require_finite_real(value, f"fixture.{field_name}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("fixture.seed must be a non-negative integer")
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("corridor width_m and height_m must be positive")
        if not 0 <= self.corridor_min_y_m < self.corridor_max_y_m <= self.height_m:
            raise ValueError("corridor y bounds must define a non-empty interval inside the map")
        if self.desired_speed_mps <= 0 or self.duration_s <= 0 or self.dt_s <= 0:
            raise ValueError("desired_speed_mps, duration_s, and dt_s must be positive")
        if self.dt_s > self.duration_s:
            raise ValueError("dt_s must not exceed duration_s")


@dataclass(frozen=True)
class OutcomeThresholds:
    """Predeclared diagnostic thresholds used to label a trace."""

    lateral_yield_m: float
    collision_proxy_distance_m: float
    freeze_speed_mps: float
    freeze_window_s: float
    conflict_distance_m: float

    def __post_init__(self) -> None:
        """Require finite non-negative thresholds and a positive conflict distance."""
        values = {
            "lateral_yield_m": self.lateral_yield_m,
            "collision_proxy_distance_m": self.collision_proxy_distance_m,
            "freeze_speed_mps": self.freeze_speed_mps,
            "freeze_window_s": self.freeze_window_s,
            "conflict_distance_m": self.conflict_distance_m,
        }
        for field_name, value in values.items():
            _require_finite_real(value, f"thresholds.{field_name}")
        if (
            any(
                value < 0
                for field_name, value in values.items()
                if field_name != "conflict_distance_m"
            )
            or self.conflict_distance_m <= 0
        ):
            raise ValueError(
                "outcome thresholds must be non-negative with positive conflict distance"
            )


@dataclass(frozen=True)
class SensitivityCase:
    """One predeclared pedestrian-model parameter row."""

    case_id: str
    pedestrian_model: str
    varied_parameter: str | None
    relative_to_reference: float | None
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate the model selector and sensitivity bookkeeping fields."""
        if not self.case_id.strip():
            raise ValueError("sensitivity case_id must be non-empty")
        normalize_pedestrian_model(self.pedestrian_model)
        if self.relative_to_reference is not None:
            _require_finite_real(self.relative_to_reference, "relative_to_reference")
            if self.relative_to_reference < 0:
                raise ValueError("relative_to_reference must be non-negative")


@dataclass(frozen=True)
class CorridorTrace:
    """Positions and velocities from one deterministic fixture execution."""

    positions: np.ndarray
    velocities: np.ndarray
    dt_s: float


@dataclass(frozen=True)
class AcceptanceConfig:
    """Validated issue #4973 corridor acceptance packet."""

    issue: int
    benchmark_evidence: bool
    evidence_tier: str
    claim_boundary: str
    reference_case_id: str
    fixture: CorridorFixtureConfig
    thresholds: OutcomeThresholds
    cases: tuple[SensitivityCase, ...]
    config_path: str
    config_sha256: str


def load_acceptance_config(path: str | Path) -> AcceptanceConfig:
    """Load and validate the predeclared YAML acceptance packet.

    Returns:
        Validated corridor acceptance configuration.
    """
    config_path = Path(path)
    raw_bytes = config_path.read_bytes()
    payload = yaml.safe_load(raw_bytes)
    if not isinstance(payload, dict):
        raise ValueError("acceptance config must be a YAML mapping")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    evidence_tier, claim_boundary = _validate_metadata(_require_mapping(payload, "metadata"))
    fixture = CorridorFixtureConfig(**_normalize_fixture(_require_mapping(payload, "fixture")))
    thresholds = OutcomeThresholds(**_require_mapping(payload, "thresholds"))
    raw_cases = payload.get("sensitivity_cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("sensitivity_cases must be a non-empty list")
    cases = tuple(_normalize_case(item) for item in raw_cases)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("sensitivity case_id values must be unique")
    reference_case_id = str(payload.get("reference_case_id", ""))
    if reference_case_id not in set(case_ids):
        raise ValueError("reference_case_id must select one sensitivity case")
    reference_case = cases[case_ids.index(reference_case_id)]
    if reference_case.pedestrian_model != HSFM_ZANLUNGO_COLLISION_PREDICTION_V1:
        raise ValueError("reference_case_id must select the opt-in Zanlungo model")
    if int(payload.get("issue", 0)) != 4973:
        raise ValueError("issue must be 4973")
    if not _parameter_bookkeeping_complete(cases, reference_case_id):
        raise ValueError(
            "sensitivity rows must change only their named parameter and record its exact ratio"
        )
    return AcceptanceConfig(
        issue=4973,
        benchmark_evidence=False,
        evidence_tier=evidence_tier,
        claim_boundary=claim_boundary,
        reference_case_id=reference_case_id,
        fixture=fixture,
        thresholds=thresholds,
        cases=cases,
        config_path=str(config_path),
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def classify_corridor_trace(
    trace: CorridorTrace,
    fixture: CorridorFixtureConfig,
    thresholds: OutcomeThresholds,
) -> dict[str, Any]:
    """Compute trace metrics and assign one explicit outcome label.

    Returns:
        JSON-safe metrics and a mutually exclusive outcome label.
    """
    positions = np.asarray(trace.positions, dtype=float)
    velocities = np.asarray(trace.velocities, dtype=float)
    if positions.ndim != 3 or positions.shape[1:] != (2, 2):
        raise ValueError("trace.positions must have shape (T, 2, 2)")
    if velocities.shape != positions.shape:
        raise ValueError("trace.velocities must match trace.positions")
    if positions.shape[0] < 1:
        raise ValueError("trace must contain at least one state")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError("trace positions and velocities must be finite")
    if not math.isfinite(trace.dt_s) or trace.dt_s <= 0:
        raise ValueError("trace.dt_s must be finite and positive")

    separation = np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1)
    speeds = np.linalg.norm(velocities, axis=-1)
    passed_by_step = positions[:, 0, 0] >= positions[:, 1, 0]
    passed_indices = np.flatnonzero(passed_by_step)
    first_pass_index = int(passed_indices[0]) if passed_indices.size else positions.shape[0] - 1
    pre_pass_positions = positions[: first_pass_index + 1]
    lateral_displacements = np.max(
        np.abs(pre_pass_positions[:, :, 1] - positions[0, :, 1][np.newaxis, :]), axis=0
    )
    freeze_mask = (
        (separation <= thresholds.conflict_distance_m)
        & ~passed_by_step
        & np.all(speeds <= thresholds.freeze_speed_mps, axis=1)
    )
    max_freeze_steps = _max_consecutive_true(freeze_mask)
    required_freeze_steps = max(1, math.ceil(thresholds.freeze_window_s / trace.dt_s))
    passed = bool(np.any(passed_by_step))
    avoided_collision_proxy = float(np.min(separation)) >= thresholds.collision_proxy_distance_m
    lateral_yield_met = bool(np.all(lateral_displacements >= thresholds.lateral_yield_m))
    freezing = max_freeze_steps >= required_freeze_steps

    if freezing:
        outcome: OutcomeLabel = "freezing"
    elif passed and lateral_yield_met and avoided_collision_proxy:
        outcome = "yielding"
    elif passed and not avoided_collision_proxy:
        outcome = "collision_proxy"
    elif passed:
        outcome = "pass_through"
    else:
        outcome = "incomplete"

    return {
        "outcome_label": outcome,
        "passed": passed,
        "first_passing_time_s": float(first_pass_index * trace.dt_s) if passed else None,
        "minimum_pairwise_center_distance_m": float(np.min(separation)),
        "maximum_lateral_displacement_m": [float(value) for value in lateral_displacements],
        "lateral_yield_criterion_met": lateral_yield_met,
        "collision_proxy_avoided": avoided_collision_proxy,
        "maximum_consecutive_freeze_steps": max_freeze_steps,
        "required_freeze_steps": required_freeze_steps,
        "freeze_criterion_met": freezing,
        "finite_positions": True,
        "finite_velocities": True,
    }


def run_corridor_trace(
    fixture: CorridorFixtureConfig,
    case: SensitivityCase,
) -> CorridorTrace:
    """Run one CPU-only corridor trace for a predeclared sensitivity row.

    Returns:
        Collected simulator state for the requested row.
    """
    model = normalize_pedestrian_model(case.pedestrian_model)
    zanlungo = ZanlungoCollisionPredictionConfig()
    if model == HSFM_ZANLUNGO_COLLISION_PREDICTION_V1:
        zanlungo = ZanlungoCollisionPredictionConfig(enabled=True, **dict(case.parameters))
    elif case.parameters:
        raise ValueError("non-Zanlungo sensitivity rows must not declare Zanlungo parameters")
    settings = SimulationSettings(
        sim_time_in_secs=fixture.duration_s,
        time_per_step_in_secs=fixture.dt_s,
        ped_density_by_difficulty=[0.0],
        difficulty=0,
        route_spawn_seed=fixture.seed,
        max_total_pedestrians=2,
        pedestrian_model=model,
        zanlungo_collision_prediction=zanlungo,
    )
    simulator = Simulator(
        settings,
        _build_corridor_map(fixture),
        robots=[],
        goal_proximity_threshold=0.0,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )
    positions = [np.asarray(simulator.ped_pos, dtype=float).copy()]
    velocities = [np.asarray(simulator.ped_vel, dtype=float).copy()]
    for _ in range(math.floor(fixture.duration_s / fixture.dt_s)):
        simulator.step_once([])
        positions.append(np.asarray(simulator.ped_pos, dtype=float).copy())
        velocities.append(np.asarray(simulator.ped_vel, dtype=float).copy())
    return CorridorTrace(
        positions=np.asarray(positions, dtype=float),
        velocities=np.asarray(velocities, dtype=float),
        dt_s=fixture.dt_s,
    )


def run_acceptance(config: AcceptanceConfig) -> dict[str, Any]:
    """Execute every row twice and build the issue #4973 acceptance report.

    Returns:
        JSON-safe report with outcomes, sensitivity lineage, and replay proof.
    """
    rows: list[dict[str, Any]] = []
    for case in config.cases:
        first = run_corridor_trace(config.fixture, case)
        replay = run_corridor_trace(config.fixture, case)
        deterministic = bool(
            np.array_equal(first.positions, replay.positions)
            and np.array_equal(first.velocities, replay.velocities)
        )
        row = {
            "case_id": case.case_id,
            "pedestrian_model": case.pedestrian_model,
            "varied_parameter": case.varied_parameter,
            "relative_to_reference": case.relative_to_reference,
            "parameters": dict(case.parameters),
            "seed": config.fixture.seed,
            "step_count": int(first.positions.shape[0] - 1),
            "trace_sha256": _trace_sha256(first),
            "replay_trace_sha256": _trace_sha256(replay),
            "replay_deterministic": deterministic,
            "metrics": classify_corridor_trace(first, config.fixture, config.thresholds),
        }
        rows.append(row)

    reference = next(row for row in rows if row["case_id"] == config.reference_case_id)
    checks = {
        "reference_outcome_is_yielding": reference["metrics"]["outcome_label"] == "yielding",
        "all_rows_replay_deterministic": all(row["replay_deterministic"] for row in rows),
        "parameter_sensitivity_bookkeeping_complete": all(
            row["varied_parameter"] is not None and bool(row["parameters"])
            for row in rows
            if row["pedestrian_model"] == HSFM_ZANLUNGO_COLLISION_PREDICTION_V1
        ),
        "benchmark_evidence_is_false": config.benchmark_evidence is False,
        "default_force_path_unchanged": SimulationSettings().pedestrian_model
        == SOCIAL_FORCE_DEFAULT,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": config.issue,
        "claim_boundary": config.claim_boundary,
        "evidence_tier": config.evidence_tier,
        "benchmark_evidence": False,
        "config": {
            "path": config.config_path,
            "sha256": config.config_sha256,
            "reference_case_id": config.reference_case_id,
        },
        "fixture": {
            "seed": config.fixture.seed,
            "duration_s": config.fixture.duration_s,
            "dt_s": config.fixture.dt_s,
            "pedestrian_count": 2,
            "robot_inserted": False,
        },
        "thresholds": config.thresholds.__dict__,
        "acceptance_checks": checks,
        "acceptance_met": all(checks.values()),
        "status": {
            "cpu_only": True,
            "slurm_or_gpu_used": False,
            "full_campaign_run": False,
        },
        "rows": rows,
    }


def write_acceptance_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write compact JSON and Markdown acceptance artifacts.

    Returns:
        Paths to the generated local artifacts.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.json"
    readme = out_dir / "README.md"
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme.write_text(render_acceptance_markdown(report), encoding="utf-8")
    return {"summary_json": summary, "readme": readme}


def render_acceptance_markdown(report: Mapping[str, Any]) -> str:
    """Render a reviewer-readable report summary.

    Returns:
        Markdown with the claim boundary before outcome interpretation.
    """
    lines = [
        "# Zanlungo corridor acceptance diagnostic",
        "",
        f"Claim boundary: {report['claim_boundary']}.",
        "Evidence status: diagnostic-only; `benchmark_evidence=false`.",
        "",
        "| case | model | varied parameter | outcome | minimum center distance (m) | "
        "replay deterministic |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['case_id']} | {row['pedestrian_model']} | "
            f"{row['varied_parameter'] or 'baseline'} | {row['metrics']['outcome_label']} | "
            f"{row['metrics']['minimum_pairwise_center_distance_m']:.3f} | "
            f"{row['replay_deterministic']} |"
        )
    lines.extend(
        [
            "",
            f"Acceptance met: **{report['acceptance_met']}**.",
            "",
            "These synthetic labels are fixture acceptance checks, not benchmark or realism claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _normalize_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for key in ("left_start", "right_start"):
        value = normalized.get(key)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"fixture.{key} must be a two-value list")
        normalized[key] = (
            _require_finite_real(value[0], f"fixture.{key}[0]"),
            _require_finite_real(value[1], f"fixture.{key}[1]"),
        )
    return normalized


def _normalize_case(payload: Any) -> SensitivityCase:
    if not isinstance(payload, dict):
        raise ValueError("each sensitivity case must be a mapping")
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("sensitivity case parameters must be a mapping")
    pedestrian_model = normalize_pedestrian_model(payload.get("pedestrian_model"))
    if pedestrian_model == HSFM_ZANLUNGO_COLLISION_PREDICTION_V1:
        _validate_zanlungo_parameters(parameters)
    return SensitivityCase(
        case_id=str(payload.get("case_id", "")),
        pedestrian_model=pedestrian_model,
        varied_parameter=payload.get("varied_parameter"),
        relative_to_reference=payload.get("relative_to_reference"),
        parameters=parameters,
    )


def _require_finite_real(value: Any, field_name: str) -> float:
    """Return a finite real value while rejecting YAML booleans explicitly."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite real number, not a boolean")
    return float(value)


def _validate_metadata(metadata: Mapping[str, Any]) -> tuple[str, str]:
    """Require the packet provenance that the diagnostic report will emit.

    Returns:
        Validated evidence tier and claim boundary.
    """
    if metadata.get("benchmark_evidence") is not False:
        raise ValueError("metadata.benchmark_evidence must be false")
    evidence_tier = metadata.get("evidence_tier")
    if evidence_tier != EVIDENCE_TIER:
        raise ValueError(f"metadata.evidence_tier must be {EVIDENCE_TIER}")
    claim_boundary = metadata.get("claim_boundary")
    if claim_boundary != CLAIM_BOUNDARY:
        raise ValueError("metadata.claim_boundary must match the diagnostic acceptance boundary")
    return evidence_tier, claim_boundary


def _validate_zanlungo_parameters(parameters: Mapping[str, Any]) -> None:
    """Reject incomplete, unexpected, or non-numeric Zanlungo packet parameters."""
    numeric_fields = {
        "interaction_strength",
        "interaction_range_m",
        "anisotropy_lambda",
        "angle_threshold_rad",
        "max_force",
    }
    required_fields = numeric_fields | {"include_ped_ped"}
    if set(parameters) != required_fields:
        raise ValueError("Zanlungo sensitivity parameters must declare the complete paper packet")
    for field_name in numeric_fields:
        _require_finite_real(parameters[field_name], f"parameters.{field_name}")
    if not isinstance(parameters["include_ped_ped"], bool):
        raise ValueError("parameters.include_ped_ped must be a boolean")
    ZanlungoCollisionPredictionConfig(enabled=True, **dict(parameters))


def _parameter_bookkeeping_complete(
    cases: tuple[SensitivityCase, ...], reference_case_id: str
) -> bool:
    reference = next(case for case in cases if case.case_id == reference_case_id)
    reference_parameters = dict(reference.parameters)
    for case in cases:
        if case.pedestrian_model != HSFM_ZANLUNGO_COLLISION_PREDICTION_V1:
            if case.parameters or case.varied_parameter is not None:
                return False
            continue
        if set(case.parameters) != set(reference_parameters):
            return False
        if case.case_id == reference_case_id:
            if case.varied_parameter != "reference" or case.relative_to_reference != 1.0:
                return False
            continue
        changed = {
            key
            for key, reference_value in reference_parameters.items()
            if case.parameters[key] != reference_value
        }
        if changed != {case.varied_parameter}:
            return False
        reference_value = reference_parameters.get(str(case.varied_parameter))
        case_value = case.parameters.get(str(case.varied_parameter))
        if (
            isinstance(reference_value, bool)
            or isinstance(case_value, bool)
            or not isinstance(reference_value, (int, float))
            or not isinstance(case_value, (int, float))
            or reference_value == 0
            or case.relative_to_reference is None
            or not math.isclose(
                float(case.relative_to_reference),
                float(case_value) / float(reference_value),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            return False
    return True


def _require_mapping(payload: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _build_corridor_map(fixture: CorridorFixtureConfig) -> MapDefinition:
    obstacles = [
        Obstacle(
            [
                (0.0, 0.0),
                (fixture.width_m, 0.0),
                (fixture.width_m, fixture.corridor_min_y_m),
                (0.0, fixture.corridor_min_y_m),
            ]
        ),
        Obstacle(
            [
                (0.0, fixture.corridor_max_y_m),
                (fixture.width_m, fixture.corridor_max_y_m),
                (fixture.width_m, fixture.height_m),
                (0.0, fixture.height_m),
            ]
        ),
    ]
    robot_spawn = _rect(0.2, 0.2, 0.8, 0.8)
    robot_goal = _rect(
        fixture.width_m - 0.8,
        fixture.height_m - 0.8,
        fixture.width_m - 0.2,
        fixture.height_m - 0.2,
    )
    pedestrians = [
        SinglePedestrianDefinition(
            id="corridor_left_to_right",
            start=fixture.left_start,
            trajectory=[(fixture.left_goal_x_m, fixture.left_start[1])],
            speed_m_s=fixture.desired_speed_mps,
            metadata={"benchmark_evidence": False, "issue": 4973},
        ),
        SinglePedestrianDefinition(
            id="corridor_right_to_left",
            start=fixture.right_start,
            trajectory=[(fixture.right_goal_x_m, fixture.right_start[1])],
            speed_m_s=fixture.desired_speed_mps,
            metadata={"benchmark_evidence": False, "issue": 4973},
        ),
    ]
    return MapDefinition(
        width=fixture.width_m,
        height=fixture.height_m,
        obstacles=obstacles,
        robot_spawn_zones=[robot_spawn],
        ped_spawn_zones=[],
        robot_goal_zones=[robot_goal],
        bounds=[
            ((0.0, 0.0), (fixture.width_m, 0.0)),
            ((fixture.width_m, 0.0), (fixture.width_m, fixture.height_m)),
            ((fixture.width_m, fixture.height_m), (0.0, fixture.height_m)),
            ((0.0, fixture.height_m), (0.0, 0.0)),
        ],
        robot_routes=[
            GlobalRoute(
                spawn_id=0,
                goal_id=0,
                waypoints=[(0.5, 0.5), (fixture.width_m - 0.5, fixture.height_m - 0.5)],
                spawn_zone=robot_spawn,
                goal_zone=robot_goal,
                source_path_id="issue_4973_diagnostic_robot_stub",
                source_label="diagnostic-only robot route stub; no robot inserted",
            )
        ],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=pedestrians,
    )


def _trace_sha256(trace: CorridorTrace) -> str:
    digest = hashlib.sha256()
    for array in (trace.positions, trace.velocities):
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _max_consecutive_true(mask: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _rect(x0: float, y0: float, x1: float, y1: float) -> Rect:
    return ((x0, y1), (x1, y1), (x1, y0))
