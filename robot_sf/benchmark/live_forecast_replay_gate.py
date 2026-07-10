"""Live same-seed forecast replay gate v1 (issues #2902, #2944, #2941).

This module implements the smallest valuable executable surface for comparing
forecast variants on a motion-rich pedestrian fixture with identical seed.
Issue #2944 narrows the default smoke to the ``none`` and ``cv`` variants and
adds an explicit per-run classification (``native``, ``blocked``,
``degraded``, ``diagnostic_only``) so the result can gate expansion to the full
forecast variant matrix.

Issue #2941 adds a minimal ``ProbabilisticPredictor`` baseline implementation
and a forecast-brake replay policy so non-none variants can influence
closed-loop metrics.  Issue #3164 freezes the replay policy controls across
forecast variants so closed-loop differences are not confounded by
variant-specific brake thresholds.  This is still a smoke proof of forecast
coupling, not a claim that the variants improve a production planner.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.analysis_workbench.simulation_trace_export import (
    SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
    SimulationTraceExport,
    SimulationTraceFrame,
    SimulationTraceSource,
)
from robot_sf.benchmark.forecast_batch import (
    ActorForecast,
    CoordinateFrame,
    ForecastBatch,
    ForecastBatchProvenance,
    validate_forecast_batch,
)
from robot_sf.benchmark.forecast_metrics import evaluate_forecast_batch
from robot_sf.benchmark.forecast_observation_adapters import (
    ForecastObservationAdapter,
    OracleFullStateForecastAdapter,
)
from robot_sf.benchmark.pedestrian_forecast import (
    ForecastBaselineFunction,
    NeighborContext,
    PedestrianState,
    constant_velocity_gaussian_baseline,
    interaction_aware_cv_baseline,
    risk_filtered_cv_baseline,
    semantic_cv_baseline,
)
from robot_sf.common.forecast_variants import FORECAST_VARIANT_CHOICES
from robot_sf.common.issue_provenance import LIVE_FORECAST_REPLAY_GATE_CONTRACT_ISSUE
from robot_sf.errors import RobotSfError
from robot_sf.gym_env.unified_config import EnvSettings
from robot_sf.nav.baseline_probabilistic_predictor import BaselineProbabilisticPredictor
from robot_sf.nav.predictive_types import ProbabilisticPrediction, ProbabilisticPredictor

LIVE_FORECAST_REPLAY_GATE_SCHEMA_VERSION = "LiveForecastReplayGate.v1"
LIVE_FORECAST_REPLAY_GATE_ISSUE = LIVE_FORECAST_REPLAY_GATE_CONTRACT_ISSUE

FORECAST_VARIANTS: tuple[str, ...] = FORECAST_VARIANT_CHOICES

SMOKE_FORECAST_VARIANTS: tuple[str, ...] = (
    "none",
    "cv",
)

REQUIRED_METRICS: tuple[str, ...] = (
    "collision",
    "near_miss",
    "min_distance",
    "stop_yield_timing",
    "progress",
    "false_positive_stops",
    "runtime",
)

# Issue #2941 run classification contract.  The classification is fail-closed:
# a run is only ``native`` when the cv forecast flows into the replay policy and
# produces closed-loop metrics that differ from the integrated no-forecast baseline.
RUN_CLASSIFICATION_NATIVE = "native"
RUN_CLASSIFICATION_BLOCKED = "blocked"
RUN_CLASSIFICATION_DEGRADED = "degraded"
RUN_CLASSIFICATION_DIAGNOSTIC_ONLY = "diagnostic_only"
VALID_RUN_CLASSIFICATIONS: tuple[str, ...] = (
    RUN_CLASSIFICATION_NATIVE,
    RUN_CLASSIFICATION_BLOCKED,
    RUN_CLASSIFICATION_DEGRADED,
    RUN_CLASSIFICATION_DIAGNOSTIC_ONLY,
)

DEFAULT_HORIZONS_S: tuple[float, float, float] = (0.5, 1.0, 2.0)
DEFAULT_COLLISION_DISTANCE_M: float = 0.5
DEFAULT_NEAR_MISS_DISTANCE_M: float = 1.5
DEFAULT_STOP_SPEED_MPS: float = 0.05
DEFAULT_PROGRESS_GOAL_PROXIMITY_M: float = 1.0
DEFAULT_RISK_DISTANCE_M: float = 3.0
DEFAULT_REPLAY_BRAKE_DISTANCE_M: float = 3.0
DEFAULT_VARIANT_FORECAST_RISK_DISTANCE_M: dict[str, float] = {
    "cv": 3.0,
    "semantic": 2.5,
    "interaction_aware": 2.0,
    "risk_filtered": 1.5,
}


@dataclass(frozen=True)
class LiveForecastReplayGateConfig:
    """Configuration for one gate run."""

    horizons_s: tuple[float, ...] = DEFAULT_HORIZONS_S
    collision_distance_m: float = DEFAULT_COLLISION_DISTANCE_M
    near_miss_distance_m: float = DEFAULT_NEAR_MISS_DISTANCE_M
    stop_speed_mps: float = DEFAULT_STOP_SPEED_MPS
    progress_goal_proximity_m: float = DEFAULT_PROGRESS_GOAL_PROXIMITY_M
    risk_distance_m: float = DEFAULT_RISK_DISTANCE_M
    replay_brake_distance_m: float = DEFAULT_REPLAY_BRAKE_DISTANCE_M
    variant_risk_distance_m: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_VARIANT_FORECAST_RISK_DISTANCE_M)
    )
    observation_adapter: ForecastObservationAdapter = field(
        default_factory=OracleFullStateForecastAdapter
    )


class LiveForecastReplayGateError(RobotSfError, ValueError):
    """Raised when the gate cannot produce a valid report."""


def load_trace_tolerant(path: Path | str) -> SimulationTraceExport:
    """Load a simulation trace export, tolerating common schema extensions.

    Some durable fixtures carry extra metadata (``dense_stress_metadata``,
    ``occlusion_status``, integer pedestrian ids) that the strict schema rejects.
    The gate needs to process these motion-rich fixtures, so this loader strips
    unknown frame properties and coerces ids to strings before building the typed
    export directly.

    Returns:
        A typed ``SimulationTraceExport`` even when the source JSON has extensions.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LiveForecastReplayGateError("trace JSON must be an object")

    source_raw = raw.get("source")
    source = source_raw if isinstance(source_raw, dict) else {}
    frames = raw.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise LiveForecastReplayGateError("trace must contain at least one frame")

    allowed_frame_keys = {"step", "time_s", "robot", "pedestrians", "planner"}

    def _value_or_default(value: Any, default: Any) -> Any:
        return default if value is None else value

    def _normalize_frame(frame: dict[str, Any]) -> SimulationTraceFrame:
        stripped = {key: frame[key] for key in allowed_frame_keys if key in frame}
        pedestrians_raw = stripped.get("pedestrians")
        pedestrians_list = pedestrians_raw if pedestrians_raw is not None else []
        pedestrians: list[dict[str, Any]] = []
        for index, pedestrian_raw in enumerate(pedestrians_list):
            pedestrian = pedestrian_raw if isinstance(pedestrian_raw, dict) else {}
            pedestrian_id = pedestrian.get("id")
            normalized_pedestrian = {
                "id": str(pedestrian_id) if pedestrian_id is not None else str(index),
                "position": list(_value_or_default(pedestrian.get("position"), [0.0, 0.0])),
                "velocity": list(_value_or_default(pedestrian.get("velocity"), [0.0, 0.0])),
            }
            for key in ("intent_label", "signal_state", "signal_label", "actor_type"):
                if key in pedestrian:
                    normalized_pedestrian[key] = pedestrian[key]
            pedestrians.append(normalized_pedestrian)

        robot_raw = stripped.get("robot")
        robot = robot_raw if isinstance(robot_raw, dict) else {}
        normalized_robot = {
            "position": list(_value_or_default(robot.get("position"), [0.0, 0.0])),
            "heading": float(_value_or_default(robot.get("heading"), 0.0)),
            "velocity": list(_value_or_default(robot.get("velocity"), [0.0, 0.0])),
        }
        if "goal" in robot:
            normalized_robot["goal"] = robot["goal"]

        planner_raw = stripped.get("planner")
        planner = (
            planner_raw
            if isinstance(planner_raw, dict)
            else {"selected_action": {"linear_velocity": 0.0, "angular_velocity": 0.0}}
        )
        return SimulationTraceFrame(
            step=int(_value_or_default(stripped.get("step"), 0)),
            time_s=float(_value_or_default(stripped.get("time_s"), 0.0)),
            robot=normalized_robot,
            pedestrians=pedestrians,
            planner=dict(planner),
        )

    schema_version = str(
        _value_or_default(raw.get("schema_version"), SIMULATION_TRACE_EXPORT_SCHEMA_VERSION)
    )
    if schema_version != SIMULATION_TRACE_EXPORT_SCHEMA_VERSION:
        raise LiveForecastReplayGateError(f"unsupported trace schema_version: {schema_version}")

    return SimulationTraceExport(
        schema_version=schema_version,
        trace_id=str(_value_or_default(raw.get("trace_id"), Path(path).stem)),
        source=SimulationTraceSource(
            scenario_id=str(_value_or_default(source.get("scenario_id"), "unknown")),
            seed=int(_value_or_default(source.get("seed"), 0)),
            planner_id=str(_value_or_default(source.get("planner_id"), "unknown")),
            episode_id=str(_value_or_default(source.get("episode_id"), "unknown")),
            generated_by=str(_value_or_default(source.get("generated_by"), "tolerant_loader")),
        ),
        evidence_boundary=str(
            _value_or_default(raw.get("evidence_boundary"), "analysis_workbench_only")
        ),
        coordinate_frame=str(_value_or_default(raw.get("coordinate_frame"), "world")),
        units=dict(
            _value_or_default(
                raw.get("units"),
                {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
            )
        ),
        frames=[_normalize_frame(frame) for frame in frames],
    )


def _robot_position_at_step(trace: SimulationTraceExport, step_index: int) -> np.ndarray | None:
    """Return the robot position at a trace step, or None when unavailable."""

    if step_index < 0 or step_index >= len(trace.frames):
        return None
    robot = trace.frames[step_index].robot
    position = robot.get("position")
    if position is None:
        return None
    return np.asarray(position, dtype=float)


def _all_pedestrian_actor_ids(trace: SimulationTraceExport) -> list[str]:
    """Return the union of pedestrian actor ids observed across the trace."""

    ids: set[str] = set()
    for frame in trace.frames:
        for pedestrian in frame.pedestrians:
            actor_id = pedestrian.get("id")
            if actor_id is not None:
                ids.add(str(actor_id))
    return sorted(ids)


def _future_ground_truth_positions(
    trace: SimulationTraceExport,
    actor_id: str,
    start_step: int,
    horizons_s: tuple[float, ...],
) -> dict[float, np.ndarray]:
    """Build ground-truth future positions for one actor from the trace.

    Returns:
        Mapping from horizon in seconds to future (x, y) position.
    """

    dt_s = _trace_dt_s(trace)
    positions: dict[float, np.ndarray] = {}
    for horizon_s in horizons_s:
        target_step = start_step + round(float(horizon_s) / dt_s)
        if target_step >= len(trace.frames):
            continue
        for pedestrian in trace.frames[target_step].pedestrians:
            if str(pedestrian.get("id")) == actor_id:
                positions[float(horizon_s)] = np.asarray(pedestrian["position"], dtype=float)
                break
    return positions


def _trace_dt_s(trace: SimulationTraceExport) -> float:
    """Return the nominal timestep of the trace."""

    if len(trace.frames) < 2:
        return 0.1
    dt_s = trace.frames[1].time_s - trace.frames[0].time_s
    return float(dt_s) if dt_s > 0.0 else 0.1


def _neighbors_at_step(
    trace: SimulationTraceExport, step_index: int, ego_actor_id: str
) -> list[NeighborContext]:
    """Return neighbor context for interaction-aware forecasts."""

    if step_index < 0 or step_index >= len(trace.frames):
        return []
    neighbors: list[NeighborContext] = []
    for pedestrian in trace.frames[step_index].pedestrians:
        if str(pedestrian.get("id")) == ego_actor_id:
            continue
        neighbors.append(
            NeighborContext(
                position=np.asarray(pedestrian["position"], dtype=float),
                velocity=np.asarray(pedestrian["velocity"], dtype=float),
                actor_type=str(pedestrian.get("actor_type") or "pedestrian"),
            )
        )
    return neighbors


def _pedestrian_state_at_step(
    trace: SimulationTraceExport, step_index: int, actor_id: str
) -> PedestrianState | None:
    """Build a typed pedestrian state from a trace frame.

    Returns:
        PedestrianState for the requested actor, or ``None`` when not present.
    """

    if step_index < 0 or step_index >= len(trace.frames):
        return None
    for pedestrian in trace.frames[step_index].pedestrians:
        if str(pedestrian.get("id")) == actor_id:
            return PedestrianState.from_trace(pedestrian)
    return None


def _baseline_for_variant(
    variant: str,
    *,
    robot_position: np.ndarray | None = None,
    risk_distance_m: float = DEFAULT_RISK_DISTANCE_M,
) -> ForecastBaselineFunction:
    """Resolve a forecast baseline function for a variant.

    Returns:
        Baseline function configured for the requested variant.
    """

    if variant == "cv":
        return constant_velocity_gaussian_baseline
    if variant == "semantic":
        return semantic_cv_baseline
    if variant == "interaction_aware":
        return interaction_aware_cv_baseline
    if variant == "risk_filtered":

        def _risk_filtered_wrapper(
            state: PedestrianState,
            horizons_s: list[float] | tuple[float, ...],
        ) -> Any:
            return risk_filtered_cv_baseline(
                state,
                horizons_s,
                robot_position=robot_position,
                risk_distance_m=risk_distance_m,
            )

        return _risk_filtered_wrapper
    raise LiveForecastReplayGateError(f"unsupported forecast variant: {variant}")


def build_variant_forecast_batch(
    trace: SimulationTraceExport,
    variant: str,
    *,
    step_index: int = 0,
    horizons_s: tuple[float, ...] = DEFAULT_HORIZONS_S,
    risk_distance_m: float = DEFAULT_RISK_DISTANCE_M,
    feature_schema: dict[str, Any] | None = None,
    observation_adapter: ForecastObservationAdapter | None = None,
) -> ForecastBatch:
    """Build a ``ForecastBatch.v1`` artifact for one forecast variant.

    The artifact uses the trace's scenario id and seed so that variants are
    comparable under the same-seed replay contract.

    Args:
        trace: Source simulation trace export.
        variant: Forecast variant from ``FORECAST_VARIANTS``.
        step_index: Frame at which to generate the forecast.
        horizons_s: Forecast horizons in seconds.
        risk_distance_m: Distance threshold for the ``risk_filtered`` variant.
        feature_schema: Optional feature schema for provenance.
        observation_adapter: Adapter controlling observation tier.

    Returns:
        Validated ``ForecastBatch.v1`` artifact for the variant.
    """

    if variant == "none":
        raise LiveForecastReplayGateError(
            "none is the integrated no-forecast baseline and has no forecast batch"
        )
    if variant not in FORECAST_VARIANTS:
        raise LiveForecastReplayGateError(f"unsupported forecast variant: {variant}")

    adapter = observation_adapter or OracleFullStateForecastAdapter()
    schema = feature_schema or {
        "name": "live_forecast_replay_gate_v1",
        "features": ["position_m", "velocity_mps"],
    }

    trace_dict = {
        "scenario_id": trace.source.scenario_id,
        "seed": trace.source.seed,
        "frames": [
            {
                "step": frame.step,
                "time_s": frame.time_s,
                "pedestrians": [dict(pedestrian) for pedestrian in frame.pedestrians],
                "robot": dict(frame.robot),
                "planner": dict(frame.planner),
            }
            for frame in trace.frames
        ],
    }

    observed = adapter.adapt_trace(
        trace_dict,
        feature_schema=schema,
        horizons_s=list(horizons_s),
        dt_s=_trace_dt_s(trace),
        step_index=step_index,
        expected_actor_ids=_all_pedestrian_actor_ids(trace),
    )

    robot_position = _robot_position_at_step(trace, step_index)
    baseline_fn = _baseline_for_variant(
        variant,
        robot_position=robot_position,
        risk_distance_m=risk_distance_m,
    )

    forecasts: list[ActorForecast] = []
    for actor in observed.actors:
        neighbors = _neighbors_at_step(trace, step_index, actor.actor_id)
        if "neighbors" in inspect.signature(baseline_fn).parameters:
            forecast = baseline_fn(actor.state, observed.provenance.horizons_s, neighbors=neighbors)  # type: ignore[call-arg]
        else:
            forecast = baseline_fn(actor.state, observed.provenance.horizons_s)
        forecasts.append(
            ActorForecast(
                actor_id=actor.actor_id,
                deterministic=np.asarray(
                    [prediction.mean for prediction in forecast.predictions],
                    dtype=float,
                ),
                occupancy_summary={
                    "model": forecast.predictions[0].metadata.get("model")
                    if forecast.predictions
                    else None,
                    "relevance_status": (
                        forecast.predictions[0].metadata.get("relevance_status")
                        if forecast.predictions
                        else None
                    ),
                    "relevance_status_by_horizon_s": {
                        f"{prediction.horizon_s:g}": prediction.metadata.get("relevance_status")
                        for prediction in forecast.predictions
                    },
                }
                if forecast.predictions
                else None,
            )
        )

    provenance = ForecastBatchProvenance(
        predictor_id=f"live-replay-{variant}-v1",
        predictor_family="none" if variant == "none" else variant,
        observation_tier=observed.provenance.observation_tier,
        frame=CoordinateFrame(name="world", units="m", axes=("x", "y")),
        dt_s=observed.provenance.dt_s,
        horizons_s=list(observed.provenance.horizons_s),
        scenario_id=trace.source.scenario_id,
        seed=trace.source.seed,
        timestamp=datetime.now(UTC).isoformat(),
        fallback_status="native",
        degraded_status="diagnostic_no_live_planner" if variant != "none" else "none",
        actor_ids=observed.provenance.actor_ids,
        actor_mask=observed.provenance.actor_mask,
        actor_mask_metadata=observed.provenance.actor_mask_metadata,
        feature_schema=schema,
        oracle_state=observed.provenance.oracle_state,
        actor_classes=observed.provenance.actor_classes,
    )

    return ForecastBatch(
        provenance=provenance,
        forecasts=forecasts,
        metadata={
            "artifact_role": "live_forecast_replay_gate_v1",
            "variant": variant,
            "source_trace_id": trace.trace_id,
            "source_planner_id": trace.source.planner_id,
            "step_index": step_index,
        },
    )


def _goal_position(trace: SimulationTraceExport) -> np.ndarray | None:
    """Extract a goal position from the trace when available.

    Returns:
        Goal position array or ``None``.
    """

    if not trace.frames:
        return None
    robot = trace.frames[-1].robot
    goal = robot.get("goal")
    if goal is not None:
        return np.asarray(goal, dtype=float)
    planner = trace.frames[0].planner
    goal = planner.get("goal") if isinstance(planner, dict) else None
    if goal is not None:
        return np.asarray(goal, dtype=float)
    return None


def _robot_action_from_frame(frame: SimulationTraceFrame) -> tuple[float, float]:
    """Extract recorded linear/angular velocities from a trace frame.

    Returns:
        Tuple of (linear_velocity, angular_velocity).
    """

    planner = frame.planner if isinstance(frame.planner, dict) else {}
    selected_action = planner.get("selected_action") if isinstance(planner, dict) else None
    if isinstance(selected_action, dict):
        return (
            float(selected_action.get("linear_velocity", 0.0)),
            float(selected_action.get("angular_velocity", 0.0)),
        )
    return 0.0, 0.0


def _build_socnav_observation(
    robot_position: np.ndarray,
    robot_heading: float,
    robot_speed: float,
    pedestrians: list[dict[str, Any]],
    dt_s: float,
    time_s: float,
) -> dict[str, Any]:
    """Build a minimal SocNav-structured observation for the baseline predictor.

    Pedestrian velocities are rotated into the ego frame to match the contract
    expected by :class:`BaselineProbabilisticPredictor`.

    Returns:
        Observation dict compatible with :class:`ProbabilisticPredictor.predict`.
    """

    cos_h = np.cos(robot_heading)
    sin_h = np.sin(robot_heading)
    world_velocities = np.asarray(
        [pedestrian.get("velocity", [0.0, 0.0]) for pedestrian in pedestrians], dtype=float
    )
    ego_velocities = np.zeros_like(world_velocities)
    if world_velocities.size > 0:
        ego_velocities[:, 0] = cos_h * world_velocities[:, 0] + sin_h * world_velocities[:, 1]
        ego_velocities[:, 1] = -sin_h * world_velocities[:, 0] + cos_h * world_velocities[:, 1]

    positions = np.asarray(
        [pedestrian.get("position", [0.0, 0.0]) for pedestrian in pedestrians], dtype=np.float32
    )
    velocities = np.asarray(ego_velocities, dtype=np.float32)
    if positions.size == 0:
        positions = np.zeros((0, 2), dtype=np.float32)
    if velocities.size == 0:
        velocities = np.zeros((0, 2), dtype=np.float32)

    return {
        "robot": {
            "position": np.asarray(robot_position, dtype=np.float32),
            "heading": np.array([float(robot_heading)], dtype=np.float32),
            "speed": np.array([float(robot_speed)], dtype=np.float32),
            "velocity_xy": np.array(
                [float(robot_speed) * cos_h, float(robot_speed) * sin_h], dtype=np.float32
            ),
            "angular_velocity": np.array([0.0], dtype=np.float32),
            "radius": np.array([0.3], dtype=np.float32),
        },
        "goal": {
            "current": np.asarray(robot_position, dtype=np.float32),
            "next": np.asarray(robot_position, dtype=np.float32),
        },
        "pedestrians": {
            "positions": positions,
            "velocities": velocities,
            "radius": np.array([0.3], dtype=np.float32),
            "count": np.array([float(len(pedestrians))], dtype=np.float32),
        },
        "map": {"size": np.array([50.0, 50.0], dtype=np.float32)},
        "sim": {
            "timestep": np.array([float(dt_s)], dtype=np.float32),
            "time_s": np.array([float(time_s)], dtype=np.float32),
        },
    }


def _forecast_brake_needed(
    prediction: ProbabilisticPrediction,
    robot_position: np.ndarray,
    brake_distance_m: float,
) -> bool:
    """Return True when any predicted mean comes within ``brake_distance_m``.

    This is the minimal forecast-aware decision used by the replay policy.  It
    treats the mean forecast as a deterministic signal and ignores covariance so
    the gate stays cheap and interpretable.
    """

    robot_pos = np.asarray(robot_position, dtype=float)
    for distribution in prediction.predictions:
        mean = np.asarray(distribution.mean, dtype=float)
        if mean.size == 0:
            continue
        distances = np.linalg.norm(mean - robot_pos, axis=1)
        if np.any(distances < brake_distance_m):
            return True
    return False


def _variant_brake_needed(
    predictor: BaselineProbabilisticPredictor | None,
    observation: dict[str, Any],
    robot_position: np.ndarray,
    risk_distance_m: float,
) -> bool:
    """Return whether a variant predictor requests braking at the current step."""

    if predictor is None:
        return False
    prediction = predictor.predict(observation)
    return _forecast_brake_needed(prediction, robot_position, risk_distance_m)


def _predictor_for_replay_variant(
    variant: str,
    config: LiveForecastReplayGateConfig,
    dt_s: float,
) -> BaselineProbabilisticPredictor | None:
    """Return the replay predictor, or None for the no-forecast baseline."""

    if variant == "none":
        return None
    return BaselineProbabilisticPredictor(
        variant=variant,
        horizons_s=config.horizons_s,
        dt_s=dt_s,
        risk_distance_m=_forecast_risk_distance_for_variant(variant, config),
    )


def _forecast_risk_distance_for_variant(
    variant: str, config: LiveForecastReplayGateConfig
) -> float:
    """Return the forecast-model risk distance for a forecast variant."""

    return float(config.variant_risk_distance_m.get(variant, config.risk_distance_m))


def _replay_brake_distance(config: LiveForecastReplayGateConfig) -> float:
    """Return the frozen replay-policy brake distance shared by forecast variants."""

    return float(config.replay_brake_distance_m)


def _frame_step_dt_s(
    trace: SimulationTraceExport,
    step_index: int,
    fallback_dt_s: float,
) -> float:
    """Return the positive frame-to-frame timestep for replay integration."""

    step_dt_s = trace.frames[step_index + 1].time_s - trace.frames[step_index].time_s
    return step_dt_s if step_dt_s > 0.0 else fallback_dt_s


def run_variant_closed_loop_replay(
    trace: SimulationTraceExport,
    variant: str,
    config: LiveForecastReplayGateConfig,
) -> dict[str, Any]:
    """Replay the trace with a forecast-aware brake policy for one variant.

    The ``none`` variant uses the same command-integration surface with
    forecast braking disabled.  Other variants instantiate a
    :class:`BaselineProbabilisticPredictor`, query it at each step, and override
    the recorded linear speed with zero when any predicted pedestrian mean comes
    within the configured replay brake distance of the robot.  The brake
    distance is intentionally shared by all forecast variants so the predictor
    is the independent variable and the replay control policy is frozen.

    The policy is intentionally minimal: it proves the forecast can influence
    closed-loop metrics without claiming that the brake heuristic is a good
    planner.

    Returns:
        Metric dictionary with the required closed-loop keys plus a
        ``replay_policy`` provenance string.
    """

    if not trace.frames:
        metrics = compute_baseline_closed_loop_metrics(
            trace,
            collision_distance_m=config.collision_distance_m,
            near_miss_distance_m=config.near_miss_distance_m,
            stop_speed_mps=config.stop_speed_mps,
            progress_goal_proximity_m=config.progress_goal_proximity_m,
        )
        metrics["replay_policy"] = "recorded_trace"
        return metrics

    dt_s = _trace_dt_s(trace)
    predictor = _predictor_for_replay_variant(variant, config, dt_s)
    brake_distance_m = _replay_brake_distance(config)

    initial_robot = trace.frames[0].robot
    robot_position = np.asarray(initial_robot.get("position", [0.0, 0.0]), dtype=float)
    robot_heading = float(initial_robot.get("heading", 0.0))

    robot_positions: list[np.ndarray] = [robot_position.copy()]
    robot_speeds: list[float] = []
    brake_steps = 0
    collision = False
    near_miss_count = 0
    min_distance = float("inf")
    false_positive_stops = 0

    for step_index, frame in enumerate(trace.frames):
        pedestrians = frame.pedestrians
        recorded_linear, recorded_angular = _robot_action_from_frame(frame)

        observation = _build_socnav_observation(
            robot_position,
            robot_heading,
            recorded_linear,
            pedestrians,
            dt_s,
            frame.time_s,
        )
        try:
            brake = _variant_brake_needed(
                predictor,
                observation,
                robot_position,
                brake_distance_m,
            )
        except (ValueError, TypeError):  # pragma: no cover - defensive fallback
            brake = False

        linear_velocity = 0.0 if brake else recorded_linear
        angular_velocity = 0.0 if brake else recorded_angular

        # Record metrics for the current step using the current robot position.
        frame_distances = [
            float(np.linalg.norm(np.asarray(pedestrian["position"], dtype=float) - robot_position))
            for pedestrian in pedestrians
        ]
        frame_min_distance = min(frame_distances) if frame_distances else float("inf")
        min_distance = min(min_distance, frame_min_distance)

        if frame_min_distance < config.collision_distance_m:
            collision = True
        elif frame_min_distance < config.near_miss_distance_m:
            near_miss_count += 1

        speed = abs(linear_velocity)
        robot_speeds.append(speed)
        is_stopped = speed < config.stop_speed_mps
        if is_stopped:
            brake_steps += 1
            if frame_min_distance > config.near_miss_distance_m:
                false_positive_stops += 1

        # Integrate forward for the next step, unless this is the last frame.
        if step_index < len(trace.frames) - 1:
            step_dt_s = _frame_step_dt_s(trace, step_index, dt_s)
            robot_heading = float(
                (robot_heading + angular_velocity * step_dt_s + np.pi) % (2.0 * np.pi) - np.pi
            )
            robot_position = robot_position + np.array(
                [np.cos(robot_heading), np.sin(robot_heading)], dtype=float
            ) * (linear_velocity * step_dt_s)
            robot_positions.append(robot_position.copy())

    start_position = robot_positions[0]
    end_position = robot_positions[-1]
    displacement = float(np.linalg.norm(end_position - start_position))
    goal = _goal_position(trace)
    if goal is not None:
        goal_distance_start = float(np.linalg.norm(goal - start_position))
        goal_distance_end = float(np.linalg.norm(goal - end_position))
        progress = max(0.0, goal_distance_start - goal_distance_end)
        reached_goal = goal_distance_end <= config.progress_goal_proximity_m
    else:
        progress = displacement
        reached_goal = None

    runtime_s = trace.frames[-1].time_s - trace.frames[0].time_s

    return {
        "collision": collision,
        "near_miss": near_miss_count,
        "min_distance_m": min_distance if np.isfinite(min_distance) else None,
        "stop_yield_timing_steps": brake_steps,
        "stop_yield_timing_s": brake_steps * dt_s,
        "progress_m": progress,
        "reached_goal": reached_goal,
        "false_positive_stops": false_positive_stops,
        "runtime_s": runtime_s,
        "replay_policy": "no_forecast_replay" if variant == "none" else "forecast_brake_replay",
    }


def compute_baseline_closed_loop_metrics(
    trace: SimulationTraceExport,
    *,
    collision_distance_m: float = DEFAULT_COLLISION_DISTANCE_M,
    near_miss_distance_m: float = DEFAULT_NEAR_MISS_DISTANCE_M,
    stop_speed_mps: float = DEFAULT_STOP_SPEED_MPS,
    progress_goal_proximity_m: float = DEFAULT_PROGRESS_GOAL_PROXIMITY_M,
) -> dict[str, Any]:
    """Compute baseline closed-loop metrics from a trace (the ``none`` variant).

    These metrics describe the recorded robot behaviour and do not depend on a
    forecast model.  They are intentionally simple so the gate stays cheap and
    deterministic.

    Returns:
        Metric dictionary with the required keys.
    """

    if not trace.frames:
        raise LiveForecastReplayGateError("trace has no frames")

    robot_positions = np.asarray([frame.robot["position"] for frame in trace.frames], dtype=float)
    robot_speeds = np.linalg.norm(
        np.asarray([frame.robot["velocity"] for frame in trace.frames], dtype=float),
        axis=1,
    )

    min_distance = float("inf")
    collision = False
    near_miss_count = 0
    stop_steps = 0
    false_positive_stops = 0

    for index, frame in enumerate(trace.frames):
        distances = [
            float(
                np.linalg.norm(
                    np.asarray(pedestrian["position"], dtype=float) - robot_positions[index]
                )
            )
            for pedestrian in frame.pedestrians
        ]
        frame_min_distance = min(distances) if distances else float("inf")
        min_distance = min(min_distance, frame_min_distance)

        if frame_min_distance < collision_distance_m:
            collision = True
        elif frame_min_distance < near_miss_distance_m:
            near_miss_count += 1

        is_stopped = robot_speeds[index] < stop_speed_mps
        if is_stopped:
            stop_steps += 1
            if frame_min_distance > near_miss_distance_m:
                false_positive_stops += 1

    start_position = robot_positions[0]
    end_position = robot_positions[-1]
    displacement = float(np.linalg.norm(end_position - start_position))
    goal = _goal_position(trace)
    if goal is not None:
        goal_distance_start = float(np.linalg.norm(goal - start_position))
        goal_distance_end = float(np.linalg.norm(goal - end_position))
        progress = max(0.0, goal_distance_start - goal_distance_end)
        reached_goal = goal_distance_end <= progress_goal_proximity_m
    else:
        progress = displacement
        reached_goal = None

    runtime_s = trace.frames[-1].time_s - trace.frames[0].time_s

    return {
        "collision": collision,
        "near_miss": near_miss_count,
        "min_distance_m": min_distance if np.isfinite(min_distance) else None,
        "stop_yield_timing_steps": stop_steps,
        "stop_yield_timing_s": stop_steps * _trace_dt_s(trace),
        "progress_m": progress,
        "reached_goal": reached_goal,
        "false_positive_stops": false_positive_stops,
        "runtime_s": runtime_s,
    }


def check_native_live_path_eligibility() -> dict[str, Any]:
    """Check whether the repository exposes a native live forecast-replay path.

    A native path requires at least one ``ProbabilisticPredictor``
    implementation and a planner/environment that can be configured to consume
    it with selectable forecast variants.

    Returns:
        Eligibility payload with ``live_path_available`` and
        ``missing_components``.
    """

    missing_components: list[str] = []

    # The protocol exists, but no concrete implementation is registered in the
    # repository for the baseline forecast variants.
    predictor_subclasses = _predictor_implementations()
    if not predictor_subclasses:
        missing_components.append(
            "no ProbabilisticPredictor implementation registered for baseline forecast variants"
        )

    # No environment/planner config key selects the forecast variant at runtime.
    if not _forecast_variant_config_key_exists():
        missing_components.append(
            "no environment or planner config key for selecting forecast_variant"
        )

    return {
        "live_path_available": not missing_components,
        "missing_components": missing_components,
        "probabilistic_predictor_implementations": predictor_subclasses,
        "forecast_variant_config_key_present": _forecast_variant_config_key_exists(),
    }


def _predictor_implementations() -> list[str]:
    """Return names of concrete ProbabilisticPredictor implementations.

    Returns:
        List of fully-qualified class names implementing the protocol.
    """

    implementations: list[str] = []
    for module_name in ("robot_sf.nav.baseline_probabilistic_predictor",):
        try:
            module = __import__(module_name, fromlist=["ProbabilisticPredictor"])
        except ImportError:
            continue
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, ProbabilisticPredictor)
                and obj is not ProbabilisticPredictor
                and not getattr(obj, "__abstractmethods__", None)
            ):
                implementations.append(f"{obj.__module__}.{obj.__qualname__}")
    return implementations


def _forecast_variant_config_key_exists() -> bool:
    """Return whether any config or config dataclass exposes forecast_variant.

    Returns:
        True when the environment settings dataclass has a ``forecast_variant`` field.
    """

    return any(field_info.name == "forecast_variant" for field_info in fields(EnvSettings))


def _available_horizons(
    trace: SimulationTraceExport,
    requested_horizons_s: tuple[float, ...],
) -> tuple[float, ...]:
    """Return horizons that fit within the trace duration.

    Drops horizons that would require frames beyond the trace end so that
    ground-truth labels exist for every kept horizon.

    Returns:
        Feasible horizon tuple.
    """

    if len(trace.frames) < 2:
        return requested_horizons_s
    dt_s = _trace_dt_s(trace)
    max_horizon_s = (len(trace.frames) - 1) * dt_s
    return tuple(h for h in requested_horizons_s if float(h) <= max_horizon_s + 1e-9)


def classify_live_forecast_replay_run(report: dict[str, Any]) -> str:
    """Classify a gate run as native, blocked, degraded, or diagnostic_only.

    The classification is fail-closed.  A run is ``native`` only when the cv
    forecast flows into the replay policy and produces closed-loop metrics that
    differ from the integrated no-forecast baseline.  When required components
    are missing the run is ``blocked``.  When the native path is technically
    present but the cv variant does not change closed-loop behavior the run is
    ``degraded``.  When only open-loop forecast diagnostics are available the
    run is ``diagnostic_only``.

    Args:
        report: Gate report produced by ``run_live_forecast_replay_gate``.

    Returns:
        One of ``VALID_RUN_CLASSIFICATIONS``.
    """

    eligibility = report.get("native_path_eligibility")
    if eligibility is None:
        eligibility = {}
    if not eligibility.get("live_path_available", False):
        if eligibility.get("missing_components"):
            return RUN_CLASSIFICATION_BLOCKED
        return RUN_CLASSIFICATION_DIAGNOSTIC_ONLY

    variant_results = report.get("variant_results", {})
    none_result = variant_results.get("none", {})
    cv_result = variant_results.get("cv", {})

    none_closed_loop = none_result.get("closed_loop_metrics")
    cv_closed_loop = cv_result.get("closed_loop_metrics")
    if none_closed_loop is None or cv_closed_loop is None:
        return RUN_CLASSIFICATION_DIAGNOSTIC_ONLY
    if not isinstance(none_closed_loop, dict) or not isinstance(cv_closed_loop, dict):
        return RUN_CLASSIFICATION_DIAGNOSTIC_ONLY

    if _closed_loop_metrics_equivalent(none_closed_loop, cv_closed_loop):
        return RUN_CLASSIFICATION_DEGRADED

    return RUN_CLASSIFICATION_NATIVE


def _closed_loop_metrics_equivalent(
    left: dict[str, Any] | Any,
    right: dict[str, Any] | Any,
    *,
    abs_tol: float = 1e-9,
) -> bool:
    """Return whether two closed-loop metric maps are equivalent for classification."""

    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    provenance_keys = {"replay_policy"}
    left_metric_keys = set(left) - provenance_keys
    right_metric_keys = set(right) - provenance_keys
    if left_metric_keys != right_metric_keys:
        return False
    for key in left_metric_keys:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, bool) or isinstance(right_value, bool):
            if type(left_value) is not type(right_value) or left_value != right_value:
                return False
            continue
        if isinstance(left_value, int | float) and isinstance(right_value, int | float):
            if not math.isclose(float(left_value), float(right_value), abs_tol=abs_tol):
                return False
            continue
        if left_value != right_value:
            return False
    return True


def _classification_reason(classification: str, report: dict[str, Any]) -> str:
    """Return a human-readable reason for the run classification."""

    if classification == RUN_CLASSIFICATION_BLOCKED:
        eligibility = report.get("native_path_eligibility")
        if eligibility is None:
            eligibility = {}
        missing = eligibility.get("missing_components", [])
        if missing:
            return "native live path blocked: " + "; ".join(missing)
        return "native live path blocked: eligibility check failed"
    if classification == RUN_CLASSIFICATION_DEGRADED:
        return (
            "native live path components are present but cv closed-loop metrics "
            "match the integrated no-forecast baseline, so cv does not affect replay behavior"
        )
    if classification == RUN_CLASSIFICATION_DIAGNOSTIC_ONLY:
        return "only open-loop forecast diagnostics are available for cv"
    if classification == RUN_CLASSIFICATION_NATIVE:
        return (
            "cv forecast produces closed-loop metrics that differ from the "
            "integrated no-forecast baseline"
        )
    return f"unknown classification: {classification}"


def _build_variant_result(
    trace: SimulationTraceExport,
    variant: str,
    config: LiveForecastReplayGateConfig,
    feasible_horizons: tuple[float, ...],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build the result dict for one forecast variant.

    Args:
        trace: Source simulation trace export.
        variant: Forecast variant to evaluate.
        config: Gate configuration.
        feasible_horizons: Horizons that fit within the trace duration.
        baseline_metrics: Recorded closed-loop metrics for fallback.

    Returns:
        JSON-compatible result dict for the variant.
    """

    if variant == "none":
        closed_loop_metrics = run_variant_closed_loop_replay(trace, variant, config)
        return {
            "forecast_batch_valid": None,
            "actor_count": 0,
            "forecast_metrics_status": "not_applicable",
            "forecast_metrics_error": None,
            "forecast_metrics_summary": None,
            "closed_loop_metrics": closed_loop_metrics,
            "closed_loop_metric_source": closed_loop_metrics.get(
                "replay_policy", "no_forecast_replay"
            ),
            "replay_policy_params": {"brake_distance_m": None},
        }
    batch = build_variant_forecast_batch(
        trace,
        variant,
        horizons_s=feasible_horizons,
        risk_distance_m=_forecast_risk_distance_for_variant(variant, config),
        observation_adapter=config.observation_adapter,
    )
    forecast_step = batch.metadata.get("step_index", 0)
    ground_truth: dict[str, list[list[float]]] = {}
    for actor_id in _all_pedestrian_actor_ids(trace):
        positions = _future_ground_truth_positions(
            trace, actor_id, forecast_step, feasible_horizons
        )
        if positions and len(positions) == len(feasible_horizons):
            ground_truth[actor_id] = [positions[float(h)].tolist() for h in feasible_horizons]

    try:
        forecast_report = evaluate_forecast_batch(
            batch,
            {
                actor_id: np.asarray(positions, dtype=float)
                for actor_id, positions in ground_truth.items()
            },
        )
        forecast_status = "ok"
        forecast_error = None
    except (ValueError, TypeError, FloatingPointError) as exc:  # pragma: no cover - defensive
        forecast_report = None
        forecast_status = "error"
        forecast_error = str(exc)

    try:
        closed_loop_metrics = run_variant_closed_loop_replay(trace, variant, config)
        closed_loop_source = closed_loop_metrics.get("replay_policy", "forecast_brake_replay")
        replay_error = None
    except (ValueError, TypeError, FloatingPointError) as exc:  # pragma: no cover - defensive
        closed_loop_metrics = dict(baseline_metrics)
        closed_loop_source = "baseline_recorded_trace_fallback"
        replay_error = str(exc)

    return {
        "forecast_batch_valid": validate_forecast_batch(batch.to_dict()) is not None,
        "actor_count": len(batch.forecasts),
        "forecast_metrics_status": forecast_status,
        "forecast_metrics_error": forecast_error,
        "forecast_metrics_summary": _summarize_forecast_metrics(forecast_report)
        if forecast_report
        else None,
        "closed_loop_metrics": closed_loop_metrics,
        "closed_loop_metric_source": closed_loop_source,
        "closed_loop_replay_error": replay_error,
        "replay_policy_params": {
            "brake_distance_m": _replay_brake_distance(config),
            "forecast_risk_distance_m": _forecast_risk_distance_for_variant(variant, config),
        },
    }


def run_live_forecast_replay_gate(
    trace: SimulationTraceExport,
    *,
    config: LiveForecastReplayGateConfig | None = None,
    variants: tuple[str, ...] = SMOKE_FORECAST_VARIANTS,
    repo_head: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Run the live same-seed forecast replay gate.

    Args:
        trace: Simulation trace export to replay and evaluate.
        config: Gate configuration.
        variants: Forecast variants to evaluate.  Defaults to the issue #2941
            smoke set (``none``, ``cv``).
        repo_head: Optional git HEAD sha for provenance.
        generated_at_utc: Optional deterministic ISO timestamp.

    Returns:
        JSON-compatible gate report.
    """

    config = config or LiveForecastReplayGateConfig()
    feasible_horizons = _available_horizons(trace, config.horizons_s)
    if not feasible_horizons:
        raise LiveForecastReplayGateError(
            "no requested forecast horizons fit within the trace duration"
        )

    if not variants:
        raise LiveForecastReplayGateError("at least one forecast variant must be requested")
    for variant in variants:
        if variant not in FORECAST_VARIANTS:
            raise LiveForecastReplayGateError(f"unsupported forecast variant: {variant}")

    eligibility = check_native_live_path_eligibility()

    baseline_metrics = compute_baseline_closed_loop_metrics(
        trace,
        collision_distance_m=config.collision_distance_m,
        near_miss_distance_m=config.near_miss_distance_m,
        stop_speed_mps=config.stop_speed_mps,
        progress_goal_proximity_m=config.progress_goal_proximity_m,
    )

    variant_results = {
        variant: _build_variant_result(
            trace,
            variant,
            config,
            feasible_horizons,
            baseline_metrics,
        )
        for variant in variants
    }

    report: dict[str, Any] = {
        "schema_version": LIVE_FORECAST_REPLAY_GATE_SCHEMA_VERSION,
        "issue": LIVE_FORECAST_REPLAY_GATE_ISSUE,
        "claim_boundary": (
            "Issue #3164 frozen-policy forecast-variant replay.  The gate evaluates "
            "forecast variants on the same recorded trace with a shared replay brake "
            "distance so forecast variant is the primary independent variable.  A "
            "BaselineProbabilisticPredictor implementation and a forecast_variant config "
            "key are now present, so the run is classified as native when cv closed-loop "
            "metrics differ from the integrated no-forecast baseline, and degraded when "
            "they match.  It does not claim that any variant improves safety, success, "
            "or runtime in a full planner stack."
        ),
        "provenance": {
            "trace_id": trace.trace_id,
            "scenario_id": trace.source.scenario_id,
            "seed": trace.source.seed,
            "planner_id": trace.source.planner_id,
            "episode_id": trace.source.episode_id,
            "repo_head": repo_head,
            "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
            "requested_horizons_s": list(config.horizons_s),
            "horizons_s": list(feasible_horizons),
            "variants": list(variants),
            "replay_policy": "frozen_forecast_brake_replay",
            "replay_brake_distance_m": _replay_brake_distance(config),
        },
        "native_path_eligibility": eligibility,
        "required_metrics": list(REQUIRED_METRICS),
        "baseline_closed_loop_metrics": dict(baseline_metrics),
        "variant_results": variant_results,
        "limitations": [
            "Non-none variants use a minimal forecast-brake replay policy with a frozen "
            "brake distance shared across forecast variants, not a production planner.  "
            "The gate proves the forecast can influence closed-loop metrics but does not "
            "prove benefit in a full planner stack.",
            "Open-loop forecast metrics are computed from a single frame per trace by default.",
            "Full-matrix expansion is gated by the run classification; only native runs "
            "should expand to the full variant matrix.",
        ],
    }
    classification = classify_live_forecast_replay_run(report)
    report["classification"] = classification
    report["classification_reason"] = _classification_reason(classification, report)
    report["status"] = classification
    report["full_matrix_expansion_recommended"] = classification == RUN_CLASSIFICATION_NATIVE
    return report


def _summarize_forecast_metrics(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract a compact summary from a ForecastBatch metrics report.

    Returns:
        Compact summary dict, or ``None`` when the input report is ``None``.
    """

    if report is None:
        return None
    aggregate_rows = report.get("aggregate_rows", [])
    summary: dict[str, dict[str, Any]] = {}
    for row in aggregate_rows:
        metric = row.get("metric")
        horizon = row.get("horizon_s")
        if metric is None or horizon is None:
            continue
        key = f"{metric}_{horizon:g}s"
        summary[key] = {
            "value": row.get("value"),
            "denominator": row.get("denominator"),
            "status": row.get("status"),
        }
    return {
        "active_actor_count": report.get("denominator_health", {}).get("active_actor_count"),
        "aggregate_row_count": len(aggregate_rows),
        "metrics": summary,
    }


def format_live_forecast_replay_gate_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary of the gate report.

    Returns:
        Markdown string with claim boundary, provenance, eligibility, metrics, and limitations.
    """

    provenance = report["provenance"]
    eligibility = report["native_path_eligibility"]
    baseline = report["baseline_closed_loop_metrics"]

    lines = [
        f"# Issue #{report['issue']} Native Forecast-Variant Replay Smoke",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## Classification",
        "",
        f"- **Classification:** {report.get('classification', 'unknown')}",
        f"- **Reason:** {report.get('classification_reason', '')}",
        f"- **Full-matrix expansion recommended:** "
        f"{report.get('full_matrix_expansion_recommended', False)}",
        "",
        "## Provenance",
        "",
        f"- **Trace:** {provenance['trace_id']}",
        f"- **Scenario:** {provenance['scenario_id']}",
        f"- **Seed:** {provenance['seed']}",
        f"- **Planner:** {provenance['planner_id']}",
        f"- **Repo HEAD:** `{provenance.get('repo_head') or 'unknown'}`",
        f"- **Generated at (UTC):** {provenance['generated_at_utc']}",
        f"- **Variants:** {provenance.get('variants', [])}",
        "",
        "## Native Path Eligibility",
        "",
        f"- **Live path available:** {eligibility['live_path_available']}",
        f"- **Predictor implementations:** {len(eligibility['probabilistic_predictor_implementations'])}",
        f"- **Forecast variant config key present:** {eligibility['forecast_variant_config_key_present']}",
    ]

    if eligibility["missing_components"]:
        lines.extend(["", "### Missing Components"])
        for component in eligibility["missing_components"]:
            lines.append(f"- {component}")

    lines.extend(
        [
            "",
            "## Recorded Trace Closed-Loop Metrics",
            "",
            f"- **Collision:** {baseline['collision']}",
            f"- **Near miss timesteps:** {baseline['near_miss']}",
            f"- **Min distance (m):** {baseline['min_distance_m']}",
            f"- **Stop/yield steps:** {baseline['stop_yield_timing_steps']}",
            f"- **Progress (m):** {baseline['progress_m']}",
            f"- **False-positive stops:** {baseline['false_positive_stops']}",
            f"- **Runtime (s):** {baseline['runtime_s']}",
            "",
            "## Variant Results",
            "",
            "| Variant | Actors | Forecast Metrics | Closed-Loop Source |",
            "|---|---|---|---|",
        ]
    )
    for variant, result in report["variant_results"].items():
        metrics_status = result["forecast_metrics_status"]
        closed_loop_source = result.get("closed_loop_metric_source", "unknown")
        lines.append(
            f"| {variant} | {result['actor_count']} | {metrics_status} | {closed_loop_source} |"
        )

    lines.extend(["", "## Limitations"])
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")

    return "\n".join(lines) + "\n"


def write_live_forecast_replay_gate_report(
    report: dict[str, Any],
    output_dir: Path,
    *,
    filename_stem: str = "live_forecast_replay_gate_report",
) -> tuple[Path, Path]:
    """Write the JSON and Markdown gate reports to disk.

    Returns:
        Tuple of (json_path, markdown_path).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{filename_stem}.json"
    md_path = output_dir / f"{filename_stem}.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(format_live_forecast_replay_gate_markdown(report), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "DEFAULT_COLLISION_DISTANCE_M",
    "DEFAULT_HORIZONS_S",
    "DEFAULT_NEAR_MISS_DISTANCE_M",
    "DEFAULT_RISK_DISTANCE_M",
    "DEFAULT_STOP_SPEED_MPS",
    "FORECAST_VARIANTS",
    "LIVE_FORECAST_REPLAY_GATE_ISSUE",
    "LIVE_FORECAST_REPLAY_GATE_SCHEMA_VERSION",
    "REQUIRED_METRICS",
    "RUN_CLASSIFICATION_BLOCKED",
    "RUN_CLASSIFICATION_DEGRADED",
    "RUN_CLASSIFICATION_DIAGNOSTIC_ONLY",
    "RUN_CLASSIFICATION_NATIVE",
    "SMOKE_FORECAST_VARIANTS",
    "VALID_RUN_CLASSIFICATIONS",
    "LiveForecastReplayGateConfig",
    "LiveForecastReplayGateError",
    "build_variant_forecast_batch",
    "check_native_live_path_eligibility",
    "classify_live_forecast_replay_run",
    "compute_baseline_closed_loop_metrics",
    "format_live_forecast_replay_gate_markdown",
    "load_trace_tolerant",
    "run_live_forecast_replay_gate",
    "write_live_forecast_replay_gate_report",
]
