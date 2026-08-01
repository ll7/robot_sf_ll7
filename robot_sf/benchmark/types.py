"""Core benchmark data structures (Phase 3.4 tasks T040, T041, T044).


These dataclasses and TypedDicts provide typed containers for scenario
specifications, episode records, resume manifests, and episode payloads.
They are deliberately minimal and avoid introducing runtime dependencies
(pure typing + stdlib) so they can be imported in lightweight tooling
(schema generation, hashing, etc.).

Serialization: writing to JSONL will typically convert instances to
``dict`` via ``dataclasses.asdict`` or explicit ``to_dict`` helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import (
    UTC,  # type: ignore[attr-defined]
    datetime,
)
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

# ---------------------------------------------------------------------------
# TypedDicts for dict-based episode payloads
# ---------------------------------------------------------------------------


class NoiseSpec(TypedDict, total=False):
    """Normalized observation-noise specification.

    All keys are optional by construction: the dict flows through stages that
    may carry only partial or pre-normalized content. The canonical shape is
    produced by ``normalize_observation_noise_spec``.
    """

    enabled: bool
    profile: str
    seed: int | None
    pose_noise_std_m: float
    heading_noise_std_rad: float
    lidar_dropout_prob: float
    lidar_dropout_value: float
    pedestrian_position_noise_std_m: float
    pedestrian_false_negative_prob: float
    pedestrian_occlusion_max_range_m: float | None
    observation_delay_steps: int
    pedestrian_false_positive_prob: float
    pedestrian_false_positive_radius_m: float
    pedestrian_false_positive_radius: float
    interpretation: str


class TrackingPrecisionSpeedContract(TypedDict, total=False):
    """Nested speed-contract block inside ``TrackingPrecisionSpec``."""

    threshold_m: float
    default_speed: float
    defensive_speed: float
    mode: str


class TrackingPrecisionSpec(TypedDict, total=False):
    """Normalized tracking-precision specification.

    All keys are optional by construction. The canonical shape is produced by
    ``normalize_tracking_precision_spec``.
    """

    enabled: bool
    target_motp_m: float
    speed_contract: TrackingPrecisionSpeedContract
    seed_salt: int
    schema_version: str
    interpretation: str


class OutcomePayload(TypedDict, total=False):
    """Episode outcome flags."""

    route_complete: bool
    collision_event: bool
    timeout_event: bool


class AdapterImpact(TypedDict, total=False):
    """Adapter-impact counters for algorithm metadata."""

    requested: bool
    native_steps: int
    adapted_steps: int
    status: str
    execution_mode: str
    adapter_fraction: float


class AlgoMeta(TypedDict, total=False):
    """Algorithm metadata dict assembled by the policy builder and enriched
    during episode finalization.

    All keys are optional: the dict grows incrementally through the episode
    lifecycle.
    """

    algorithm: str
    canonical_algorithm: str
    baseline_category: str
    policy_semantics: str
    status: str
    fallback_reason: str
    benchmark_track: dict[str, Any]
    config: dict[str, Any]
    config_hash: str
    kinematics_feasibility: dict[str, Any]
    observation_spec: dict[str, Any]
    observation_level: dict[str, Any]
    planner_kinematics: dict[str, Any]
    planner_contract: dict[str, Any]
    adapter_impact: AdapterImpact
    tracking_precision: dict[str, Any]
    ammv_feasibility: dict[str, Any]
    planner_runtime: dict[str, Any]
    foresight_prediction: dict[str, Any]
    planner_decision_trace: PlannerDecisionTrace
    topology_guided_episode: dict[str, Any]
    simulation_step_trace: dict[str, Any]
    safety_wrapper: dict[str, Any]
    cbf_safety_filter: dict[str, Any]
    intent_conditioned_behavior: dict[str, Any]
    cyclist_like_vru: dict[str, Any]
    fast_bicycle_actor: dict[str, Any]
    synthetic_actuation: dict[str, Any]
    latency_stress: dict[str, Any]
    public_requirement: dict[str, Any]
    observation_visibility: dict[str, Any]
    safety_shield_contract: dict[str, Any]
    shield_stats: dict[str, Any]
    native_command: dict[str, Any]
    planner_diagnostics: dict[str, Any]
    fallback_or_degraded: bool
    _native_run_state: dict[str, Any]
    upstream_reference: dict[str, Any]
    stochastic_reference: bool
    distinct_from_goal_baseline: bool
    learned_checkpoint_observation_contract: dict[str, Any]


class PlannerTargetGoal(TypedDict, total=False):
    """DWA target-goal detail embedded in a planner-decision step."""

    kind: str
    x: float
    y: float


class PlannerDynamicWindow(TypedDict, total=False):
    """Reachable linear/angular bounds embedded in a DWA trace step."""

    v_min: float
    v_max: float
    w_min: float
    w_max: float


class PlannerDecisionTraceEntry(TypedDict, total=False):
    """Single step entry in the planner-decision trace."""

    step: int
    selected_source: str
    selected_command: list[float]
    selected_score: float | None
    static_recenter: float
    route_arc_progress: float
    goal_progress: float
    progress_windows: dict[str, float]
    distance_to_goal_m: float
    route_progress_from_start_m: float
    robot_x_m: float
    robot_y_m: float
    topology_guided: dict[str, Any]
    topology_guided_config: dict[str, Any]
    topology_lane_status: str
    topology_fallback_status: str
    topology_fallback_reason: str
    topology_candidate_availability: dict[str, Any]
    topology_command_influence: dict[str, Any]
    constraint_reason: str
    candidate_total: int
    candidate_feasible: int
    candidate_infeasible: int
    feasible_score_min: float
    feasible_score_max: float
    dynamic_window: PlannerDynamicWindow
    target_goal: PlannerTargetGoal
    global_route_probe_activated: bool


class PlannerDecisionTrace(TypedDict, total=False):
    """Episode-level planner-decision trace envelope."""

    schema_version: str
    dt: float
    initial_goal_distance_m: float
    steps: list[PlannerDecisionTraceEntry]


class EpisodeRecordDict(TypedDict, total=False):
    """Top-level episode record returned by ``run_map_episode``.

    All keys are optional; the canonical shape is produced by
    ``_finalize_episode_record``.
    """

    version: str
    episode_id: str
    scenario_id: str
    seed: int
    scenario_params: dict[str, Any]
    metrics: dict[str, Any]
    safety_predicates: dict[str, Any]
    public_requirement: dict[str, Any]
    algorithm_metadata: AlgoMeta
    observation_noise: NoiseSpec
    observation_noise_hash: str
    observation_noise_stats: dict[str, Any]
    tracking_precision: TrackingPrecisionSpec
    tracking_precision_hash: str
    algo: str
    observation_mode: str
    observation_level: str
    config_hash: str
    git_hash: str
    timestamps: dict[str, str]
    status: str
    steps: int
    horizon: int
    wall_time_sec: float
    timing: dict[str, float]
    termination_reason: str
    outcome: OutcomePayload
    integrity: dict[str, Any]
    pedestrian_model: dict[str, Any]
    development_pedestrian_model: str
    evaluation_pedestrian_model: str
    failure_mechanism: dict[str, Any]
    interaction_exposure: dict[str, Any]
    event_ledger: dict[str, Any]
    benchmark_track: str
    track_schema_version: str
    result_provenance: dict[str, Any]
    metric_parameters: dict[str, Any]
    notes: str
    tags: list[str]
    identity: dict[str, Any]
    video: dict[str, Any]
    low_progress_window: dict[str, Any]
    recenter_activation_count: int
    distance_to_goal_delta: dict[str, Any]
    local_minimum_indicator: dict[str, Any]
    row_status: str


@dataclass(slots=True)
class ScenarioSpec:
    """Scenario specification (single row from scenario matrix).

    Required fields align with `scenario-matrix.schema.v1.json`.
    Additional algorithm-specific configuration can be passed via
    the optional `algo_config_path` or embedded metadata dict.
    """

    id: str
    algo: str
    map: str
    episodes: int
    seed: int
    notes: str | None = None
    algo_config_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:  # stable conversion
        """Convert to a JSON-serializable dict.

        Returns:
            Dict representation of the scenario spec.
        """
        return asdict(self)


@dataclass(slots=True)
class MetricsBundle:
    """Container for computed metric values.

    Internally just wraps a mapping but gives a semantic type for future
    validation or access helpers (e.g., enforcing presence of required keys).
    """

    values: dict[str, float]

    def get(self, name: str, default: float | None = None) -> float | None:
        """Return a metric value or a default.

        Returns:
            Metric value if present, otherwise the provided default.
        """
        return self.values.get(name, default)

    def to_dict(self) -> dict[str, float]:
        """Convert to a plain dict.

        Returns:
            Dict of metric values.
        """
        return dict(self.values)


@dataclass(slots=True)
class EpisodeRecord:
    """High-level episode record suitable for JSONL persistence.

    The `raw` field can contain implementation-specific extras (timing, identity
    materials, debug traces) that are not part of the stable metrics payload.
    """

    version: str
    episode_id: str
    scenario_id: str
    seed: int
    metrics: MetricsBundle
    algo: str | None = None
    horizon: int | None = None
    timing: dict[str, float] | None = None
    tags: list[str] | None = None
    identity: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict.

        Returns:
            Dict representation of the episode record.
        """
        d = asdict(self)
        # flatten metrics bundle for JSON writing
        d["metrics"] = self.metrics.to_dict()
        return d


@dataclass(slots=True)
class SNQIWeights:
    """Weight file content for SNQI computation (subset for early phases)."""

    version: str
    weights: Mapping[str, float]
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict.

        Returns:
            Dict representation of SNQI weights.
        """
        return {"version": self.version, "weights": dict(self.weights), "meta": self.meta or {}}


@dataclass(slots=True)
class ResumeManifest:
    """Resume manifest describing completed episode ids (Phase 3.6/3.3 link)."""

    version: str
    episodes: list[str]
    meta: dict[str, Any] | None = None
    generated_at: str = field(
        default_factory=lambda: (
            datetime.now(UTC).astimezone(UTC).replace(microsecond=0).isoformat()
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict.

        Returns:
            Dict representation of the resume manifest.
        """
        return {
            "version": self.version,
            "episodes": list(self.episodes),
            "meta": self.meta or {},
            "generated_at": self.generated_at,
        }


__all__ = [
    "AdapterImpact",
    "AlgoMeta",
    "EpisodeRecord",
    "EpisodeRecordDict",
    "MetricsBundle",
    "NoiseSpec",
    "OutcomePayload",
    "PlannerDecisionTrace",
    "PlannerDecisionTraceEntry",
    "PlannerDynamicWindow",
    "PlannerTargetGoal",
    "ResumeManifest",
    "SNQIWeights",
    "ScenarioSpec",
    "TrackingPrecisionSpec",
    "TrackingPrecisionSpeedContract",
]
