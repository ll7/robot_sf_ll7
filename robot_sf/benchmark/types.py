"""Core benchmark data structures (Phase 3.4 tasks T040, T041, T044).

These dataclasses provide typed containers for scenario specifications,
episode records, and resume manifests. They are deliberately minimal and
avoid introducing runtime dependencies (pure typing + stdlib) so they can
be imported in lightweight tooling (schema generation, hashing, etc.).

Serialization: writing to JSONL will typically convert instances to
``dict`` via ``dataclasses.asdict`` or explicit ``to_dict`` helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import (
    UTC,  # type: ignore[attr-defined]
    datetime,
)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from robot_sf.benchmark.algorithm_readiness import BenchmarkProfile


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


@dataclass(frozen=True, slots=True)
class PlannerRuntime:
    """Planner lifecycle hooks, policy callable, and native-action flag.

    Groups the six ``planner_*`` parameters of ``_run_episode_step_loop``
    (policy_fn, planner_bind_env, planner_reset, planner_close, planner_stats,
    planner_native_action) into a single typed object.
    """

    policy_fn: Callable[..., Any]
    planner_bind_env: Callable[..., Any] | None = None
    planner_reset: Callable[..., Any] | None = None
    planner_close: Callable[..., Any] | None = None
    planner_stats: Callable[..., Any] | None = None
    planner_native_action: bool = False


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """Observation-noise parameters for the episode step loop.

    Groups the ``noise_spec``, ``noise_rng``, ``noise_state``, and
    ``noise_stats`` keyword arguments into a single typed object.
    """

    spec: dict[str, Any]
    rng: Any | None = None
    state: Any | None = None
    stats: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class MapBatchConfig:
    """Consolidated keyword arguments for ``run_map_batch``.

    Mirrors the keyword-only parameters of ``run_map_batch`` (excluding
    ``batch_config`` itself and the ``scenario_path`` I/O argument), so a caller
    can bundle and validate the batch configuration in one typed object before
    passing it to the runner via ``run_map_batch(..., batch_config=cfg)``.
    """

    horizon: int | None = None
    dt: float | None = None
    record_forces: bool = True
    snqi_weights: dict[str, float] | None = None
    snqi_baseline: dict[str, dict[str, float]] | None = None
    algo: str = "goal"
    algo_config_path: str | None = None
    benchmark_profile: BenchmarkProfile = "baseline-safe"
    socnav_missing_prereq_policy: str = "fail-fast"
    adapter_impact_eval: bool = False
    experimental_ped_impact: bool = False
    ped_impact_radius_m: float = 2.0
    ped_impact_window_steps: int = 5
    observation_mode: str | None = None
    observation_level: str | None = None
    benchmark_track: str | None = None
    track_schema_version: str | None = None
    observation_noise: dict[str, Any] | None = None
    tracking_precision: dict[str, Any] | None = None
    synthetic_actuation_profile: dict[str, Any] | None = None
    latency_stress_profile: dict[str, Any] | None = None
    safety_wrapper: dict[str, Any] | None = None
    cbf_safety_filter: dict[str, Any] | None = None
    record_planner_decision_trace: bool = False
    record_simulation_step_trace: bool = False
    multiprocessing_context: Any | None = None
    workers: int = 1
    resume: bool = True
    circuit_breaker_threshold: int | None = None


__all__ = [
    "EpisodeRecord",
    "MapBatchConfig",
    "MetricsBundle",
    "NoiseConfig",
    "PlannerRuntime",
    "ResumeManifest",
    "SNQIWeights",
    "ScenarioSpec",
]
