"""Configuration and public contracts for adversarial scenario search."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping

    from robot_sf.adversarial.attribution import FailureAttribution
    from robot_sf.adversarial.certification import CertificationStatus


@dataclass(frozen=True)
class Pose2D:
    """Planar pose used by the adversarial candidate contract."""

    x: float
    y: float
    theta: float = 0.0

    def as_waypoint(self) -> list[float]:
        """Return the pose position as a route waypoint."""
        return [float(self.x), float(self.y)]

    def to_json(self) -> dict[str, float]:
        """Return a JSON-serializable payload."""
        return {"x": float(self.x), "y": float(self.y), "theta": float(self.theta)}


@dataclass(frozen=True)
class CandidateSpec:
    """One sampled adversarial scenario candidate."""

    start: Pose2D
    goal: Pose2D
    spawn_time_s: float
    pedestrian_speed_mps: float
    pedestrian_delay_s: float
    scenario_seed: int
    pedestrian_acceleration_mps2: float | None = None
    group_size: int | None = None
    vru_profile: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate payload."""
        payload: dict[str, Any] = {
            "start": self.start.to_json(),
            "goal": self.goal.to_json(),
            "spawn_time_s": float(self.spawn_time_s),
            "pedestrian_speed_mps": float(self.pedestrian_speed_mps),
            "pedestrian_delay_s": float(self.pedestrian_delay_s),
            "scenario_seed": int(self.scenario_seed),
        }
        if self.pedestrian_acceleration_mps2 is not None:
            payload["pedestrian_acceleration_mps2"] = float(self.pedestrian_acceleration_mps2)
        if self.group_size is not None:
            payload["group_size"] = int(self.group_size)
        if self.vru_profile is not None:
            payload["vru_profile"] = str(self.vru_profile)
        return payload


@dataclass(frozen=True)
class MultiPedCandidateSpec:
    """One pedestrian in a scripted multi-pedestrian adversarial candidate."""

    id: str
    start: Pose2D
    goal: Pose2D
    spawn_time_s: float = 0.0
    speed_mps: float = 1.0
    delay_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        index: int,
    ) -> MultiPedCandidateSpec:
        """Build one multi-pedestrian entry from a YAML/JSON mapping."""
        if not isinstance(payload, dict):
            raise ValueError(f"pedestrians[{index}] must be a mapping")
        raw_id = payload.get("id")
        pedestrian_id = str(raw_id).strip() if raw_id is not None else ""
        if not pedestrian_id:
            raise ValueError(f"pedestrians[{index}].id must be non-empty")
        start = _pose_from_mapping(payload.get("start"), name=f"pedestrians[{index}].start")
        goal = _pose_from_mapping(payload.get("goal"), name=f"pedestrians[{index}].goal")
        raw_metadata = payload.get("metadata", {})
        if raw_metadata is None:
            raw_metadata = {}
        if not isinstance(raw_metadata, dict):
            raise ValueError(f"pedestrians[{index}].metadata must be a mapping")
        spawn_time_raw = payload.get("spawn_time_s")
        speed_raw = payload.get("speed_mps", payload.get("pedestrian_speed_mps"))
        delay_raw = payload.get("delay_s", payload.get("pedestrian_delay_s"))
        return cls(
            id=pedestrian_id,
            start=start,
            goal=goal,
            spawn_time_s=float(spawn_time_raw) if spawn_time_raw is not None else 0.0,
            speed_mps=float(speed_raw) if speed_raw is not None else 1.0,
            delay_s=float(delay_raw) if delay_raw is not None else 0.0,
            metadata=dict(raw_metadata),
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible payload for this pedestrian candidate."""
        return {
            "id": self.id,
            "start": self.start.to_json(),
            "goal": self.goal.to_json(),
            "spawn_time_s": float(self.spawn_time_s),
            "speed_mps": float(self.speed_mps),
            "delay_s": float(self.delay_s),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MultiPedAdversarialConfig:
    """Validated schema for scripted multi-pedestrian adversarial candidates.

    This contract is intentionally additive and schema-only. Runtime environment
    integration remains a separate benchmark-sensitive step.
    """

    family: str
    scenario_seed: int
    pedestrians: list[MultiPedCandidateSpec]
    schema_version: str = "adversarial-multi-ped.v1"
    min_start_goal_distance_m: float = 0.25

    @classmethod
    def from_file(cls, path: str | Path) -> MultiPedAdversarialConfig:
        """Load and validate a multi-pedestrian adversarial config YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Multi-ped adversarial config must be a mapping: {path}")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MultiPedAdversarialConfig:
        """Build a multi-pedestrian adversarial config from a mapping."""
        pedestrians = payload.get("pedestrians")
        if not isinstance(pedestrians, list) or not pedestrians:
            raise ValueError("pedestrians must be a non-empty list")
        constraints = payload.get("constraints", {})
        if constraints is None:
            constraints = {}
        if not isinstance(constraints, dict):
            raise ValueError("constraints must be a mapping")
        config = cls(
            schema_version=str(payload.get("schema_version", "adversarial-multi-ped.v1")),
            family=str(payload.get("family", "")).strip(),
            scenario_seed=int(payload.get("scenario_seed", 0)),
            min_start_goal_distance_m=float(constraints.get("min_start_goal_distance_m", 0.25)),
            pedestrians=[
                MultiPedCandidateSpec.from_mapping(entry, index=index)
                for index, entry in enumerate(pedestrians)
            ],
        )
        if not config.family:
            raise ValueError("family must be non-empty")
        if config.schema_version != "adversarial-multi-ped.v1":
            raise ValueError("schema_version must be adversarial-multi-ped.v1")
        if (
            not math.isfinite(config.min_start_goal_distance_m)
            or config.min_start_goal_distance_m < 0.0
        ):
            raise ValueError("min_start_goal_distance_m must be finite and >= 0")
        errors = config.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return config

    def validate(self) -> list[str]:
        """Return validation errors for the multi-pedestrian candidate contract."""
        errors: list[str] = []
        if self.scenario_seed < 0:
            errors.append("scenario_seed must be non-negative")
        if (
            not math.isfinite(self.min_start_goal_distance_m)
            or self.min_start_goal_distance_m < 0.0
        ):
            errors.append("min_start_goal_distance_m must be finite and >= 0")
        ids = [ped.id for ped in self.pedestrians]
        if len(ids) != len(set(ids)):
            seen: set[str] = set()
            duplicates: list[str] = []
            for ped_id in ids:
                if ped_id in seen and ped_id not in duplicates:
                    duplicates.append(ped_id)
                seen.add(ped_id)
            errors.append(
                f"pedestrians ids must be unique; duplicates: {', '.join(sorted(duplicates))}"
            )
        for index, pedestrian in enumerate(self.pedestrians):
            errors.extend(_validate_multi_pedestrian_entry(pedestrian, index=index))
            dx = float(pedestrian.goal.x) - float(pedestrian.start.x)
            dy = float(pedestrian.goal.y) - float(pedestrian.start.y)
            distance = math.hypot(dx, dy)
            if distance < self.min_start_goal_distance_m:
                errors.append(
                    f"pedestrians[{index}] start and goal distance ({distance:.3f}m) is less "
                    f"than min_start_goal_distance_m ({self.min_start_goal_distance_m}m)"
                )
        return errors

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible multi-pedestrian adversarial payload."""
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "scenario_seed": int(self.scenario_seed),
            "constraints": {"min_start_goal_distance_m": self.min_start_goal_distance_m},
            "pedestrians": [pedestrian.to_json() for pedestrian in self.pedestrians],
        }


def _pose_from_mapping(payload: object, *, name: str) -> Pose2D:
    """Build a pose from a mapping with x/y and optional theta fields."""
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    if "x" not in payload or "y" not in payload:
        raise ValueError(f"{name} must define x and y")
    theta_raw = payload.get("theta")
    theta = float(theta_raw) if theta_raw is not None else 0.0
    return Pose2D(float(payload["x"]), float(payload["y"]), theta)


def _validate_multi_pedestrian_entry(
    pedestrian: MultiPedCandidateSpec,
    *,
    index: int,
) -> list[str]:
    """Return validation errors for one pedestrian entry."""
    errors: list[str] = []
    values = {
        "start.x": pedestrian.start.x,
        "start.y": pedestrian.start.y,
        "start.theta": pedestrian.start.theta,
        "goal.x": pedestrian.goal.x,
        "goal.y": pedestrian.goal.y,
        "goal.theta": pedestrian.goal.theta,
        "spawn_time_s": pedestrian.spawn_time_s,
        "speed_mps": pedestrian.speed_mps,
        "delay_s": pedestrian.delay_s,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            errors.append(f"pedestrians[{index}].{name} must be finite")
    if pedestrian.spawn_time_s < 0.0:
        errors.append(f"pedestrians[{index}].spawn_time_s must be non-negative")
    if pedestrian.speed_mps <= 0.0:
        errors.append(f"pedestrians[{index}].speed_mps must be positive")
    if pedestrian.delay_s < 0.0:
        errors.append(f"pedestrians[{index}].delay_s must be non-negative")
    return errors


@dataclass(frozen=True)
class CandidateEvaluation:
    """Evaluation result for one candidate."""

    candidate: CandidateSpec
    certification_status: CertificationStatus
    objective_value: float | None
    failure_attribution: FailureAttribution | None
    episode_record_path: Path | None
    trajectory_csv_path: Path | None
    scenario_yaml_path: Path | None
    bundle_path: Path | None = None
    error: str | None = None

    def with_objective(self, objective_value: float | None) -> CandidateEvaluation:
        """Return a copy with an objective score attached."""
        return replace(self, objective_value=objective_value)


@dataclass(frozen=True)
class SearchRunResult:
    """Summary returned by :func:`run_adversarial_search`."""

    manifest_path: Path
    best_candidate: CandidateEvaluation | None
    best_bundle_path: Path | None
    num_candidates: int
    num_valid_candidates: int
    num_invalid_candidates: int
    num_failed_evaluations: int

    @property
    def best_objective_value(self) -> float | None:
        """Return the best objective value, if any candidate scored."""
        if self.best_candidate is None:
            return None
        return self.best_candidate.objective_value


@dataclass(frozen=True)
class RangeConfig:
    """Inclusive numeric sampling range."""

    min: float
    max: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, name: str) -> RangeConfig:
        """Build a range from YAML payload."""
        if "min" not in payload or "max" not in payload:
            raise ValueError(f"search_space.{name} must define min and max")
        lo = float(payload["min"])
        hi = float(payload["max"])
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError(f"search_space.{name} bounds must be finite")
        if lo > hi:
            raise ValueError(f"search_space.{name}.min must be <= max")
        return cls(min=lo, max=hi)

    def sample(self, rng: Random) -> float:
        """Sample a value from the range."""
        return float(rng.uniform(self.min, self.max))

    def contains(self, value: float) -> bool:
        """Return whether a value falls inside the inclusive range."""
        return self.min <= float(value) <= self.max

    def to_json(self) -> dict[str, float]:
        """Return a JSON-serializable range payload."""
        return {"min": float(self.min), "max": float(self.max)}


@dataclass(frozen=True)
class SearchSpaceConfig:
    """Validated candidate sampling space for adversarial search."""

    start_x: RangeConfig
    start_y: RangeConfig
    goal_x: RangeConfig
    goal_y: RangeConfig
    spawn_time_s: RangeConfig = field(default_factory=lambda: RangeConfig(0.0, 0.0))
    pedestrian_speed_mps: RangeConfig = field(default_factory=lambda: RangeConfig(1.0, 1.0))
    pedestrian_delay_s: RangeConfig = field(default_factory=lambda: RangeConfig(0.0, 0.0))
    scenario_seed: RangeConfig = field(default_factory=lambda: RangeConfig(1.0, 1.0))
    min_start_goal_distance_m: float = 0.25
    pedestrian_id: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> SearchSpaceConfig:
        """Load and validate a search-space YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Search-space config must be a mapping: {path}")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SearchSpaceConfig:
        """Build a search-space config from a mapping."""
        variables = payload.get("variables", payload)
        if not isinstance(variables, dict):
            raise ValueError("search-space variables must be a mapping")

        def _range(name: str, default: tuple[float, float] | None = None) -> RangeConfig:
            """Read one named range from the variables mapping.

            Returns:
                RangeConfig: Explicit or defaulted range config.
            """
            raw_value = variables.get(name)
            if raw_value is None:
                if default is None:
                    raise ValueError(f"search_space.{name} is required")
                return RangeConfig(*default)
            if not isinstance(raw_value, dict):
                raise ValueError(f"search_space.{name} must be a mapping")
            return RangeConfig.from_mapping(raw_value, name=name)

        constraints = payload.get("constraints", {})
        if constraints is None:
            constraints = {}
        if not isinstance(constraints, dict):
            raise ValueError("search-space constraints must be a mapping")
        pedestrian = payload.get("pedestrian", {})
        if pedestrian is None:
            pedestrian = {}
        if not isinstance(pedestrian, dict):
            raise ValueError("search-space pedestrian section must be a mapping")
        pedestrian_id = pedestrian.get("id")
        if pedestrian_id is not None:
            pedestrian_id = str(pedestrian_id).strip() or None

        config = cls(
            start_x=_range("start_x"),
            start_y=_range("start_y"),
            goal_x=_range("goal_x"),
            goal_y=_range("goal_y"),
            spawn_time_s=_range("spawn_time_s", (0.0, 0.0)),
            pedestrian_speed_mps=_range("pedestrian_speed_mps", (1.0, 1.0)),
            pedestrian_delay_s=_range("pedestrian_delay_s", (0.0, 0.0)),
            scenario_seed=_range("scenario_seed", (1.0, 1.0)),
            min_start_goal_distance_m=float(constraints.get("min_start_goal_distance_m", 0.25)),
            pedestrian_id=pedestrian_id,
        )
        if not config.scenario_seed.min.is_integer() or not config.scenario_seed.max.is_integer():
            raise ValueError("search-space scenario_seed bounds must be integers")
        if (
            not math.isfinite(config.min_start_goal_distance_m)
            or config.min_start_goal_distance_m < 0.0
        ):
            raise ValueError("search-space min_start_goal_distance_m must be finite and >= 0")
        return config

    def sample_candidate(self, rng: Random) -> CandidateSpec:
        """Sample a candidate deterministically from the provided RNG."""
        return CandidateSpec(
            start=Pose2D(self.start_x.sample(rng), self.start_y.sample(rng)),
            goal=Pose2D(self.goal_x.sample(rng), self.goal_y.sample(rng)),
            spawn_time_s=self.spawn_time_s.sample(rng),
            pedestrian_speed_mps=self.pedestrian_speed_mps.sample(rng),
            pedestrian_delay_s=self.pedestrian_delay_s.sample(rng),
            scenario_seed=rng.randint(int(self.scenario_seed.min), int(self.scenario_seed.max)),
        )

    def validate_candidate(self, candidate: CandidateSpec) -> list[str]:
        """Return validation errors for a candidate; empty means valid."""
        errors: list[str] = []
        values = {
            "start.x": candidate.start.x,
            "start.y": candidate.start.y,
            "goal.x": candidate.goal.x,
            "goal.y": candidate.goal.y,
            "spawn_time_s": candidate.spawn_time_s,
            "pedestrian_speed_mps": candidate.pedestrian_speed_mps,
            "pedestrian_delay_s": candidate.pedestrian_delay_s,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)):
                errors.append(f"{name} must be finite")
        if not self.start_x.contains(candidate.start.x):
            errors.append("start.x outside search space")
        if not self.start_y.contains(candidate.start.y):
            errors.append("start.y outside search space")
        if not self.goal_x.contains(candidate.goal.x):
            errors.append("goal.x outside search space")
        if not self.goal_y.contains(candidate.goal.y):
            errors.append("goal.y outside search space")
        if not self.spawn_time_s.contains(candidate.spawn_time_s):
            errors.append("spawn_time_s outside search space")
        if not self.pedestrian_speed_mps.contains(candidate.pedestrian_speed_mps):
            errors.append("pedestrian_speed_mps outside search space")
        if not self.pedestrian_delay_s.contains(candidate.pedestrian_delay_s):
            errors.append("pedestrian_delay_s outside search space")
        if not self.scenario_seed.contains(float(candidate.scenario_seed)):
            errors.append("scenario_seed outside search space")
        if candidate.spawn_time_s < 0.0:
            errors.append("spawn_time_s must be non-negative")
        if candidate.pedestrian_speed_mps <= 0.0:
            errors.append("pedestrian_speed_mps must be positive")
        if candidate.pedestrian_delay_s < 0.0:
            errors.append("pedestrian_delay_s must be non-negative")
        if candidate.scenario_seed < 0:
            errors.append("scenario_seed must be non-negative")
        dx = float(candidate.goal.x) - float(candidate.start.x)
        dy = float(candidate.goal.y) - float(candidate.start.y)
        if math.hypot(dx, dy) < self.min_start_goal_distance_m:
            errors.append("start and goal are closer than min_start_goal_distance_m")
        return errors

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable search-space payload."""
        return {
            "variables": {
                "start_x": self.start_x.to_json(),
                "start_y": self.start_y.to_json(),
                "goal_x": self.goal_x.to_json(),
                "goal_y": self.goal_y.to_json(),
                "spawn_time_s": self.spawn_time_s.to_json(),
                "pedestrian_speed_mps": self.pedestrian_speed_mps.to_json(),
                "pedestrian_delay_s": self.pedestrian_delay_s.to_json(),
                "scenario_seed": self.scenario_seed.to_json(),
            },
            "constraints": {"min_start_goal_distance_m": self.min_start_goal_distance_m},
            "pedestrian": {"id": self.pedestrian_id} if self.pedestrian_id else {},
        }


@dataclass(frozen=True)
class SearchConfig:
    """Top-level adversarial-search run configuration."""

    policy: str
    scenario_template: Path
    search_space_path: Path
    search_space: SearchSpaceConfig
    objective: str
    output_dir: Path
    budget: int = 64
    seed: int = 0
    algo_config_path: Path | None = None
    horizon: int | None = None
    dt: float | None = None
    workers: int = 1
    record_forces: bool = True
    require_certification: bool = False
    benchmark_profile: str = "baseline-safe"
    snqi_weights_path: Path | None = None
    snqi_baseline_path: Path | None = None

    @classmethod
    def from_files(
        cls,
        *,
        policy: str,
        scenario_template: Path,
        search_space: Path,
        objective: str,
        output_dir: Path,
        budget: int = 64,
        seed: int = 0,
        algo_config_path: Path | None = None,
        horizon: int | None = None,
        dt: float | None = None,
        workers: int = 1,
        record_forces: bool = True,
        require_certification: bool = False,
        benchmark_profile: str = "baseline-safe",
        snqi_weights_path: Path | None = None,
        snqi_baseline_path: Path | None = None,
    ) -> SearchConfig:
        """Create a search config by loading the search-space YAML."""
        return cls(
            policy=policy,
            scenario_template=Path(scenario_template),
            search_space_path=Path(search_space),
            search_space=SearchSpaceConfig.from_file(search_space),
            objective=objective,
            output_dir=Path(output_dir),
            budget=int(budget),
            seed=int(seed),
            algo_config_path=Path(algo_config_path) if algo_config_path else None,
            horizon=horizon,
            dt=dt,
            workers=int(workers),
            record_forces=bool(record_forces),
            require_certification=bool(require_certification),
            benchmark_profile=benchmark_profile,
            snqi_weights_path=Path(snqi_weights_path) if snqi_weights_path else None,
            snqi_baseline_path=Path(snqi_baseline_path) if snqi_baseline_path else None,
        )

    def validate(self) -> None:
        """Validate top-level search settings."""
        if not self.policy.strip():
            raise ValueError("policy must be non-empty")
        if self.budget < 1:
            raise ValueError("budget must be >= 1")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if not self.scenario_template.exists():
            raise FileNotFoundError(f"Scenario template not found: {self.scenario_template}")
        if self.algo_config_path is not None and not self.algo_config_path.exists():
            raise FileNotFoundError(f"Algorithm config not found: {self.algo_config_path}")

    def load_optional_json(self, path: Path | None) -> dict[str, Any] | None:
        """Load an optional JSON config file."""
        if path is None:
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable config payload for manifests."""
        return {
            "policy": self.policy,
            "scenario_template": self.scenario_template.as_posix(),
            "search_space_path": self.search_space_path.as_posix(),
            "search_space": self.search_space.to_json(),
            "objective": self.objective,
            "output_dir": self.output_dir.as_posix(),
            "budget": int(self.budget),
            "seed": int(self.seed),
            "algo_config_path": self.algo_config_path.as_posix() if self.algo_config_path else None,
            "horizon": self.horizon,
            "dt": self.dt,
            "workers": int(self.workers),
            "record_forces": bool(self.record_forces),
            "require_certification": bool(self.require_certification),
            "benchmark_profile": self.benchmark_profile,
            "snqi_weights_path": self.snqi_weights_path.as_posix()
            if self.snqi_weights_path
            else None,
            "snqi_baseline_path": self.snqi_baseline_path.as_posix()
            if self.snqi_baseline_path
            else None,
        }
