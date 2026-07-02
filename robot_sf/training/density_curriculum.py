"""Deterministic pedestrian-density curriculum utilities for training configs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

DENSITY_CURRICULUM_SCHEMA_VERSION = "density_curriculum.v1"
CURRICULUM_CLAIM_BOUNDARY = "training curriculum mechanism only; not benchmark evidence"


@dataclass(frozen=True, slots=True)
class DensityCurriculumStage:
    """One timestep-gated density curriculum stage."""

    id: str
    until_timesteps: int | None
    density_m2: float | None = None
    difficulty: int | None = None
    max_peds_per_group: int | None = None
    include_scenarios: tuple[str, ...] = ()
    exclude_scenarios: tuple[str, ...] = ()
    scenario_weights: dict[str, float] = field(default_factory=dict)
    parameterized_profile: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DensityCurriculumSchedule:
    """Validated deterministic density curriculum schedule."""

    enabled: bool
    stages: tuple[DensityCurriculumStage, ...] = ()
    advance_rule: str = "timestep"
    enforce_nondecreasing_density: bool = True
    schema_version: str = DENSITY_CURRICULUM_SCHEMA_VERSION
    claim_boundary: str = CURRICULUM_CLAIM_BOUNDARY

    def stage_for_timestep(self, timestep: int) -> DensityCurriculumStage | None:
        """Return the active stage at a global PPO timestep."""
        if not self.enabled:
            return None
        if timestep < 0:
            raise ValueError("density_curriculum timestep must be non-negative.")
        for stage in self.stages:
            if stage.until_timesteps is None or timestep < stage.until_timesteps:
                return stage
        return self.stages[-1]


def build_density_curriculum_schedule(
    payload: Mapping[str, Any] | None,
) -> DensityCurriculumSchedule:
    """Build and validate a density curriculum schedule from YAML payload.

    Returns:
        Validated schedule with disabled empty schedules represented explicitly.
    """
    if not payload:
        return DensityCurriculumSchedule(enabled=False)
    if not isinstance(payload, Mapping):
        raise ValueError("density_curriculum must be a mapping.")

    enabled = bool(payload.get("enabled", False))
    advance_rule = str(payload.get("advance_rule", "timestep"))
    if advance_rule != "timestep":
        raise ValueError("density_curriculum.advance_rule must be 'timestep'.")

    raw_stages = payload.get("stages", ())
    if not isinstance(raw_stages, list | tuple):
        raise ValueError("density_curriculum.stages must be a list.")
    if enabled and not raw_stages:
        raise ValueError("enabled density_curriculum requires at least one stage.")

    stages = tuple(_parse_stage(stage, index) for index, stage in enumerate(raw_stages))
    enforce = bool(payload.get("enforce_nondecreasing_density", True))
    _validate_stages(stages, enforce_nondecreasing_density=enforce)
    return DensityCurriculumSchedule(
        enabled=enabled,
        stages=stages,
        advance_rule=advance_rule,
        enforce_nondecreasing_density=enforce,
    )


def apply_density_curriculum_stage_to_scenario(
    scenario: Mapping[str, Any],
    stage: DensityCurriculumStage | None,
) -> dict[str, Any]:
    """Apply a stage to a scenario mapping using existing simulation config fields.

    Returns:
        Scenario copy with stage-specific simulation settings applied.
    """
    updated = _deep_copy_mapping(scenario)
    if stage is None:
        return updated

    sim_config = dict(updated.get("simulation_config") or updated.get("sim_config") or {})
    if stage.difficulty is not None:
        sim_config["difficulty"] = int(stage.difficulty)
    if stage.density_m2 is not None:
        sim_config["ped_density_by_difficulty"] = [float(stage.density_m2)]
        sim_config.setdefault("difficulty", 0)
    if stage.max_peds_per_group is not None:
        sim_config["max_peds_per_group"] = int(stage.max_peds_per_group)
    updated["simulation_config"] = sim_config
    updated.pop("sim_config", None)

    if stage.parameterized_profile is not None:
        updated["parameterized_profile"] = dict(stage.parameterized_profile)
    return updated


def stage_metadata(stage: DensityCurriculumStage | None) -> dict[str, Any]:
    """Return compact metadata for logs and manifests."""
    if stage is None:
        return {
            "schema_version": DENSITY_CURRICULUM_SCHEMA_VERSION,
            "enabled": False,
            "claim_boundary": CURRICULUM_CLAIM_BOUNDARY,
        }
    return {
        "schema_version": DENSITY_CURRICULUM_SCHEMA_VERSION,
        "enabled": True,
        "stage_id": stage.id,
        "until_timesteps": stage.until_timesteps,
        "density_m2": stage.density_m2,
        "difficulty": stage.difficulty,
        "max_peds_per_group": stage.max_peds_per_group,
        "claim_boundary": CURRICULUM_CLAIM_BOUNDARY,
    }


def curriculum_metadata(schedule: DensityCurriculumSchedule) -> dict[str, Any]:
    """Return schedule metadata without implying training-result evidence."""
    return {
        "schema_version": schedule.schema_version,
        "enabled": schedule.enabled,
        "advance_rule": schedule.advance_rule,
        "stage_count": len(schedule.stages),
        "stages": [stage_metadata(stage) for stage in schedule.stages],
        "claim_boundary": schedule.claim_boundary,
    }


def _parse_stage(raw: object, index: int) -> DensityCurriculumStage:
    if not isinstance(raw, Mapping):
        raise ValueError("density_curriculum.stages entries must be mappings.")
    stage_id = str(raw.get("id") or "").strip()
    if not stage_id:
        raise ValueError(f"density_curriculum stage {index} missing id.")
    until_raw = raw.get("until_timesteps")
    until_timesteps = int(until_raw) if until_raw is not None else None
    if until_timesteps is not None and until_timesteps <= 0:
        raise ValueError("density_curriculum until_timesteps must be positive or null.")

    density_m2 = _optional_float(raw.get("density_m2"), "density_m2")
    difficulty = _optional_int(raw.get("difficulty"), "difficulty")
    max_peds_per_group = _optional_int(raw.get("max_peds_per_group"), "max_peds_per_group")

    scenario_weights_raw = raw.get("scenario_weights") or {}
    if not isinstance(scenario_weights_raw, Mapping):
        raise ValueError("density_curriculum scenario_weights must be a mapping.")
    scenario_weights = {str(key): float(value) for key, value in scenario_weights_raw.items()}
    if any(weight < 0.0 or not isfinite(weight) for weight in scenario_weights.values()):
        raise ValueError("density_curriculum scenario_weights must be finite non-negative values.")

    profile_raw = raw.get("parameterized_profile")
    if profile_raw is not None and not isinstance(profile_raw, Mapping):
        raise ValueError("density_curriculum parameterized_profile must be a mapping.")

    return DensityCurriculumStage(
        id=stage_id,
        until_timesteps=until_timesteps,
        density_m2=density_m2,
        difficulty=difficulty,
        max_peds_per_group=max_peds_per_group,
        include_scenarios=_string_tuple(raw.get("include_scenarios", ())),
        exclude_scenarios=_string_tuple(raw.get("exclude_scenarios", ())),
        scenario_weights=scenario_weights,
        parameterized_profile=dict(profile_raw) if profile_raw is not None else None,
    )


def _validate_stages(
    stages: tuple[DensityCurriculumStage, ...],
    *,
    enforce_nondecreasing_density: bool,
) -> None:
    seen_ids: set[str] = set()
    previous_until: int | None = None
    previous_density: float | None = None
    for index, stage in enumerate(stages):
        if stage.id in seen_ids:
            raise ValueError(f"Duplicate density_curriculum stage id: {stage.id}")
        seen_ids.add(stage.id)

        if index < len(stages) - 1 and stage.until_timesteps is None:
            raise ValueError(
                "Only the final density_curriculum stage may use null until_timesteps."
            )
        if previous_until is not None and stage.until_timesteps is not None:
            if stage.until_timesteps <= previous_until:
                raise ValueError("density_curriculum until_timesteps must strictly increase.")
        if stage.until_timesteps is not None:
            previous_until = stage.until_timesteps

        if enforce_nondecreasing_density and stage.density_m2 is not None:
            if previous_density is not None and stage.density_m2 < previous_density:
                raise ValueError("density_curriculum density_m2 must be non-decreasing.")
            previous_density = stage.density_m2


def _optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed < 0.0 or not isfinite(parsed):
        raise ValueError(f"density_curriculum {field_name} must be finite and non-negative.")
    return parsed


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"density_curriculum {field_name} must be non-negative.")
    return parsed


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError("density_curriculum scenario filters must be lists.")
    return tuple(str(item) for item in value)


def _deep_copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            copied[str(key)] = _deep_copy_mapping(item)
        elif isinstance(item, list):
            copied[str(key)] = [
                _deep_copy_mapping(entry) if isinstance(entry, Mapping) else entry for entry in item
            ]
        else:
            copied[str(key)] = item
    return copied
