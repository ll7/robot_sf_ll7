"""Preflight helpers for sustained-flow continuous-spawn scenario slices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from robot_sf.errors import RobotSfError
from robot_sf.scenario_certification.sustained_flow import (
    EXPECTED_CONTINUOUS_SPAWN_DEFINITION,
    EXPECTED_ROUTE_RESPAWN_RUNTIME_CONFIG,
    EXPECTED_SUSTAINED_FLOW_TIERS,
    EXPECTED_SUSTAINED_PROGRESS_METRIC,
    REQUIRED_BLOCKERS_BEFORE_BENCHMARK_USE,
    SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE,
    generate_expected_sustained_flow_scenarios,
    runtime_definition_status_for_support,
)
from robot_sf.training.scenario_loader import load_scenarios

PREFLIGHT_SCHEMA_VERSION = "sustained_flow_preflight.v1"
SUSTAINED_FLOW_ARCHETYPE = "sustained_flow_t_intersection"
SUPPORTED_SPAWN_PROCESS = "poisson_respawn"
RUNTIME_SUPPORTED_VALUE = SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE


@dataclass(frozen=True, slots=True)
class SustainedFlowRuntimeReadiness:
    """Structured runtime-readiness verdict for sustained-flow candidates."""

    status: str
    expected_runtime_support: str
    observed_runtime_support: tuple[str, ...]

    @property
    def supported(self) -> bool:
        """Whether every variant advertises real continuous-spawn runtime support."""

        return self.status == "supported"

    def to_payload(self) -> dict[str, Any]:
        """Return JSON/YAML friendly representation."""

        return {
            "status": self.status,
            "supported": self.supported,
            "expected_runtime_support": self.expected_runtime_support,
            "observed_runtime_support": list(self.observed_runtime_support),
        }


@dataclass(frozen=True, slots=True)
class SustainedFlowVariant:
    """Resolved continuous-spawn variant from a scenario matrix row."""

    name: str
    density_tier: str
    ped_density: float
    spawn_rate_per_min: float
    target_density_tier: str
    spawn_definition: dict[str, object]
    current_runtime_support: str
    runtime_definition_status: str
    runtime_definition_ready: bool
    route_respawn_runtime_config: dict[str, object]
    success_metric: dict[str, object]
    required_before_benchmark_use: tuple[str, ...]
    max_episode_steps: int
    seeds: tuple[int, ...]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON/YAML friendly representation."""

        payload = asdict(self)
        payload["required_before_benchmark_use"] = list(self.required_before_benchmark_use)
        payload["seeds"] = list(self.seeds)
        return payload


@dataclass(frozen=True, slots=True)
class SustainedFlowPreflight:
    """Fail-closed preflight summary for sustained-flow benchmark candidates."""

    schema_version: str
    status: str
    matrix_path: str
    variant_count: int
    variants: tuple[SustainedFlowVariant, ...]
    runtime_readiness: SustainedFlowRuntimeReadiness
    blocking_reasons: tuple[str, ...]

    @property
    def benchmark_eligible(self) -> bool:
        """Whether the matrix may be interpreted as benchmark evidence."""

        return self.status == "available" and self.runtime_readiness.supported

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON/YAML friendly representation."""

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "benchmark_eligible": self.benchmark_eligible,
            "matrix_path": self.matrix_path,
            "variant_count": self.variant_count,
            "runtime_readiness": self.runtime_readiness.to_payload(),
            "variants": [variant.to_payload() for variant in self.variants],
            "blocking_reasons": list(self.blocking_reasons),
        }


class SustainedFlowPreflightError(RobotSfError, ValueError):
    """Raised when a sustained-flow matrix is malformed enough to fail."""


def preflight_sustained_flow_matrix(matrix_path: str | Path) -> SustainedFlowPreflight:
    """Enumerate sustained-flow variants and classify benchmark eligibility.

    The first #3813 slice is intentionally metadata-only: these rows are useful
    for scenario-generator enumeration and review, but they must fail closed for
    benchmark attribution until continuous-spawn runtime support exists.

    Returns:
        Structured preflight status, enumerated variants, and blocking reasons.
    """

    path = Path(matrix_path)
    scenarios = load_scenarios(path)
    blocking_reasons: list[str] = []
    try:
        variants = tuple(_variant_from_scenario(scenario) for scenario in scenarios)
    except SustainedFlowPreflightError as exc:
        variants = ()
        blocking_reasons.append(str(exc))
    else:
        blocking_reasons.extend(_variant_blockers(variants))
        blocking_reasons.extend(_variant_tier_order_blockers(variants))
        blocking_reasons.extend(_variant_name_tier_blockers(variants))
        blocking_reasons.extend(_generator_drift_blockers(variants))

    if not variants:
        blocking_reasons.append("no sustained-flow variants were enumerated")
    if variants and not _strictly_increasing_spawn_rates(variants):
        blocking_reasons.append("spawn_rate_per_min must increase across density tiers")
    if variants and not _strictly_increasing_ped_density(variants):
        blocking_reasons.append("ped_density must increase across density tiers")
    runtime_readiness = _runtime_readiness(variants)

    status = "available" if not blocking_reasons else "not_available"
    return SustainedFlowPreflight(
        schema_version=PREFLIGHT_SCHEMA_VERSION,
        status=status,
        matrix_path=path.as_posix(),
        variant_count=len(variants),
        variants=variants,
        runtime_readiness=runtime_readiness,
        blocking_reasons=tuple(blocking_reasons),
    )


def _variant_from_scenario(scenario: Mapping[str, Any]) -> SustainedFlowVariant:
    name = _required_str(scenario, "name")
    metadata = _required_mapping(scenario, "metadata", scenario_name=name)
    continuous_spawn = _required_mapping(metadata, "continuous_spawn", scenario_name=name)
    simulation_config = _required_mapping(scenario, "simulation_config", scenario_name=name)

    if metadata.get("archetype") != SUSTAINED_FLOW_ARCHETYPE:
        raise SustainedFlowPreflightError(
            f"{name}: metadata.archetype must be {SUSTAINED_FLOW_ARCHETYPE!r}"
        )
    if metadata.get("flow") != "continuous_crossing":
        raise SustainedFlowPreflightError(f"{name}: metadata.flow must be 'continuous_crossing'")
    if continuous_spawn.get("intended_process") != SUPPORTED_SPAWN_PROCESS:
        raise SustainedFlowPreflightError(
            f"{name}: continuous_spawn.intended_process must be {SUPPORTED_SPAWN_PROCESS!r}"
        )
    _validate_continuous_spawn_definition(
        continuous_spawn.get("definition"),
        scenario_name=name,
    )

    success_metric = _required_mapping(metadata, "success_metric", scenario_name=name)
    if success_metric != EXPECTED_SUSTAINED_PROGRESS_METRIC:
        raise SustainedFlowPreflightError(
            f"{name}: metadata.success_metric must match sustained progress-rate contract"
        )
    required_before_benchmark_use = metadata.get("requires_before_benchmark_use")
    if not isinstance(required_before_benchmark_use, list) or required_before_benchmark_use != list(
        REQUIRED_BLOCKERS_BEFORE_BENCHMARK_USE
    ):
        raise SustainedFlowPreflightError(
            f"{name}: metadata.requires_before_benchmark_use must match sustained-flow "
            "fail-closed blockers"
        )

    seeds = scenario.get("seeds", [])
    if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
        raise SustainedFlowPreflightError(f"{name}: seeds must be a list of integers")

    return SustainedFlowVariant(
        name=name,
        density_tier=_required_str(metadata, "density", scenario_name=name),
        ped_density=_required_float(simulation_config, "ped_density", scenario_name=name),
        spawn_rate_per_min=_required_float(
            continuous_spawn, "spawn_rate_per_min", scenario_name=name
        ),
        target_density_tier=_required_str(
            continuous_spawn, "target_density_tier", scenario_name=name
        ),
        spawn_definition=dict(EXPECTED_CONTINUOUS_SPAWN_DEFINITION),
        current_runtime_support=_required_str(
            continuous_spawn, "current_runtime_support", scenario_name=name
        ),
        runtime_definition_status=_required_str(
            continuous_spawn, "runtime_definition_status", scenario_name=name
        ),
        runtime_definition_ready=_required_bool(
            continuous_spawn, "runtime_definition_ready", scenario_name=name
        ),
        route_respawn_runtime_config={
            field: simulation_config.get(field) for field in EXPECTED_ROUTE_RESPAWN_RUNTIME_CONFIG
        },
        success_metric=dict(success_metric),
        required_before_benchmark_use=tuple(required_before_benchmark_use),
        max_episode_steps=_required_int(simulation_config, "max_episode_steps", scenario_name=name),
        seeds=tuple(seeds),
    )


def _variant_blockers(variants: tuple[SustainedFlowVariant, ...]) -> tuple[str, ...]:
    blockers: list[str] = []
    for variant in variants:
        if variant.target_density_tier != variant.density_tier:
            blockers.append(
                f"{variant.name}: continuous_spawn.target_density_tier "
                f"{variant.target_density_tier!r}, expected {variant.density_tier!r}"
            )
        expected_definition_status, expected_definition_ready = (
            runtime_definition_status_for_support(variant.current_runtime_support)
        )
        if variant.runtime_definition_status != expected_definition_status:
            blockers.append(
                f"{variant.name}: continuous-spawn runtime definition status "
                f"{variant.runtime_definition_status!r}, expected "
                f"{expected_definition_status!r}"
            )
        if variant.runtime_definition_ready is not expected_definition_ready:
            blockers.append(
                f"{variant.name}: continuous-spawn runtime definition readiness "
                f"{variant.runtime_definition_ready!r}, expected "
                f"{expected_definition_ready!r}"
            )
        if variant.current_runtime_support != RUNTIME_SUPPORTED_VALUE:
            blockers.append(
                f"{variant.name}: continuous-spawn runtime support is "
                f"{variant.current_runtime_support!r}, expected {RUNTIME_SUPPORTED_VALUE!r}"
            )
        if variant.route_respawn_runtime_config != EXPECTED_ROUTE_RESPAWN_RUNTIME_CONFIG:
            blockers.append(
                f"{variant.name}: route-respawn runtime config "
                f"{variant.route_respawn_runtime_config!r}, expected "
                f"{EXPECTED_ROUTE_RESPAWN_RUNTIME_CONFIG!r}"
            )
        if variant.max_episode_steps <= 0:
            blockers.append(f"{variant.name}: max_episode_steps must be positive")
        if not variant.seeds:
            blockers.append(f"{variant.name}: at least one seed is required")
    return tuple(blockers)


def _variant_tier_order_blockers(variants: tuple[SustainedFlowVariant, ...]) -> tuple[str, ...]:
    """Fail closed unless the matrix enumerates exactly light, medium, heavy.

    Returns:
        Blocking reason when density-tier enumeration drifts.
    """

    expected_tiers = tuple(tier for tier, *_ in EXPECTED_SUSTAINED_FLOW_TIERS)
    observed_tiers = tuple(variant.density_tier for variant in variants)
    if observed_tiers == expected_tiers:
        return ()
    return ("density tiers must enumerate light, medium, heavy exactly once",)


def _variant_name_tier_blockers(variants: tuple[SustainedFlowVariant, ...]) -> tuple[str, ...]:
    """Fail closed when canonical sustained-flow names drift from their density tier.

    Returns:
        Blocking reasons for variant names that do not match their density tier.
    """

    expected_name_by_tier = {
        tier: f"issue_3813_{SUSTAINED_FLOW_ARCHETYPE}_{tier}"
        for tier, *_ in EXPECTED_SUSTAINED_FLOW_TIERS
    }
    blockers = []
    for variant in variants:
        expected_name = expected_name_by_tier.get(variant.density_tier)
        if expected_name is None:
            continue
        if variant.name != expected_name:
            blockers.append(
                f"{variant.name}: scenario name must be {expected_name!r} "
                f"for density tier {variant.density_tier!r}"
            )
    return tuple(blockers)


def _validate_continuous_spawn_definition(value: Any, *, scenario_name: str) -> None:
    if not isinstance(value, Mapping):
        raise SustainedFlowPreflightError(
            f"{scenario_name}: continuous_spawn.definition must be a mapping"
        )
    missing_or_extra_keys = set(value) ^ set(EXPECTED_CONTINUOUS_SPAWN_DEFINITION)
    if missing_or_extra_keys:
        raise SustainedFlowPreflightError(
            f"{scenario_name}: continuous_spawn.definition keys must match "
            "non-clearing demand contract "
            f"(mismatched keys: {sorted(missing_or_extra_keys)})"
        )
    for key, expected in EXPECTED_CONTINUOUS_SPAWN_DEFINITION.items():
        if value.get(key) != expected:
            raise SustainedFlowPreflightError(
                f"{scenario_name}: continuous_spawn.definition.{key} must be {expected!r}"
            )


def _generator_drift_blockers(variants: tuple[SustainedFlowVariant, ...]) -> tuple[str, ...]:
    """Fail closed when the checked-in matrix drifts from the canonical generator.

    Returns:
        Blocking reasons when generator-owned variant fields drift.
    """
    observed_runtime_support = tuple(
        dict.fromkeys(variant.current_runtime_support for variant in variants)
    )
    expected_runtime_support = (
        observed_runtime_support[0]
        if len(observed_runtime_support) == 1
        else RUNTIME_SUPPORTED_VALUE
    )
    generated_variants = tuple(
        _variant_from_scenario(row)
        for row in generate_expected_sustained_flow_scenarios(
            current_runtime_support=expected_runtime_support
        )
    )
    observed_profile = tuple(_variant_generator_profile(variant) for variant in variants)
    expected_profile = tuple(_variant_generator_profile(variant) for variant in generated_variants)
    if observed_profile == expected_profile:
        return ()
    return ("sustained-flow matrix must match canonical generated variant definitions",)


def _variant_generator_profile(variant: SustainedFlowVariant) -> tuple[object, ...]:
    return (
        variant.name,
        variant.density_tier,
        variant.ped_density,
        variant.spawn_rate_per_min,
        variant.target_density_tier,
        tuple(sorted(variant.spawn_definition.items())),
        variant.current_runtime_support,
        variant.runtime_definition_status,
        variant.runtime_definition_ready,
        tuple(sorted(variant.route_respawn_runtime_config.items())),
        tuple(sorted(variant.success_metric.items())),
        variant.required_before_benchmark_use,
        variant.max_episode_steps,
        variant.seeds,
    )


def _runtime_readiness(
    variants: tuple[SustainedFlowVariant, ...],
) -> SustainedFlowRuntimeReadiness:
    observed = tuple(dict.fromkeys(variant.current_runtime_support for variant in variants))
    status = "supported" if variants and observed == (RUNTIME_SUPPORTED_VALUE,) else "not_supported"
    return SustainedFlowRuntimeReadiness(
        status=status,
        expected_runtime_support=RUNTIME_SUPPORTED_VALUE,
        observed_runtime_support=observed,
    )


def _strictly_increasing_spawn_rates(variants: tuple[SustainedFlowVariant, ...]) -> bool:
    rates = [variant.spawn_rate_per_min for variant in variants]
    return all(left < right for left, right in pairwise(rates))


def _strictly_increasing_ped_density(variants: tuple[SustainedFlowVariant, ...]) -> bool:
    densities = [variant.ped_density for variant in variants]
    return all(left < right for left, right in pairwise(densities))


def _required_mapping(
    payload: Mapping[str, Any], key: str, *, scenario_name: str
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise SustainedFlowPreflightError(f"{scenario_name}: {key} must be a mapping")
    return value


def _required_str(payload: Mapping[str, Any], key: str, *, scenario_name: str = "scenario") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SustainedFlowPreflightError(f"{scenario_name}: {key} must be a non-empty string")
    return value


def _required_float(payload: Mapping[str, Any], key: str, *, scenario_name: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise SustainedFlowPreflightError(f"{scenario_name}: {key} must be numeric")
    return float(value)


def _required_int(payload: Mapping[str, Any], key: str, *, scenario_name: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SustainedFlowPreflightError(f"{scenario_name}: {key} must be an integer")
    return value


def _required_bool(payload: Mapping[str, Any], key: str, *, scenario_name: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise SustainedFlowPreflightError(f"{scenario_name}: {key} must be a boolean")
    return value


__all__ = [
    "PREFLIGHT_SCHEMA_VERSION",
    "RUNTIME_SUPPORTED_VALUE",
    "SustainedFlowPreflight",
    "SustainedFlowPreflightError",
    "SustainedFlowRuntimeReadiness",
    "SustainedFlowVariant",
    "preflight_sustained_flow_matrix",
]
