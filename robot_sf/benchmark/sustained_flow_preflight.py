"""Preflight helpers for sustained-flow continuous-spawn scenario slices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from robot_sf.training.scenario_loader import load_scenarios

PREFLIGHT_SCHEMA_VERSION = "sustained_flow_preflight.v1"
SUSTAINED_FLOW_ARCHETYPE = "sustained_flow_t_intersection"
SUPPORTED_SPAWN_PROCESS = "poisson_respawn"
RUNTIME_SUPPORTED_VALUE = "runtime_continuous_spawn"


@dataclass(frozen=True, slots=True)
class SustainedFlowVariant:
    """Resolved continuous-spawn variant from a scenario matrix row."""

    name: str
    density_tier: str
    ped_density: float
    spawn_rate_per_min: float
    current_runtime_support: str
    max_episode_steps: int
    seeds: tuple[int, ...]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON/YAML friendly representation."""

        payload = asdict(self)
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
    blocking_reasons: tuple[str, ...]

    @property
    def benchmark_eligible(self) -> bool:
        """Whether the matrix may be interpreted as benchmark evidence."""

        return self.status == "available"

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON/YAML friendly representation."""

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "benchmark_eligible": self.benchmark_eligible,
            "matrix_path": self.matrix_path,
            "variant_count": self.variant_count,
            "variants": [variant.to_payload() for variant in self.variants],
            "blocking_reasons": list(self.blocking_reasons),
        }


class SustainedFlowPreflightError(ValueError):
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
    variants = tuple(_variant_from_scenario(scenario) for scenario in scenarios)
    blocking_reasons = list(_variant_blockers(variants))

    if not variants:
        blocking_reasons.append("no sustained-flow variants were enumerated")
    if variants and not _strictly_increasing_spawn_rates(variants):
        blocking_reasons.append("spawn_rate_per_min must increase across density tiers")
    if variants and not _strictly_increasing_ped_density(variants):
        blocking_reasons.append("ped_density must increase across density tiers")

    status = "available" if not blocking_reasons else "not_available"
    return SustainedFlowPreflight(
        schema_version=PREFLIGHT_SCHEMA_VERSION,
        status=status,
        matrix_path=path.as_posix(),
        variant_count=len(variants),
        variants=variants,
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
        current_runtime_support=_required_str(
            continuous_spawn, "current_runtime_support", scenario_name=name
        ),
        max_episode_steps=_required_int(simulation_config, "max_episode_steps", scenario_name=name),
        seeds=tuple(seeds),
    )


def _variant_blockers(variants: tuple[SustainedFlowVariant, ...]) -> tuple[str, ...]:
    blockers: list[str] = []
    for variant in variants:
        if variant.current_runtime_support != RUNTIME_SUPPORTED_VALUE:
            blockers.append(
                f"{variant.name}: continuous-spawn runtime support is "
                f"{variant.current_runtime_support!r}, expected {RUNTIME_SUPPORTED_VALUE!r}"
            )
        if variant.max_episode_steps <= 0:
            blockers.append(f"{variant.name}: max_episode_steps must be positive")
        if not variant.seeds:
            blockers.append(f"{variant.name}: at least one seed is required")
    return tuple(blockers)


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


__all__ = [
    "PREFLIGHT_SCHEMA_VERSION",
    "RUNTIME_SUPPORTED_VALUE",
    "SustainedFlowPreflight",
    "SustainedFlowPreflightError",
    "SustainedFlowVariant",
    "preflight_sustained_flow_matrix",
]
