"""Validated multi-map training protocol and declarative randomization helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


MULTI_MAP_PROTOCOL_SCHEMA_VERSION = "multi-map-train-test.v1"
DOMAIN_RANDOMIZATION_SCHEMA_VERSION = "training-domain-randomization.v1"
SUPPORTED_ZERO_SHOT_DECAY_METRICS = frozenset({"success_rate", "snqi", "path_efficiency"})


def _names(raw: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ValueError(f"{field_name} must be a non-empty list of scenario IDs")
    names = tuple(str(value).strip() for value in raw)
    if not names or any(not value for value in names):
        raise ValueError(f"{field_name} must be a non-empty list of scenario IDs")
    if len({value.lower() for value in names}) != len(names):
        raise ValueError(f"{field_name} must not contain duplicate scenario IDs")
    return names


def _range(raw: object, *, field_name: str, lower_bound: float = 0.0) -> tuple[float, float]:
    if isinstance(raw, str) or not isinstance(raw, Sequence) or len(raw) != 2:
        raise ValueError(f"{field_name} must be a two-item numeric range")
    lower, upper = (float(raw[0]), float(raw[1]))
    if not all(isfinite(value) for value in (lower, upper)) or lower < lower_bound or upper < lower:
        raise ValueError(f"{field_name} must be finite with {lower_bound} <= lower <= upper")
    return lower, upper


@dataclass(frozen=True, slots=True)
class DomainRandomization:
    """Training-only randomization sampled independently for each new scenario env."""

    peds_speed_mult: tuple[float, float]
    ped_density_multiplier: tuple[float, float]
    route_spawn_jitter_frac: tuple[float, float]

    @classmethod
    def from_raw(cls, raw: object) -> DomainRandomization | None:
        """Parse an optional declarative domain-randomization profile.

        Returns:
            Parsed profile, or None when the config does not declare one.
        """
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("domain_randomization must be a mapping")
        if raw.get("schema_version") != DOMAIN_RANDOMIZATION_SCHEMA_VERSION:
            raise ValueError(
                "domain_randomization.schema_version must be "
                f"{DOMAIN_RANDOMIZATION_SCHEMA_VERSION!r}"
            )
        return cls(
            peds_speed_mult=_range(raw.get("peds_speed_mult"), field_name="peds_speed_mult"),
            ped_density_multiplier=_range(
                raw.get("ped_density_multiplier"), field_name="ped_density_multiplier"
            ),
            route_spawn_jitter_frac=_range(
                raw.get("route_spawn_jitter_frac"), field_name="route_spawn_jitter_frac"
            ),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a stable manifest-friendly representation."""
        return {
            "schema_version": DOMAIN_RANDOMIZATION_SCHEMA_VERSION,
            "peds_speed_mult": list(self.peds_speed_mult),
            "ped_density_multiplier": list(self.ped_density_multiplier),
            "route_spawn_jitter_frac": list(self.route_spawn_jitter_frac),
        }


@dataclass(frozen=True, slots=True)
class MultiMapTrainTestProtocol:
    """Immutable train/held-out split and zero-shot metric contract for PPO."""

    train_scenarios: tuple[str, ...]
    held_out_scenarios: tuple[str, ...]
    zero_shot_decay_metric: str

    @classmethod
    def from_raw(cls, raw: object) -> MultiMapTrainTestProtocol | None:
        """Parse an optional multi-map protocol without inspecting scenario files.

        Returns:
            Parsed protocol, or None when the config does not declare one.
        """
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("multi_map_protocol must be a mapping")
        if raw.get("schema_version") != MULTI_MAP_PROTOCOL_SCHEMA_VERSION:
            raise ValueError(
                f"multi_map_protocol.schema_version must be {MULTI_MAP_PROTOCOL_SCHEMA_VERSION!r}"
            )
        train_scenarios = _names(raw.get("train_scenarios"), field_name="train_scenarios")
        held_out_scenarios = _names(raw.get("held_out_scenarios"), field_name="held_out_scenarios")
        overlap = {name.lower() for name in train_scenarios} & {
            name.lower() for name in held_out_scenarios
        }
        if overlap:
            raise ValueError(f"train_scenarios and held_out_scenarios overlap: {sorted(overlap)}")
        metric = str(raw.get("zero_shot_decay_metric", "")).strip()
        if metric not in SUPPORTED_ZERO_SHOT_DECAY_METRICS:
            supported = ", ".join(sorted(SUPPORTED_ZERO_SHOT_DECAY_METRICS))
            raise ValueError(f"zero_shot_decay_metric must be one of: {supported}")
        return cls(train_scenarios, held_out_scenarios, metric)

    def validate_scenarios(self, scenarios: Sequence[Mapping[str, Any]]) -> None:
        """Fail closed when a declared split does not resolve in the scenario manifest."""
        known = {
            str(scenario.get("name") or scenario.get("scenario_id") or "").lower()
            for scenario in scenarios
        }
        declared = self.train_scenarios + self.held_out_scenarios
        unknown = sorted(name for name in declared if name.lower() not in known)
        if unknown:
            raise ValueError(f"multi_map_protocol declares unknown scenario IDs: {unknown}")

    def as_dict(self) -> dict[str, object]:
        """Return a stable manifest-friendly representation."""
        return {
            "schema_version": MULTI_MAP_PROTOCOL_SCHEMA_VERSION,
            "train_scenarios": list(self.train_scenarios),
            "held_out_scenarios": list(self.held_out_scenarios),
            "zero_shot_decay_metric": self.zero_shot_decay_metric,
        }


def apply_domain_randomization(
    scenario: Mapping[str, Any],
    randomization: DomainRandomization | None,
    *,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Return a per-episode randomized scenario and the sampled values.

    The profile changes only values owned by ``simulation_config``. A zero-density
    marker-based scenario remains zero-density, preserving its declared spawn mode.
    """
    if randomization is None:
        return deepcopy(dict(scenario)), {}
    randomized = deepcopy(dict(scenario))
    simulation = randomized.setdefault("simulation_config", {})
    if not isinstance(simulation, dict):
        raise ValueError("scenario simulation_config must be a mapping for domain randomization")
    peds_speed_mult = float(rng.uniform(*randomization.peds_speed_mult))
    density_multiplier = float(rng.uniform(*randomization.ped_density_multiplier))
    route_spawn_jitter_frac = float(rng.uniform(*randomization.route_spawn_jitter_frac))
    simulation["peds_speed_mult"] = peds_speed_mult
    if "ped_density" in simulation:
        simulation["ped_density"] = float(simulation["ped_density"]) * density_multiplier
    simulation["route_spawn_jitter_frac"] = route_spawn_jitter_frac
    return randomized, {
        "peds_speed_mult": peds_speed_mult,
        "ped_density_multiplier": density_multiplier,
        "route_spawn_jitter_frac": route_spawn_jitter_frac,
    }
