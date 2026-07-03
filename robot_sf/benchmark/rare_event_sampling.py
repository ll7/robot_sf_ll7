"""Importance-sampling primitives for rare-event benchmark smoke runs.

This module is deliberately planner-agnostic. It owns the sampling, likelihood-ratio bookkeeping,
pure scenario mutation, and estimator math needed by the issue #4163 first slice; benchmark runners
can consume the emitted rows without treating them as benchmark-strength failure-rate evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

SCHEMA_VERSION = "rare_event_sampling.v1"


class RareEventSamplingError(ValueError):
    """Raised when the rare-event sampling contract is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class ParameterDistribution:
    """Base and proposal distribution for one scenario perturbation knob."""

    name: str
    base: str
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None
    proposal_low: float | None = None
    proposal_high: float | None = None
    proposal_mean: float | None = None
    proposal_std: float | None = None
    proposal_shift: float = 0.0

    @classmethod
    def from_payload(cls, name: str, payload: Mapping[str, Any]) -> ParameterDistribution:
        """Build a distribution from a YAML/JSON payload.

        Returns:
            Validated parameter distribution.
        """

        if not isinstance(payload, Mapping):
            raise RareEventSamplingError(f"parameters.{name}: expected mapping")
        base = str(payload.get("base", "")).strip()
        if base not in {"uniform", "normal"}:
            raise RareEventSamplingError(
                f"parameters.{name}.base: expected 'uniform' or 'normal', got {base!r}"
            )
        dist = cls(
            name=name,
            base=base,
            low=_optional_float(payload.get("low")),
            high=_optional_float(payload.get("high")),
            mean=_optional_float(payload.get("mean")),
            std=_optional_float(payload.get("std")),
            proposal_low=_optional_float(payload.get("proposal_low")),
            proposal_high=_optional_float(payload.get("proposal_high")),
            proposal_mean=_optional_float(payload.get("proposal_mean")),
            proposal_std=_optional_float(payload.get("proposal_std")),
            proposal_shift=float(payload.get("proposal_shift", 0.0)),
        )
        dist.validate()
        return dist

    def validate(self) -> None:
        """Reject malformed distributions before sampling."""

        if self.base == "uniform":
            if self.low is None or self.high is None or not self.low < self.high:
                raise RareEventSamplingError(f"parameters.{self.name}: uniform requires low < high")
            proposal_low, proposal_high = self.proposal_bounds
            if not proposal_low < proposal_high:
                raise RareEventSamplingError(
                    f"parameters.{self.name}: proposal uniform bounds must satisfy low < high"
                )
        if self.base == "normal":
            if self.mean is None or self.std is None or self.std <= 0.0:
                raise RareEventSamplingError(f"parameters.{self.name}: normal requires std > 0")
            if self.proposal_std is not None and self.proposal_std <= 0.0:
                raise RareEventSamplingError(
                    f"parameters.{self.name}: proposal_std must be positive"
                )

    @property
    def proposal_bounds(self) -> tuple[float, float]:
        """Return the proposal support for a uniform base distribution.

        Returns:
            Inclusive lower and upper proposal bounds.
        """

        if self.low is None or self.high is None:
            raise RareEventSamplingError(f"parameters.{self.name}: uniform bounds missing")
        proposal_low = self.proposal_low
        proposal_high = self.proposal_high
        if proposal_low is None:
            proposal_low = self.low + self.proposal_shift
        if proposal_high is None:
            proposal_high = self.high + self.proposal_shift
        return proposal_low, proposal_high

    @property
    def proposal_normal(self) -> tuple[float, float]:
        """Return the proposal mean/std for a normal base distribution.

        Returns:
            Proposal mean and standard deviation.
        """

        if self.mean is None or self.std is None:
            raise RareEventSamplingError(f"parameters.{self.name}: normal parameters missing")
        proposal_mean = self.proposal_mean
        if proposal_mean is None:
            proposal_mean = self.mean + self.proposal_shift
        proposal_std = self.proposal_std if self.proposal_std is not None else self.std
        return proposal_mean, proposal_std

    def sample(self, rng: random.Random) -> float:
        """Sample a parameter value from the proposal distribution.

        Returns:
            Proposal-distributed parameter value.
        """

        if self.base == "uniform":
            low, high = self.proposal_bounds
            return rng.uniform(low, high)
        mean, std = self.proposal_normal
        return rng.gauss(mean, std)

    def base_pdf(self, value: float) -> float:
        """Evaluate the base density at a sampled value.

        Returns:
            Base probability density.
        """

        if self.base == "uniform":
            if self.low is None or self.high is None:
                raise RareEventSamplingError(f"parameters.{self.name}: uniform bounds missing")
            if value < self.low or value > self.high:
                return 0.0
            return 1.0 / (self.high - self.low)
        if self.mean is None or self.std is None:
            raise RareEventSamplingError(f"parameters.{self.name}: normal parameters missing")
        return _normal_pdf(value, self.mean, self.std)

    def proposal_pdf(self, value: float) -> float:
        """Evaluate the proposal density at a sampled value.

        Returns:
            Proposal probability density.
        """

        if self.base == "uniform":
            low, high = self.proposal_bounds
            if value < low or value > high:
                return 0.0
            return 1.0 / (high - low)
        mean, std = self.proposal_normal
        return _normal_pdf(value, mean, std)


@dataclass(frozen=True, slots=True)
class RareEventSamplingSpec:
    """Versioned rare-event sampling specification."""

    schema_version: str
    proposal: str
    parameters: tuple[ParameterDistribution, ...]
    objective_event: str
    samples: int
    seed: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RareEventSamplingSpec:
        """Parse and validate a rare-event sampling spec payload.

        Returns:
            Validated rare-event sampling spec.
        """

        if not isinstance(payload, Mapping):
            raise RareEventSamplingError("rare-event sampling spec must be a mapping")
        schema_version = str(payload.get("schema_version", "")).strip()
        if schema_version != SCHEMA_VERSION:
            raise RareEventSamplingError(
                f"schema_version: expected {SCHEMA_VERSION!r}, got {schema_version!r}"
            )
        proposal = str(payload.get("proposal", "")).strip()
        if proposal != "tilted_distribution":
            raise RareEventSamplingError(
                f"proposal: expected 'tilted_distribution', got {proposal!r}"
            )
        raw_parameters = payload.get("parameters")
        if not isinstance(raw_parameters, Mapping) or not raw_parameters:
            raise RareEventSamplingError("parameters: expected non-empty mapping")
        parameters = tuple(
            ParameterDistribution.from_payload(name, param_payload)
            for name, param_payload in sorted(raw_parameters.items())
        )
        samples = int(payload.get("samples", 0))
        if samples <= 0:
            raise RareEventSamplingError("samples must be positive")
        objective_event = str(payload.get("objective_event", "")).strip()
        if not objective_event:
            raise RareEventSamplingError("objective_event must be non-empty")
        return cls(
            schema_version=schema_version,
            proposal=proposal,
            parameters=parameters,
            objective_event=objective_event,
            samples=samples,
            seed=int(payload.get("seed", 0)),
        )


@dataclass(frozen=True, slots=True)
class SampledScenarioRow:
    """One proposal-sampled parameter vector with likelihood bookkeeping."""

    sample_index: int
    seed: int
    parameters: dict[str, float]
    base_probability: float
    proposal_probability: float
    likelihood_ratio: float
    parameter_vector_hash: str

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON-serializable row payload."""

        return {
            "sample_index": self.sample_index,
            "seed": self.seed,
            "parameters": dict(self.parameters),
            "base_probability": self.base_probability,
            "proposal_probability": self.proposal_probability,
            "likelihood_ratio": self.likelihood_ratio,
            "parameter_vector_hash": self.parameter_vector_hash,
        }


@dataclass(frozen=True, slots=True)
class ImportanceSamplingEstimate:
    """Importance-sampling estimator output for a binary rare event."""

    schema_version: str
    objective_event: str
    samples: int
    estimate: float
    standard_error: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    effective_sample_size: float
    naive_monte_carlo_estimate: float
    importance_sampling_variance: float
    naive_monte_carlo_variance: float
    variance_ratio_vs_naive: float | None
    event_count: int
    weight_sum: float

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON-serializable estimate payload."""

        return {
            "schema_version": self.schema_version,
            "objective_event": self.objective_event,
            "samples": self.samples,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "confidence_interval": list(self.confidence_interval),
            "confidence_level": self.confidence_level,
            "effective_sample_size": self.effective_sample_size,
            "naive_monte_carlo_estimate": self.naive_monte_carlo_estimate,
            "importance_sampling_variance": self.importance_sampling_variance,
            "naive_monte_carlo_variance": self.naive_monte_carlo_variance,
            "variance_ratio_vs_naive": self.variance_ratio_vs_naive,
            "event_count": self.event_count,
            "weight_sum": self.weight_sum,
        }


def sample_scenario_rows(spec: RareEventSamplingSpec) -> list[SampledScenarioRow]:
    """Sample proposal rows and compute base/proposal likelihood ratios.

    Returns:
        Proposal-sampled rows with likelihood-ratio bookkeeping.
    """

    rng = random.Random(spec.seed)
    rows: list[SampledScenarioRow] = []
    for sample_index in range(spec.samples):
        parameters: dict[str, float] = {}
        base_probability = 1.0
        proposal_probability = 1.0
        for distribution in spec.parameters:
            value = distribution.sample(rng)
            parameters[distribution.name] = value
            base_probability *= distribution.base_pdf(value)
            proposal_probability *= distribution.proposal_pdf(value)
        if proposal_probability <= 0.0 or not math.isfinite(proposal_probability):
            raise RareEventSamplingError(
                f"sample {sample_index}: proposal probability must be finite and positive"
            )
        likelihood_ratio = base_probability / proposal_probability
        if not math.isfinite(likelihood_ratio):
            raise RareEventSamplingError(f"sample {sample_index}: non-finite likelihood ratio")
        rows.append(
            SampledScenarioRow(
                sample_index=sample_index,
                seed=spec.seed + sample_index,
                parameters=parameters,
                base_probability=base_probability,
                proposal_probability=proposal_probability,
                likelihood_ratio=likelihood_ratio,
                parameter_vector_hash=parameter_vector_hash(parameters),
            )
        )
    return rows


def parameter_vector_hash(parameters: Mapping[str, float]) -> str:
    """Return a stable short hash for sampled parameter values.

    Returns:
        Stable 16-character SHA-256 prefix.
    """

    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def apply_sampled_scenario_mutation(
    scenario_payload: Mapping[str, Any],
    row: SampledScenarioRow,
) -> dict[str, Any]:
    """Apply supported sampled knobs to a scenario payload without mutating the input.

    The mutation layer intentionally preserves the tracked YAML shape and records unsupported knobs
    as metadata rather than editing unrelated scenario fields.

    Returns:
        Mutated scenario payload copy.
    """

    scenario = copy.deepcopy(dict(scenario_payload))
    simulation_config = _ensure_mapping_child(scenario, "simulation_config")
    metadata = _ensure_mapping_child(scenario, "metadata")
    mutation_metadata = {
        "schema_version": SCHEMA_VERSION,
        "sample_index": row.sample_index,
        "sample_seed": row.seed,
        "parameter_vector_hash": row.parameter_vector_hash,
        "parameters": dict(row.parameters),
        "likelihood_ratio": row.likelihood_ratio,
    }
    metadata["rare_event_sampling"] = mutation_metadata

    if "ped_density" in row.parameters:
        simulation_config["ped_density"] = row.parameters["ped_density"]
    if "crossing_time_offset_s" in row.parameters:
        simulation_config["crossing_time_offset_s"] = row.parameters["crossing_time_offset_s"]
    if "pedestrian_speed_multiplier" in row.parameters:
        _apply_speed_multiplier(scenario, row.parameters["pedestrian_speed_multiplier"])
    if "goal_placement_offset" in row.parameters:
        metadata["rare_event_sampling"]["goal_placement_offset"] = row.parameters[
            "goal_placement_offset"
        ]
    return scenario


def estimate_failure_probability(
    rows: list[SampledScenarioRow],
    events: list[bool],
    *,
    objective_event: str = "rare_event",
    confidence_level: float = 0.95,
) -> ImportanceSamplingEstimate:
    """Estimate a base-measure event probability from proposal-sampled binary outcomes.

    Returns:
        Importance-sampling estimate with confidence interval and effective sample size.
    """

    if len(rows) != len(events) or not rows:
        raise RareEventSamplingError("rows and events must have the same positive length")
    if not 0.0 < confidence_level < 1.0:
        raise RareEventSamplingError("confidence_level must be between 0 and 1")
    weights = [row.likelihood_ratio for row in rows]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise RareEventSamplingError("likelihood ratios must be finite and non-negative")
    weighted_indicators = [
        weight * float(event) for weight, event in zip(weights, events, strict=True)
    ]
    samples = len(rows)
    estimate = math.fsum(weighted_indicators) / samples
    if samples > 1:
        variance = math.fsum((value - estimate) ** 2 for value in weighted_indicators) / (
            samples - 1
        )
        standard_error = math.sqrt(variance / samples)
    else:
        standard_error = 0.0
        variance = 0.0
    alpha = 1.0 - confidence_level
    z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    lower = max(0.0, estimate - z_value * standard_error)
    upper = min(1.0, estimate + z_value * standard_error)
    weight_sum = math.fsum(weights)
    weight_square_sum = math.fsum(weight * weight for weight in weights)
    effective_sample_size = (
        (weight_sum * weight_sum) / weight_square_sum if weight_square_sum > 0.0 else 0.0
    )
    naive_monte_carlo_estimate = sum(events) / samples
    if samples > 1:
        naive_monte_carlo_variance = (
            naive_monte_carlo_estimate * (1.0 - naive_monte_carlo_estimate) / samples
        )
    else:
        naive_monte_carlo_variance = 0.0
    importance_sampling_variance = standard_error * standard_error
    variance_ratio_vs_naive = (
        importance_sampling_variance / naive_monte_carlo_variance
        if naive_monte_carlo_variance > 0.0
        else None
    )

    return ImportanceSamplingEstimate(
        schema_version=SCHEMA_VERSION,
        objective_event=objective_event,
        samples=samples,
        estimate=estimate,
        standard_error=standard_error,
        confidence_interval=(lower, upper),
        confidence_level=confidence_level,
        effective_sample_size=effective_sample_size,
        naive_monte_carlo_estimate=naive_monte_carlo_estimate,
        importance_sampling_variance=importance_sampling_variance,
        naive_monte_carlo_variance=naive_monte_carlo_variance,
        variance_ratio_vs_naive=variance_ratio_vs_naive,
        event_count=sum(bool(event) for event in events),
        weight_sum=weight_sum,
    )


def build_sampling_summary(
    *,
    spec: RareEventSamplingSpec,
    rows: list[SampledScenarioRow],
    events: list[bool],
    scenario_payload: Mapping[str, Any] | None = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Build a compact smoke summary with provenance and estimator output.

    Returns:
        JSON-serializable smoke summary.
    """

    estimate = estimate_failure_probability(
        rows,
        events,
        objective_event=spec.objective_event,
        confidence_level=confidence_level,
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "proposal": spec.proposal,
        "objective_event": spec.objective_event,
        "samples": spec.samples,
        "seed": spec.seed,
        "estimator": estimate.to_payload(),
        "sample_provenance": {
            "row_count": len(rows),
            "parameter_vector_hashes": [row.parameter_vector_hash for row in rows],
            "likelihood_ratio_min": min(row.likelihood_ratio for row in rows),
            "likelihood_ratio_max": max(row.likelihood_ratio for row in rows),
        },
        "claim_boundary": (
            "diagnostic rare-event harness smoke only; not a benchmark campaign or "
            "failure-rate claim"
        ),
    }
    if scenario_payload is not None:
        scenario_name = (
            scenario_payload.get("name") if isinstance(scenario_payload, Mapping) else None
        )
        summary["scenario_provenance"] = {
            "scenario_name": scenario_name,
            "mutated_parameter_vector_hash": rows[0].parameter_vector_hash if rows else None,
        }
    return summary


def _apply_speed_multiplier(scenario: dict[str, Any], multiplier: float) -> None:
    pedestrians = scenario.get("single_pedestrians")
    if not isinstance(pedestrians, list):
        _ensure_mapping_child(scenario, "metadata")["pedestrian_speed_multiplier"] = multiplier
        return
    for pedestrian in pedestrians:
        if not isinstance(pedestrian, dict):
            continue
        for key in ("speed", "preferred_speed", "v_pref"):
            if isinstance(pedestrian.get(key), int | float):
                pedestrian[key] = float(pedestrian[key]) * multiplier


def _ensure_mapping_child(parent: dict[str, Any], key: str) -> dict[str, Any]:
    child = parent.get(key)
    if child is None:
        child = {}
        parent[key] = child
    if not isinstance(child, dict):
        raise RareEventSamplingError(f"scenario.{key}: expected mapping")
    return child


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _normal_pdf(value: float, mean: float, std: float) -> float:
    exponent = -0.5 * ((value - mean) / std) ** 2
    return math.exp(exponent) / (std * math.sqrt(2.0 * math.pi))


__all__ = [
    "SCHEMA_VERSION",
    "ImportanceSamplingEstimate",
    "ParameterDistribution",
    "RareEventSamplingError",
    "RareEventSamplingSpec",
    "SampledScenarioRow",
    "apply_sampled_scenario_mutation",
    "build_sampling_summary",
    "estimate_failure_probability",
    "parameter_vector_hash",
    "sample_scenario_rows",
]
