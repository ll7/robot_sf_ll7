"""Leakage-safe parametric curriculum fixtures for research diagnostics.

This module defines the smallest reusable contract needed to test a structured
social-navigation curriculum before training is authorized. It samples a
typed parameter vector, creates independent training/evaluation manifests,
checks split leakage, and verifies deterministic replay. It deliberately does
not create environments, train policies, or report navigation outcomes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

import numpy as np

PARAMETRIC_CURRICULUM_SCHEMA_VERSION = "parametric_curriculum.v1"
PARAMETRIC_CURRICULUM_DIAGNOSTIC_SCHEMA = "parametric_curriculum_diagnostic.v1"
CURRICULUM_CLAIM_BOUNDARY = (
    "fixture-only curriculum methodology; not training, safety, or benchmark evidence"
)
ParameterKind = Literal["continuous", "categorical"]
SamplingStrategy = Literal["fixed", "random", "structured"]


def canonical_sha256(payload: object) -> str:
    """Return the SHA-256 digest of a canonical JSON-compatible payload."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ParameterDimension:
    """One continuous or categorical dimension of a scenario parameter vector."""

    name: str
    kind: ParameterKind
    minimum: float | None = None
    maximum: float | None = None
    values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate dimension bounds and categorical values."""

        name = self.name.strip()
        if not name:
            raise ValueError("parameter dimension name must not be empty")
        object.__setattr__(self, "name", name)
        if self.kind not in {"continuous", "categorical"}:
            raise ValueError(f"unsupported parameter dimension kind: {self.kind!r}")
        if self.kind == "continuous":
            if self.minimum is None or self.maximum is None:
                raise ValueError(f"continuous dimension {name!r} requires minimum and maximum")
            minimum = float(self.minimum)
            maximum = float(self.maximum)
            if not isfinite(minimum) or not isfinite(maximum) or minimum >= maximum:
                raise ValueError(f"continuous dimension {name!r} requires finite minimum < maximum")
            object.__setattr__(self, "minimum", minimum)
            object.__setattr__(self, "maximum", maximum)
            if self.values:
                raise ValueError(f"continuous dimension {name!r} cannot define values")
        else:
            values = tuple(str(value).strip() for value in self.values)
            if not values or any(not value for value in values):
                raise ValueError(f"categorical dimension {name!r} requires non-empty values")
            if len(set(values)) != len(values):
                raise ValueError(f"categorical dimension {name!r} has duplicate values")
            object.__setattr__(self, "values", values)
            if self.minimum is not None or self.maximum is not None:
                raise ValueError(f"categorical dimension {name!r} cannot define numeric bounds")

    def validate(self, value: object) -> None:
        """Reject values outside this dimension's declared support."""

        if self.kind == "continuous":
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"parameter {self.name!r} must be numeric") from exc
            if not isfinite(numeric) or numeric < self.minimum or numeric > self.maximum:
                raise ValueError(
                    f"parameter {self.name!r} must be finite in [{self.minimum}, {self.maximum}]"
                )
            return
        if not isinstance(value, str) or value not in self.values:
            raise ValueError(f"parameter {self.name!r} must be one of {list(self.values)}")

    def sample(self, rng: np.random.Generator, *, normalized: float | None = None) -> object:
        """Sample one value, optionally at a deterministic normalized position.

        Returns:
            One value within the dimension's declared support.
        """

        if self.kind == "continuous":
            if normalized is None:
                return float(rng.uniform(self.minimum, self.maximum))
            clipped = min(max(float(normalized), 0.0), 1.0)
            return float(self.minimum + clipped * (self.maximum - self.minimum))
        if normalized is None:
            index = int(rng.integers(0, len(self.values)))
        else:
            index = min(int(float(normalized) * len(self.values)), len(self.values) - 1)
        return self.values[index]

    def to_dict(self) -> dict[str, Any]:
        """Return the schema-facing dimension representation."""

        if self.kind == "continuous":
            return {
                "name": self.name,
                "kind": self.kind,
                "minimum": self.minimum,
                "maximum": self.maximum,
            }
        return {"name": self.name, "kind": self.kind, "values": list(self.values)}


@dataclass(frozen=True, slots=True)
class ScenarioParameterSpace:
    """Validated ordered parameter space used by curriculum fixture manifests."""

    dimensions: tuple[ParameterDimension, ...]

    def __post_init__(self) -> None:
        """Reject empty or ambiguously named parameter spaces."""

        if not self.dimensions:
            raise ValueError("parameter space requires at least one dimension")
        names = [dimension.name for dimension in self.dimensions]
        if len(set(names)) != len(names):
            raise ValueError("parameter space dimension names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        """Return parameter names in their declared order."""

        return tuple(dimension.name for dimension in self.dimensions)

    def validate(self, parameters: Mapping[str, object]) -> dict[str, object]:
        """Validate one complete parameter vector and return a plain mapping.

        Returns:
            Normalized JSON-compatible parameter mapping.
        """

        expected = set(self.names)
        actual = set(parameters)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(f"parameter vector keys mismatch: missing={missing}, extra={extra}")
        normalized: dict[str, object] = {}
        for dimension in self.dimensions:
            value = parameters[dimension.name]
            dimension.validate(value)
            normalized[dimension.name] = (
                float(value) if dimension.kind == "continuous" else str(value)
            )
        return normalized

    def sample(
        self,
        *,
        seed: int,
        count: int,
        strategy: SamplingStrategy,
    ) -> tuple[dict[str, object], ...]:
        """Create deterministic parameter vectors for a named sampling strategy.

        Returns:
            Ordered parameter vectors generated from ``seed``.
        """

        if count <= 0:
            raise ValueError("parameter sample count must be positive")
        if strategy not in {"fixed", "random", "structured"}:
            raise ValueError(f"unsupported curriculum sampling strategy: {strategy!r}")
        rng = np.random.default_rng(int(seed))
        samples: list[dict[str, object]] = []
        for index in range(count):
            if strategy == "fixed":
                normalized_positions = [0.5] * len(self.dimensions)
            elif strategy == "structured":
                progress = 0.5 if count == 1 else index / (count - 1)
                normalized_positions = [
                    progress if dim % 2 == 0 else 1.0 - progress
                    for dim in range(len(self.dimensions))
                ]
            else:
                normalized_positions = [None] * len(self.dimensions)
            parameters = {
                dimension.name: dimension.sample(rng, normalized=normalized)
                for dimension, normalized in zip(self.dimensions, normalized_positions, strict=True)
            }
            samples.append(self.validate(parameters))
        return tuple(samples)

    def to_dict(self) -> dict[str, Any]:
        """Return the ordered schema-facing parameter-space representation."""

        return {
            "schema_version": PARAMETRIC_CURRICULUM_SCHEMA_VERSION,
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
            "digest": canonical_sha256([dimension.to_dict() for dimension in self.dimensions]),
        }


@dataclass(frozen=True, slots=True)
class CurriculumManifest:
    """Immutable split manifest with a stable digest and replay inputs."""

    split: Literal["train", "evaluation"]
    seed: int
    strategy: SamplingStrategy
    entries: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        """Validate manifest identity fields and entry shape."""

        if self.split not in {"train", "evaluation"}:
            raise ValueError("curriculum manifest split must be train or evaluation")
        if self.strategy not in {"fixed", "random", "structured"}:
            raise ValueError(f"unsupported curriculum manifest strategy: {self.strategy!r}")
        if not self.entries:
            raise ValueError("curriculum manifest must contain at least one entry")
        ids = [str(entry.get("scenario_id") or "") for entry in self.entries]
        if any(not identifier for identifier in ids) or len(set(ids)) != len(ids):
            raise ValueError("curriculum manifest scenario ids must be unique and non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical manifest payload."""

        return {
            "schema_version": PARAMETRIC_CURRICULUM_SCHEMA_VERSION,
            "split": self.split,
            "seed": int(self.seed),
            "strategy": self.strategy,
            "entries": [dict(entry) for entry in self.entries],
        }

    @property
    def digest(self) -> str:
        """Return the stable manifest digest."""

        return canonical_sha256(self.to_dict())

    @property
    def parameter_hashes(self) -> frozenset[str]:
        """Return stable hashes for parameter vectors in this manifest."""

        return frozenset(canonical_sha256(entry["parameters"]) for entry in self.entries)


def build_manifest(
    space: ScenarioParameterSpace,
    *,
    split: Literal["train", "evaluation"],
    seed: int,
    count: int,
    strategy: SamplingStrategy,
    scenario_prefix: str,
) -> CurriculumManifest:
    """Sample and label one independent training or evaluation manifest.

    Returns:
        Immutable manifest with deterministic entries and split identity.
    """

    prefix = scenario_prefix.strip()
    if not prefix:
        raise ValueError("scenario_prefix must not be empty")
    samples = space.sample(seed=seed, count=count, strategy=strategy)
    entries = tuple(
        {
            "scenario_id": f"{prefix}-{index:03d}",
            "parameters": parameters,
        }
        for index, parameters in enumerate(samples)
    )
    return CurriculumManifest(split=split, seed=seed, strategy=strategy, entries=entries)


def validate_no_leakage(
    train_manifest: CurriculumManifest,
    evaluation_manifest: CurriculumManifest,
) -> dict[str, Any]:
    """Fail closed when training and evaluation identities or vectors overlap.

    Returns:
        Passed leakage-check record when no overlap is found.
    """

    if train_manifest.split != "train" or evaluation_manifest.split != "evaluation":
        raise ValueError("leakage check requires train and evaluation manifests")
    train_ids = {entry["scenario_id"] for entry in train_manifest.entries}
    evaluation_ids = {entry["scenario_id"] for entry in evaluation_manifest.entries}
    overlapping_ids = sorted(train_ids & evaluation_ids)
    overlapping_parameters = sorted(
        train_manifest.parameter_hashes & evaluation_manifest.parameter_hashes
    )
    if overlapping_ids or overlapping_parameters:
        raise ValueError(
            "curriculum manifest leakage detected: "
            f"scenario_ids={overlapping_ids}, parameter_hashes={overlapping_parameters}"
        )
    return {
        "status": "passed",
        "overlapping_scenario_ids": [],
        "overlapping_parameter_hashes": [],
    }


def verify_replay(
    space: ScenarioParameterSpace,
    manifest: CurriculumManifest,
) -> bool:
    """Recreate a manifest from its seed and prove byte-level entry equality.

    Returns:
        ``True`` when regenerated entries match exactly.
    """

    regenerated = build_manifest(
        space,
        split=manifest.split,
        seed=manifest.seed,
        count=len(manifest.entries),
        strategy=manifest.strategy,
        scenario_prefix=str(manifest.entries[0]["scenario_id"]).rsplit("-", 1)[0],
    )
    if regenerated.entries != manifest.entries:
        raise ValueError(f"curriculum manifest replay mismatch for {manifest.split} split")
    return True


def build_parametric_curriculum_report(
    space: ScenarioParameterSpace,
    *,
    seed: int,
    train_count: int,
    evaluation_count: int,
) -> dict[str, Any]:
    """Build a fixture-only report for no, random, and structured curriculum lanes.

    Returns:
        JSON-compatible diagnostic report with method cards and manifests.
    """

    evaluation = build_manifest(
        space,
        split="evaluation",
        seed=seed + 10_000,
        count=evaluation_count,
        strategy="random",
        scenario_prefix="evaluation",
    )
    methods: list[dict[str, Any]] = []
    for offset, (method_id, strategy) in enumerate(
        (
            ("no_curriculum", "fixed"),
            ("random_curriculum", "random"),
            ("structured_curriculum", "structured"),
        )
    ):
        train = build_manifest(
            space,
            split="train",
            seed=seed + offset,
            count=train_count,
            strategy=strategy,
            scenario_prefix=f"{method_id}-train",
        )
        leakage = validate_no_leakage(train, evaluation)
        replay_verified = verify_replay(space, train) and verify_replay(space, evaluation)
        methods.append(
            {
                "method_id": method_id,
                "strategy": strategy,
                "training_manifest": train.to_dict() | {"sha256": train.digest},
                "evaluation_manifest": evaluation.to_dict() | {"sha256": evaluation.digest},
                "matched_train_count": train_count,
                "replay_verified": replay_verified,
                "leakage_check": leakage,
                "training_executed": False,
                "result_status": "not_executed",
            }
        )
    return {
        "schema_version": PARAMETRIC_CURRICULUM_DIAGNOSTIC_SCHEMA,
        "evidence_tier": "diagnostic-only",
        "claim_boundary": CURRICULUM_CLAIM_BOUNDARY,
        "seed": int(seed),
        "parameter_space": space.to_dict(),
        "methods": methods,
        "simulator_executed": False,
        "training_executed": False,
        "benchmark_evidence": False,
        "held_out_evaluation": True,
        "approval_required_for_training": True,
    }


def build_parameter_space(payload: Mapping[str, Any]) -> ScenarioParameterSpace:
    """Build a parameter space from a config-first dimension mapping.

    Returns:
        Validated ordered parameter space.
    """

    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("parameter_space must be a non-empty mapping")
    dimensions: list[ParameterDimension] = []
    for name, raw in payload.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"parameter dimension {name!r} must be a mapping")
        kind = raw.get("kind")
        if kind == "continuous":
            dimensions.append(
                ParameterDimension(
                    name=str(name),
                    kind="continuous",
                    minimum=raw.get("minimum"),
                    maximum=raw.get("maximum"),
                )
            )
        elif kind == "categorical":
            values = raw.get("values")
            if not isinstance(values, Sequence) or isinstance(values, str):
                raise ValueError(f"categorical dimension {name!r} values must be a list")
            dimensions.append(
                ParameterDimension(name=str(name), kind="categorical", values=tuple(values))
            )
        else:
            raise ValueError(f"parameter dimension {name!r} has unsupported kind {kind!r}")
    return ScenarioParameterSpace(tuple(dimensions))


__all__ = [
    "CURRICULUM_CLAIM_BOUNDARY",
    "PARAMETRIC_CURRICULUM_DIAGNOSTIC_SCHEMA",
    "PARAMETRIC_CURRICULUM_SCHEMA_VERSION",
    "CurriculumManifest",
    "ParameterDimension",
    "ScenarioParameterSpace",
    "build_manifest",
    "build_parameter_space",
    "build_parametric_curriculum_report",
    "canonical_sha256",
    "validate_no_leakage",
    "verify_replay",
]
