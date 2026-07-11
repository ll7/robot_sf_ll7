"""Rank generated scenario hypotheses with deterministic, configurable scores.

The selector adapts min-max normalization to the current generated archive. It
only prioritizes hypotheses for review; it never certifies them or turns them
into benchmark evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from robot_sf.benchmark.scenario_generation.catalog_schema import validate_catalog_entry
from robot_sf.errors import RobotSfError

CONFIG_SCHEMA_VERSION = "generated-scenario-adaptive-proposal-selection.v1"
SELECTION_SCHEMA_VERSION = "generated-scenario-adaptive-proposal-selection-result.v1"
SELECTOR_ID = "adaptive_min_max_rank.v1"
CLAIM_BOUNDARY = "generated scenario hypotheses only"
_DIRECTIONS = {"higher_is_better", "lower_is_better"}


class GeneratedScenarioAdaptiveSelectionError(RobotSfError, ValueError):
    """Raised when generated hypotheses cannot safely be ranked."""


@dataclass(frozen=True, slots=True)
class ScoringCriterion:
    """One numeric generated-entry field contributing to proposal priority."""

    field: str
    direction: Literal["higher_is_better", "lower_is_better"]
    weight: float


@dataclass(frozen=True, slots=True)
class AdaptiveSelectionSpec:
    """Validated controls for adaptive deterministic proposal selection."""

    proposal_count: int
    criteria: tuple[ScoringCriterion, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AdaptiveSelectionSpec:
        """Parse a versioned proposal-selection configuration.

        Returns:
            Validated adaptive selection controls.
        """

        if not isinstance(payload, Mapping):
            raise GeneratedScenarioAdaptiveSelectionError("selector config must be a mapping")
        if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise GeneratedScenarioAdaptiveSelectionError(
                f"schema_version must be {CONFIG_SCHEMA_VERSION!r}"
            )
        if payload.get("claim_boundary") != CLAIM_BOUNDARY:
            raise GeneratedScenarioAdaptiveSelectionError(
                f"claim_boundary must be {CLAIM_BOUNDARY!r}"
            )
        proposal_count = _integer(payload.get("proposal_count"), "proposal_count")
        if proposal_count <= 0:
            raise GeneratedScenarioAdaptiveSelectionError("proposal_count must be > 0")
        selector = payload.get("selector")
        if not isinstance(selector, Mapping) or selector.get("type") != SELECTOR_ID:
            raise GeneratedScenarioAdaptiveSelectionError(f"selector.type must be {SELECTOR_ID!r}")
        raw_criteria = selector.get("criteria")
        if not isinstance(raw_criteria, Sequence) or isinstance(raw_criteria, str | bytes):
            raise GeneratedScenarioAdaptiveSelectionError(
                "selector.criteria must be a non-empty sequence"
            )
        if not raw_criteria:
            raise GeneratedScenarioAdaptiveSelectionError(
                "selector.criteria must be a non-empty sequence"
            )
        criteria = tuple(_parse_criterion(item, index) for index, item in enumerate(raw_criteria))
        fields = [criterion.field for criterion in criteria]
        if len(set(fields)) != len(fields):
            raise GeneratedScenarioAdaptiveSelectionError(
                "selector.criteria field values must be unique"
            )
        return cls(proposal_count=proposal_count, criteria=criteria)


def load_adaptive_selection_config(path: Path) -> dict[str, Any]:
    """Load and validate the standalone selector configuration.

    Returns:
        Validated mutable configuration mapping.
    """

    if not path.is_file():
        raise GeneratedScenarioAdaptiveSelectionError(f"config file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise GeneratedScenarioAdaptiveSelectionError("selector config must be a mapping")
    config = dict(payload)
    AdaptiveSelectionSpec.from_payload(config)
    _non_empty_string(config.get("source_archive"), "source_archive")
    _non_empty_string(config.get("output_path"), "output_path")
    return config


def select_generated_proposals(
    entries: Sequence[Mapping[str, Any]],
    *,
    spec: AdaptiveSelectionSpec,
) -> dict[str, Any]:
    """Rank and select generated hypotheses using archive-adaptive normalization.

    Each configured numeric field is min-max normalized over the complete current
    archive. Direction-adjusted components are combined as a weighted mean. Stable
    scenario ids break score ties, so source ordering cannot change the result.

    Returns:
        Scoring ranges, all candidate scores, and the selected proposal rows.
    """

    normalized = _validated_entries(entries)
    normalized.sort(key=lambda entry: entry["scenario_id"])
    scenario_ids = [str(entry["scenario_id"]) for entry in normalized]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise GeneratedScenarioAdaptiveSelectionError("archive scenario_id values must be unique")
    if spec.proposal_count > len(normalized):
        raise GeneratedScenarioAdaptiveSelectionError(
            f"proposal_count {spec.proposal_count} exceeds archive size {len(normalized)}"
        )

    values_by_field: dict[str, list[float]] = {}
    for criterion in spec.criteria:
        values_by_field[criterion.field] = [
            _numeric_field(entry, criterion.field, scenario_id=str(entry["scenario_id"]))
            for entry in normalized
        ]

    ranges = {
        field: {"min": min(values), "max": max(values)} for field, values in values_by_field.items()
    }
    total_weight = sum(criterion.weight for criterion in spec.criteria)
    scored: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(normalized):
        components = []
        weighted_sum = 0.0
        for criterion in spec.criteria:
            raw_value = values_by_field[criterion.field][entry_index]
            bounds = ranges[criterion.field]
            normalized_value = _normalized_desirability(
                raw_value,
                minimum=bounds["min"],
                maximum=bounds["max"],
                direction=criterion.direction,
            )
            contribution = criterion.weight * normalized_value
            weighted_sum += contribution
            components.append(
                {
                    "field": criterion.field,
                    "direction": criterion.direction,
                    "weight": criterion.weight,
                    "raw_value": raw_value,
                    "normalized_value": normalized_value,
                    "weighted_contribution": contribution,
                }
            )
        scored.append(
            {
                "scenario_id": entry["scenario_id"],
                "score": weighted_sum / total_weight,
                "score_components": components,
                "entry": entry,
            }
        )

    scored.sort(key=lambda row: (-row["score"], row["scenario_id"]))
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    return {
        "normalization_ranges": ranges,
        "scored_candidates": scored,
        "selected": deepcopy(scored[: spec.proposal_count]),
    }


def run_adaptive_selection(
    config_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Rank a generated catalog and persist complete selection provenance.

    Returns:
        Persisted adaptive proposal-selection payload.
    """

    config_path = config_path.resolve()
    config = load_adaptive_selection_config(config_path)
    spec = AdaptiveSelectionSpec.from_payload(config)
    source_archive = _resolve_config_path(
        _non_empty_string(config["source_archive"], "source_archive"), config_path=config_path
    )
    if not source_archive.is_file():
        raise GeneratedScenarioAdaptiveSelectionError(
            f"source archive does not exist or is not a file: {source_archive}"
        )
    raw_archive = source_archive.read_bytes()
    archive = yaml.safe_load(raw_archive)
    if not isinstance(archive, Mapping):
        raise GeneratedScenarioAdaptiveSelectionError("source archive must be a mapping")
    if archive.get("schema_version") != "generated-scenario-catalog.v1":
        raise GeneratedScenarioAdaptiveSelectionError(
            "source archive schema_version must be 'generated-scenario-catalog.v1'"
        )
    _validate_archive_metadata(archive.get("metadata"))
    entries = archive.get("entries")
    selection = select_generated_proposals(entries, spec=spec)
    output_path = output_path or _resolve_config_path(
        _non_empty_string(config["output_path"], "output_path"), config_path=config_path
    )
    if output_path.exists():
        raise FileExistsError(f"output_path already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selector": {
            "id": SELECTOR_ID,
            "proposal_count": spec.proposal_count,
            "normalization": "current_archive_min_max.v1",
            "criteria": [
                {
                    "field": criterion.field,
                    "direction": criterion.direction,
                    "weight": criterion.weight,
                }
                for criterion in spec.criteria
            ],
        },
        "source_archive": {
            "path": source_archive.as_posix(),
            "sha256": hashlib.sha256(raw_archive).hexdigest(),
            "schema_version": archive["schema_version"],
            "entry_count": len(entries),
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "governance": {
            "required_manual_review": True,
            "benchmark_evidence": False,
            "scenario_certification": False,
            "automatic_promotion": False,
        },
        **selection,
    }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_criterion(raw: object, index: int) -> ScoringCriterion:
    if not isinstance(raw, Mapping):
        raise GeneratedScenarioAdaptiveSelectionError(
            f"selector.criteria[{index}] must be a mapping"
        )
    unknown = set(raw) - {"field", "direction", "weight"}
    if unknown:
        raise GeneratedScenarioAdaptiveSelectionError(
            f"selector.criteria[{index}] has unknown fields: {sorted(unknown)}"
        )
    field = _non_empty_string(raw.get("field"), f"selector.criteria[{index}].field")
    if field.startswith(".") or field.endswith(".") or ".." in field:
        raise GeneratedScenarioAdaptiveSelectionError(
            f"selector.criteria[{index}].field must be a dotted mapping path"
        )
    direction = raw.get("direction")
    if direction not in _DIRECTIONS:
        raise GeneratedScenarioAdaptiveSelectionError(
            f"selector.criteria[{index}].direction must be one of {sorted(_DIRECTIONS)}"
        )
    weight = _positive_finite(raw.get("weight"), f"selector.criteria[{index}].weight")
    return ScoringCriterion(field=field, direction=direction, weight=weight)


def _numeric_field(entry: Mapping[str, Any], field: str, *, scenario_id: str) -> float:
    value: object = entry
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise GeneratedScenarioAdaptiveSelectionError(
                f"scenario {scenario_id!r} is missing configured scoring field {field!r}"
            )
        value = value[part]
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise GeneratedScenarioAdaptiveSelectionError(
            f"scenario {scenario_id!r} scoring field {field!r} must be a finite number"
        )
    return float(value)


def _normalized_desirability(
    value: float,
    *,
    minimum: float,
    maximum: float,
    direction: str,
) -> float:
    if maximum == minimum:
        return 1.0
    normalized = (value - minimum) / (maximum - minimum)
    return normalized if direction == "higher_is_better" else 1.0 - normalized


def _validated_entries(entries: object) -> list[dict[str, Any]]:
    if not isinstance(entries, Sequence) or isinstance(entries, str | bytes):
        raise GeneratedScenarioAdaptiveSelectionError(
            "archive entries must be a non-empty sequence"
        )
    if not entries:
        raise GeneratedScenarioAdaptiveSelectionError("generated scenario archive is empty")
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise GeneratedScenarioAdaptiveSelectionError(
                f"archive entry {index} must be a mapping"
            )
        normalized_entry = deepcopy(dict(entry))
        try:
            validate_catalog_entry(normalized_entry)
        except (TypeError, ValueError) as error:
            raise GeneratedScenarioAdaptiveSelectionError(
                f"archive entry {index} is invalid: {error}"
            ) from error
        normalized.append(normalized_entry)
    return normalized


def _validate_archive_metadata(raw_metadata: object) -> None:
    if not isinstance(raw_metadata, Mapping):
        raise GeneratedScenarioAdaptiveSelectionError("source archive metadata must be a mapping")
    required_values = {
        "source": "auto_generated",
        "required_manual_review": True,
        "benchmark_evidence": False,
    }
    for key, expected in required_values.items():
        if raw_metadata.get(key) != expected:
            raise GeneratedScenarioAdaptiveSelectionError(
                f"source archive metadata.{key} must be {expected!r}"
            )


def _resolve_config_path(value: str, *, config_path: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def _integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GeneratedScenarioAdaptiveSelectionError(f"{path} must be an integer")
    return value


def _positive_finite(value: object, path: str) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise GeneratedScenarioAdaptiveSelectionError(f"{path} must be finite and > 0")
    return float(value)


def _non_empty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeneratedScenarioAdaptiveSelectionError(f"{path} must be a non-empty string")
    return value.strip()


__all__ = [
    "CLAIM_BOUNDARY",
    "CONFIG_SCHEMA_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "SELECTOR_ID",
    "AdaptiveSelectionSpec",
    "GeneratedScenarioAdaptiveSelectionError",
    "ScoringCriterion",
    "load_adaptive_selection_config",
    "run_adaptive_selection",
    "select_generated_proposals",
]
