"""Deterministically bias selection toward critical generated-scenario records."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.scenario_generation.catalog_schema import validate_catalog_entry
from robot_sf.errors import RobotSfError

CONFIG_SCHEMA_VERSION = "generated-scenario-rare-event-sampling.v1"
SELECTION_SCHEMA_VERSION = "generated-scenario-rare-event-selection.v1"
SAMPLER_ID = "criticality_weighted_without_replacement.v1"
CLAIM_BOUNDARY = "generated scenario hypotheses only"


class GeneratedScenarioArchiveSamplingError(RobotSfError, ValueError):
    """Raised when an archive cannot safely produce a rare-event selection."""


@dataclass(frozen=True, slots=True)
class ArchiveSamplingSpec:
    """Validated controls for criticality-weighted archive sampling."""

    seed: int
    sample_size: int
    clearance_floor_m: float
    exponent: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ArchiveSamplingSpec:
        """Parse and fail closed on the versioned sampler configuration.

        Returns:
            Validated archive-sampling controls.
        """

        if not isinstance(payload, Mapping):
            raise GeneratedScenarioArchiveSamplingError("sampler config must be a mapping")
        if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise GeneratedScenarioArchiveSamplingError(
                f"schema_version must be {CONFIG_SCHEMA_VERSION!r}"
            )
        if payload.get("claim_boundary") != CLAIM_BOUNDARY:
            raise GeneratedScenarioArchiveSamplingError(
                f"claim_boundary must be {CLAIM_BOUNDARY!r}"
            )
        sampler = payload.get("sampler")
        if not isinstance(sampler, Mapping) or sampler.get("type") != SAMPLER_ID:
            raise GeneratedScenarioArchiveSamplingError(f"sampler.type must be {SAMPLER_ID!r}")
        if sampler.get("metric") != "min_clearance_m":
            raise GeneratedScenarioArchiveSamplingError("sampler.metric must be 'min_clearance_m'")
        if sampler.get("direction") != "lower_is_more_critical":
            raise GeneratedScenarioArchiveSamplingError(
                "sampler.direction must be 'lower_is_more_critical'"
            )
        seed = _integer(payload.get("seed"), "seed")
        sample_size = _integer(payload.get("sample_size"), "sample_size")
        if sample_size <= 0:
            raise GeneratedScenarioArchiveSamplingError("sample_size must be > 0")
        clearance_floor_m = _positive_finite(
            sampler.get("clearance_floor_m"), "sampler.clearance_floor_m"
        )
        exponent = _positive_finite(sampler.get("exponent"), "sampler.exponent")
        return cls(
            seed=seed,
            sample_size=sample_size,
            clearance_floor_m=clearance_floor_m,
            exponent=exponent,
        )


def load_archive_sampling_config(path: Path) -> dict[str, Any]:
    """Load a sampler config and validate its complete standalone contract.

    Returns:
        Validated mutable configuration mapping.
    """

    if not path.is_file():
        raise GeneratedScenarioArchiveSamplingError(f"config file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise GeneratedScenarioArchiveSamplingError("sampler config must be a mapping")
    config = dict(payload)
    ArchiveSamplingSpec.from_payload(config)
    _non_empty_string(config.get("source_archive"), "source_archive")
    _non_empty_string(config.get("output_path"), "output_path")
    return config


def sample_generated_archive(
    entries: Sequence[Mapping[str, Any]],
    *,
    spec: ArchiveSamplingSpec,
) -> list[dict[str, Any]]:
    """Select entries with deterministic criticality weighting and no replacement.

    The Efraimidis--Spirakis priority-key method gives each entry a probability
    proportional to ``max(min_clearance_m, floor) ** -exponent``. Sorting the
    archive by stable scenario id before drawing makes results independent of
    source-file entry order.

    Returns:
        Selection rows ordered by draw priority, including reproducibility data.
    """

    normalized = _validated_entries(entries)
    normalized.sort(key=lambda entry: entry["scenario_id"])
    scenario_ids = [str(entry["scenario_id"]) for entry in normalized]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise GeneratedScenarioArchiveSamplingError("archive scenario_id values must be unique")
    if spec.sample_size > len(normalized):
        raise GeneratedScenarioArchiveSamplingError(
            f"sample_size {spec.sample_size} exceeds archive size {len(normalized)}"
        )

    rng = random.Random(spec.seed)
    candidates: list[dict[str, Any]] = []
    for entry in normalized:
        clearance_m = float(entry["criticality"]["source_metrics"]["min_clearance_m"])
        weight = _criticality_weight(clearance_m, spec)
        random_draw = 1.0 - rng.random()
        selection_key = math.log(random_draw) / weight
        candidates.append(
            {
                "scenario_id": entry["scenario_id"],
                "criticality": {
                    "metric": "min_clearance_m",
                    "value": clearance_m,
                    "weight": weight,
                },
                "random_draw": random_draw,
                "selection_key": selection_key,
                "entry": entry,
            }
        )
    candidates.sort(key=lambda row: (-row["selection_key"], row["scenario_id"]))
    selected = candidates[: spec.sample_size]
    for selection_rank, row in enumerate(selected, start=1):
        row["selection_rank"] = selection_rank
    return selected


def run_archive_sampling(
    config_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Load a generated catalog, sample it, and persist selection provenance.

    Returns:
        Persisted selection payload with source and draw provenance.
    """

    config_path = config_path.resolve()
    config = load_archive_sampling_config(config_path)
    spec = ArchiveSamplingSpec.from_payload(config)
    source_archive = _resolve_config_path(
        _non_empty_string(config["source_archive"], "source_archive"), config_path=config_path
    )
    if not source_archive.is_file():
        raise GeneratedScenarioArchiveSamplingError(
            f"source archive does not exist or is not a file: {source_archive}"
        )
    raw_archive = source_archive.read_bytes()
    archive = yaml.safe_load(raw_archive)
    if not isinstance(archive, Mapping):
        raise GeneratedScenarioArchiveSamplingError("source archive must be a mapping")
    if archive.get("schema_version") != "generated-scenario-catalog.v1":
        raise GeneratedScenarioArchiveSamplingError(
            "source archive schema_version must be 'generated-scenario-catalog.v1'"
        )
    _validate_archive_metadata(archive.get("metadata"))
    entries = archive.get("entries")
    selected = sample_generated_archive(entries, spec=spec)
    output_path = output_path or _resolve_config_path(
        _non_empty_string(config["output_path"], "output_path"), config_path=config_path
    )
    if output_path.exists():
        raise FileExistsError(f"output_path already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "sampler": {
            "id": SAMPLER_ID,
            "seed": spec.seed,
            "sample_size": spec.sample_size,
            "metric": "min_clearance_m",
            "direction": "lower_is_more_critical",
            "clearance_floor_m": spec.clearance_floor_m,
            "exponent": spec.exponent,
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
        },
        "selected": selected,
    }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GeneratedScenarioArchiveSamplingError(f"{path} must be an integer")
    return value


def _resolve_config_path(value: str, *, config_path: Path) -> Path:
    """Resolve an absolute path or a path relative to the owning sampler config."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def _validated_entries(entries: object) -> list[dict[str, Any]]:
    if not isinstance(entries, Sequence) or isinstance(entries, str | bytes):
        raise GeneratedScenarioArchiveSamplingError("archive entries must be a non-empty sequence")
    if not entries:
        raise GeneratedScenarioArchiveSamplingError("generated scenario archive is empty")
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise GeneratedScenarioArchiveSamplingError(f"archive entry {index} must be a mapping")
        normalized_entry = deepcopy(dict(entry))
        try:
            validate_catalog_entry(normalized_entry)
        except (TypeError, ValueError) as error:
            raise GeneratedScenarioArchiveSamplingError(
                f"archive entry {index} is invalid: {error}"
            ) from error
        normalized.append(normalized_entry)
    return normalized


def _criticality_weight(clearance_m: float, spec: ArchiveSamplingSpec) -> float:
    if not math.isfinite(clearance_m) or clearance_m < 0.0:
        raise GeneratedScenarioArchiveSamplingError(
            "criticality.source_metrics.min_clearance_m must be finite and >= 0"
        )
    try:
        weight = max(clearance_m, spec.clearance_floor_m) ** -spec.exponent
    except OverflowError as error:
        raise GeneratedScenarioArchiveSamplingError(
            "criticality weight must be finite and positive"
        ) from error
    if not math.isfinite(weight) or weight <= 0.0:
        raise GeneratedScenarioArchiveSamplingError(
            "criticality weight must be finite and positive"
        )
    return weight


def _validate_archive_metadata(raw_metadata: object) -> None:
    if not isinstance(raw_metadata, Mapping):
        raise GeneratedScenarioArchiveSamplingError("source archive metadata must be a mapping")
    required_values = {
        "source": "auto_generated",
        "required_manual_review": True,
        "benchmark_evidence": False,
    }
    for key, expected in required_values.items():
        if raw_metadata.get(key) != expected:
            raise GeneratedScenarioArchiveSamplingError(
                f"source archive metadata.{key} must be {expected!r}"
            )


def _positive_finite(value: object, path: str) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise GeneratedScenarioArchiveSamplingError(f"{path} must be finite and > 0")
    return float(value)


def _non_empty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeneratedScenarioArchiveSamplingError(f"{path} must be a non-empty string")
    return value.strip()


__all__ = [
    "CLAIM_BOUNDARY",
    "CONFIG_SCHEMA_VERSION",
    "SAMPLER_ID",
    "SELECTION_SCHEMA_VERSION",
    "ArchiveSamplingSpec",
    "GeneratedScenarioArchiveSamplingError",
    "load_archive_sampling_config",
    "run_archive_sampling",
    "sample_generated_archive",
]
