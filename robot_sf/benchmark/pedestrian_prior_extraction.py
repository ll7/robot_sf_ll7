"""Fixture/local pedestrian-prior extraction helpers for issue #2918.

This module extracts compact bounded prior summaries from already-available
trajectory records. It intentionally does not download, stage, or store raw
external datasets. Dataset-backed claim admission remains owned by
``pedestrian_prior_extraction_manifest``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.pedestrian_prior_extraction_manifest import (
    PRIOR_EXTRACTION_EVIDENCE_BOUNDARY,
    REQUIRED_PRIOR_PARAMETERS,
)

PEDESTRIAN_PRIOR_EXTRACTION_REPORT_SCHEMA_VERSION = "pedestrian_prior_extraction_report.v1"


class PedestrianPriorExtractionError(ValueError):
    """Raised when trajectory input cannot support prior extraction."""


@dataclass(frozen=True)
class PriorParameterSummary:
    """Compact bounded summary for one extracted prior parameter."""

    name: str
    units: str
    count: int
    minimum: float
    maximum: float
    mean: float
    value_status: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""

        return asdict(self)


@dataclass(frozen=True)
class PedestrianPriorExtractionReport:
    """Compact prior extraction report without raw trajectory samples."""

    schema_version: str
    source_id: str
    value_status: str
    evidence_boundary: str
    parameter_summaries: list[PriorParameterSummary]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""

        payload = asdict(self)
        payload["parameter_summaries"] = [summary.to_dict() for summary in self.parameter_summaries]
        return payload


def load_pedestrian_prior_trajectory_fixture(path: str | Path) -> dict[str, Any]:
    """Load a fixture/local trajectory file in YAML or JSON format.

    Returns:
        Parsed fixture payload.
    """

    fixture_path = Path(path)
    if not fixture_path.is_file():
        raise PedestrianPriorExtractionError(f"trajectory fixture not found: {fixture_path}")
    try:
        payload = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise PedestrianPriorExtractionError(f"invalid YAML/JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PedestrianPriorExtractionError("trajectory fixture must be a mapping")
    return dict(payload)


def extract_pedestrian_prior_report(
    payload: Mapping[str, Any],
    *,
    value_status: str = "proxy-placeholder",
) -> PedestrianPriorExtractionReport:
    """Extract compact pedestrian-prior summaries from trajectory records.

    Args:
        payload: Mapping with ``observations`` records containing
            ``pedestrian_id``, ``time``, ``x``, and ``y``.
        value_status: Status stamped on extracted values. Use
            ``proxy-placeholder`` for fixtures; reserve ``dataset-backed`` for
            separately staged and manifest-admitted external data.

    Returns:
        Compact extraction report containing summaries, not raw trajectories.
    """

    if value_status not in {"proxy-placeholder", "dataset-backed"}:
        raise PedestrianPriorExtractionError(
            "value_status must be 'proxy-placeholder' or 'dataset-backed'"
        )
    observations = _parse_observations(payload.get("observations"))
    bounds = _parse_bounds(payload.get("bounds"))
    source_id = str(payload.get("source_id") or "pedestrian_prior_fixture")
    provenance = _provenance(payload, value_status=value_status)

    speeds = _segment_speeds(observations)
    headings = _pedestrian_headings(observations)
    densities = _frame_densities(observations, bounds)
    distances = _interaction_distances(observations)
    stop_durations = _stop_yield_durations(observations)

    values_by_parameter = {
        "walking_speed": ("m/s", speeds),
        "crossing_angle": ("deg", headings),
        "density": ("ped/m^2", densities),
        "interaction_distance": ("m", distances),
        "stop_yield_timing": ("s", stop_durations),
    }
    summaries = [
        _summarize(name, units, values, value_status=value_status)
        for name, (units, values) in values_by_parameter.items()
    ]
    return PedestrianPriorExtractionReport(
        schema_version=PEDESTRIAN_PRIOR_EXTRACTION_REPORT_SCHEMA_VERSION,
        source_id=source_id,
        value_status=value_status,
        evidence_boundary=PRIOR_EXTRACTION_EVIDENCE_BOUNDARY,
        parameter_summaries=summaries,
        provenance=provenance,
    )


def extract_pedestrian_prior_report_from_file(
    path: str | Path,
    *,
    value_status: str = "proxy-placeholder",
) -> PedestrianPriorExtractionReport:
    """Load a trajectory fixture/local file and extract compact prior summaries.

    Returns:
        Compact extraction report containing summaries, not raw trajectories.
    """

    return extract_pedestrian_prior_report(
        load_pedestrian_prior_trajectory_fixture(path),
        value_status=value_status,
    )


def write_pedestrian_prior_extraction_report(
    report: PedestrianPriorExtractionReport,
    path: str | Path,
) -> None:
    """Write compact extraction report as deterministic JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_observations(raw_observations: Any) -> list[dict[str, float | str]]:
    if not isinstance(raw_observations, list) or not raw_observations:
        raise PedestrianPriorExtractionError("observations must be a non-empty list")
    observations: list[dict[str, float | str]] = []
    for index, raw in enumerate(raw_observations):
        if not isinstance(raw, Mapping):
            raise PedestrianPriorExtractionError(f"observations[{index}] must be a mapping")
        try:
            pedestrian_id = str(raw["pedestrian_id"])
            time = float(raw["time"])
            x = float(raw["x"])
            y = float(raw["y"])
        except KeyError as exc:
            raise PedestrianPriorExtractionError(
                f"observations[{index}] missing required field {exc.args[0]!r}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise PedestrianPriorExtractionError(
                f"observations[{index}] has non-numeric time/x/y"
            ) from exc
        observations.append(
            {
                "pedestrian_id": pedestrian_id,
                "time": time,
                "x": x,
                "y": y,
            }
        )
    return sorted(observations, key=lambda item: (str(item["pedestrian_id"]), float(item["time"])))


def _parse_bounds(raw_bounds: Any) -> tuple[float, float, float, float]:
    if raw_bounds is None:
        raise PedestrianPriorExtractionError("bounds are required for density extraction")
    if not isinstance(raw_bounds, Mapping):
        raise PedestrianPriorExtractionError("bounds must be a mapping")
    try:
        min_x = float(raw_bounds["min_x"])
        max_x = float(raw_bounds["max_x"])
        min_y = float(raw_bounds["min_y"])
        max_y = float(raw_bounds["max_y"])
    except KeyError as exc:
        raise PedestrianPriorExtractionError(
            f"bounds missing required field {exc.args[0]!r}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise PedestrianPriorExtractionError("bounds values must be numeric") from exc
    if max_x <= min_x or max_y <= min_y:
        raise PedestrianPriorExtractionError("bounds must have positive width and height")
    return (min_x, max_x, min_y, max_y)


def _group_by_pedestrian(
    observations: Iterable[Mapping[str, float | str]],
) -> dict[str, list[Mapping[str, float | str]]]:
    grouped: dict[str, list[Mapping[str, float | str]]] = {}
    for observation in observations:
        grouped.setdefault(str(observation["pedestrian_id"]), []).append(observation)
    return grouped


def _segment_speeds(observations: list[Mapping[str, float | str]]) -> list[float]:
    speeds: list[float] = []
    for track in _group_by_pedestrian(observations).values():
        ordered = sorted(track, key=lambda item: float(item["time"]))
        for start, end in pairwise(ordered):
            dt = float(end["time"]) - float(start["time"])
            if dt <= 0.0:
                raise PedestrianPriorExtractionError("pedestrian tracks must have increasing time")
            speeds.append(
                math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
                / dt
            )
    if not speeds:
        raise PedestrianPriorExtractionError("at least one two-point pedestrian track is required")
    return speeds


def _pedestrian_headings(observations: list[Mapping[str, float | str]]) -> list[float]:
    headings: list[float] = []
    for track in _group_by_pedestrian(observations).values():
        ordered = sorted(track, key=lambda item: float(item["time"]))
        if len(ordered) < 2:
            continue
        dx = float(ordered[-1]["x"]) - float(ordered[0]["x"])
        dy = float(ordered[-1]["y"]) - float(ordered[0]["y"])
        if dx == 0.0 and dy == 0.0:
            headings.append(0.0)
        else:
            headings.append(abs(math.degrees(math.atan2(dy, dx))))
    if not headings:
        raise PedestrianPriorExtractionError("at least one moving pedestrian heading is required")
    return headings


def _frame_densities(
    observations: list[Mapping[str, float | str]],
    bounds: tuple[float, float, float, float],
) -> list[float]:
    min_x, max_x, min_y, max_y = bounds
    area = (max_x - min_x) * (max_y - min_y)
    frame_counts: dict[float, int] = {}
    for observation in observations:
        x = float(observation["x"])
        y = float(observation["y"])
        if min_x <= x <= max_x and min_y <= y <= max_y:
            time = float(observation["time"])
            frame_counts[time] = frame_counts.get(time, 0) + 1
    if not frame_counts:
        raise PedestrianPriorExtractionError("no observations fall inside density bounds")
    return [count / area for count in frame_counts.values()]


def _interaction_distances(observations: list[Mapping[str, float | str]]) -> list[float]:
    by_time: dict[float, list[Mapping[str, float | str]]] = {}
    for observation in observations:
        by_time.setdefault(float(observation["time"]), []).append(observation)
    distances: list[float] = []
    for frame in by_time.values():
        if len(frame) < 2:
            continue
        for index, observation in enumerate(frame):
            nearest = min(
                math.hypot(
                    float(observation["x"]) - float(other["x"]),
                    float(observation["y"]) - float(other["y"]),
                )
                for other_index, other in enumerate(frame)
                if other_index != index
            )
            distances.append(nearest)
    if not distances:
        raise PedestrianPriorExtractionError(
            "at least one timestamp with two pedestrians is required for interaction distance"
        )
    return distances


def _stop_yield_durations(
    observations: list[Mapping[str, float | str]],
    *,
    speed_threshold_mps: float = 0.05,
) -> list[float]:
    durations: list[float] = []
    for track in _group_by_pedestrian(observations).values():
        ordered = sorted(track, key=lambda item: float(item["time"]))
        current_duration = 0.0
        for start, end in pairwise(ordered):
            dt = float(end["time"]) - float(start["time"])
            if dt <= 0.0:
                raise PedestrianPriorExtractionError("pedestrian tracks must have increasing time")
            speed = (
                math.hypot(
                    float(end["x"]) - float(start["x"]),
                    float(end["y"]) - float(start["y"]),
                )
                / dt
            )
            if speed <= speed_threshold_mps:
                current_duration += dt
            elif current_duration > 0.0:
                durations.append(current_duration)
                current_duration = 0.0
        if current_duration > 0.0:
            durations.append(current_duration)
    return durations or [0.0]


def _summarize(
    name: str,
    units: str,
    values: list[float],
    *,
    value_status: str,
) -> PriorParameterSummary:
    if name not in REQUIRED_PRIOR_PARAMETERS:
        raise PedestrianPriorExtractionError(f"unexpected prior parameter: {name}")
    if not values:
        raise PedestrianPriorExtractionError(f"no values extracted for {name}")
    return PriorParameterSummary(
        name=name,
        units=units,
        count=len(values),
        minimum=round(min(values), 6),
        maximum=round(max(values), 6),
        mean=round(sum(values) / len(values), 6),
        value_status=value_status,
    )


def _provenance(payload: Mapping[str, Any], *, value_status: str) -> dict[str, Any]:
    raw_provenance = payload.get("provenance", {})
    provenance = dict(raw_provenance) if isinstance(raw_provenance, Mapping) else {}
    provenance.setdefault(
        "source_kind", "fixture" if value_status == "proxy-placeholder" else "staged"
    )
    provenance.setdefault("raw_trajectory_storage", "not_stored_in_git")
    provenance.setdefault("claim_boundary", PRIOR_EXTRACTION_EVIDENCE_BOUNDARY)
    return provenance
