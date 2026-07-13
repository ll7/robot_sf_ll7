"""Collision-envelope / physical-footprint clearance-semantics diagnostic (issue #3207).

Reported social-navigation traces routinely conflate three *different* physical
quantities when they talk about "clearance" or "collision":

- **center-to-center distance** — the raw distance between the robot center and a
  pedestrian center; it says nothing about body size or planning footprint;
- **proxy-envelope surface clearance** — distance minus the sum of the *planning*
  radii (robot proxy footprint + pedestrian proxy radius). A negative value is an
  envelope *overlap*, which the planner treats as a breach even when the physical
  bodies never touch (e.g. a 1.00 m robot proxy + 0.40 m pedestrian radius at a
  1.37 m terminal center distance gives ~-0.03 m of envelope overlap);
- **geometric-body clearance** — distance minus the sum of the *physical* body
  radii; only a value <= 0 is an actual body contact.

This module converts that ambiguity into an explicit, testable contract. It
formally defines the distinct clearance quantities, evaluates them for a given
encounter geometry, and enumerates a bounded robot-proxy / pedestrian-radius
sweep plus a collision/near-miss threshold-sensitivity table so that cross-planner
conclusions can be checked for sensitivity to proxy-footprint choices.

It is deliberately pure and diagnostic. It does not run benchmark episodes, change
any frozen-release collision/near-miss metric semantics, or promote a planner
ranking. It only enumerates how a *fixed* geometric encounter would be classified
under different footprint and threshold assumptions.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FOOTPRINT_CLEARANCE_SCHEMA = "footprint-clearance-semantics.v1"
FOOTPRINT_CLEARANCE_CONFIG_KEY = "footprint_semantics"

CLAIM_BOUNDARY = (
    "footprint_clearance_diagnostic_not_benchmark_evidence: formalizes distinct clearance "
    "quantities and enumerates a bounded robot-proxy / pedestrian-radius sweep and a "
    "collision/near-miss threshold-sensitivity table. It also enumerates a threshold-value "
    "sweep that distinguishes threshold-induced class changes from footprint-induced class "
    "changes. It is diagnostic only, does not change frozen-release collision/near-miss "
    "metric semantics, does not run benchmark episodes, and does not establish a planner "
    "ranking, simulator realism, sim-to-real validity, or paper-facing benchmark evidence."
)

# The six formally distinct quantities the maintainer scope addition (2026-07-12) asks to
# separate. Kept as explicit constants so downstream code and reports reference one vocabulary.
CENTER_TO_CENTER_DISTANCE = "center_to_center_distance"
PROXY_ENVELOPE_SURFACE_CLEARANCE = "proxy_envelope_surface_clearance"
GEOMETRIC_BODY_CLEARANCE = "geometric_body_clearance"
OBSTACLE_CONTACT = "obstacle_contact"
PEDESTRIAN_CONTACT = "pedestrian_contact"
CONSERVATIVE_BUFFER_BREACH = "conservative_buffer_breach"

CLEARANCE_QUANTITY_DEFINITIONS: dict[str, str] = {
    CENTER_TO_CENTER_DISTANCE: (
        "Raw distance between the robot center and the pedestrian center; independent of any "
        "body-size or planning-footprint assumption."
    ),
    PROXY_ENVELOPE_SURFACE_CLEARANCE: (
        "Center distance minus the sum of the planning proxy radii (robot proxy footprint + "
        "pedestrian proxy radius). Negative values are envelope overlaps the planner treats as a "
        "breach even when physical bodies do not touch."
    ),
    GEOMETRIC_BODY_CLEARANCE: (
        "Center distance minus the sum of the physical body radii. Values <= the contact "
        "threshold are actual body contact."
    ),
    OBSTACLE_CONTACT: (
        "Physical body contact between the robot body surface and a static obstacle surface "
        "(distinct from a pedestrian encounter); true when the supplied obstacle body-surface "
        "distance is <= the contact threshold."
    ),
    PEDESTRIAN_CONTACT: (
        "Physical body contact between the robot and a pedestrian; true when geometric-body "
        "clearance is <= the contact threshold."
    ),
    CONSERVATIVE_BUFFER_BREACH: (
        "Proxy-envelope surface clearance falls below the conservative safety buffer; a "
        "conservative near-miss flag that does not imply physical contact."
    ),
}

# Ordered most-severe (closest) first. Used to classify a single encounter into one label.
ENCOUNTER_CLASSES = (
    "pedestrian_contact",
    "proxy_envelope_overlap",
    "near_miss",
    "conservative_buffer_breach",
    "clear",
)


@dataclass(frozen=True)
class ClearanceGeometry:
    """Robot and pedestrian planning-proxy and physical-body radii (metres)."""

    robot_proxy_radius_m: float
    pedestrian_radius_m: float
    robot_body_radius_m: float
    pedestrian_body_radius_m: float

    def __post_init__(self) -> None:
        """Validate radii are finite, non-negative, and body <= proxy."""
        for name in (
            "robot_proxy_radius_m",
            "pedestrian_radius_m",
            "robot_body_radius_m",
            "pedestrian_body_radius_m",
        ):
            _require_finite_non_negative(getattr(self, name), key=name)
        if self.robot_body_radius_m > self.robot_proxy_radius_m:
            raise ValueError("robot_body_radius_m must not exceed robot_proxy_radius_m")
        if self.pedestrian_body_radius_m > self.pedestrian_radius_m:
            raise ValueError("pedestrian_body_radius_m must not exceed pedestrian_radius_m")


@dataclass(frozen=True)
class ClearanceThresholds:
    """Collision / near-miss decision thresholds (metres)."""

    contact_threshold_m: float
    near_miss_threshold_m: float
    conservative_buffer_m: float

    def __post_init__(self) -> None:
        """Validate thresholds are finite and non-negative."""
        for name in ("contact_threshold_m", "near_miss_threshold_m", "conservative_buffer_m"):
            _require_finite_non_negative(getattr(self, name), key=name)
        if self.contact_threshold_m > self.near_miss_threshold_m:
            raise ValueError("contact_threshold_m must not exceed near_miss_threshold_m")
        if self.near_miss_threshold_m > self.conservative_buffer_m:
            raise ValueError("near_miss_threshold_m must not exceed conservative_buffer_m")


def evaluate_clearance(
    geometry: ClearanceGeometry,
    thresholds: ClearanceThresholds,
    *,
    center_to_center_distance_m: float,
    obstacle_body_surface_distance_m: float | None = None,
) -> dict[str, Any]:
    """Evaluate the six distinct clearance quantities for one encounter geometry.

    Args:
        geometry: Robot and pedestrian proxy and body radii.
        thresholds: Collision / near-miss decision thresholds.
        center_to_center_distance_m: Robot-center to pedestrian-center distance.
        obstacle_body_surface_distance_m: Optional robot-body-surface to obstacle-surface
            distance; when omitted the obstacle-contact quantity is reported as ``None``.

    Returns:
        Mapping keyed by the six clearance-quantity names plus the derived encounter class.
    """
    distance = _require_finite_non_negative(
        center_to_center_distance_m, key="center_to_center_distance_m"
    )
    proxy_clearance = distance - (geometry.robot_proxy_radius_m + geometry.pedestrian_radius_m)
    body_clearance = distance - (geometry.robot_body_radius_m + geometry.pedestrian_body_radius_m)
    pedestrian_contact = body_clearance <= thresholds.contact_threshold_m
    conservative_buffer_breach = proxy_clearance < thresholds.conservative_buffer_m
    if obstacle_body_surface_distance_m is None:
        obstacle_contact: bool | None = None
    else:
        obstacle_distance = _require_finite_non_negative(
            obstacle_body_surface_distance_m, key="obstacle_body_surface_distance_m"
        )
        obstacle_contact = obstacle_distance <= thresholds.contact_threshold_m
    return {
        CENTER_TO_CENTER_DISTANCE: distance,
        PROXY_ENVELOPE_SURFACE_CLEARANCE: proxy_clearance,
        GEOMETRIC_BODY_CLEARANCE: body_clearance,
        PEDESTRIAN_CONTACT: bool(pedestrian_contact),
        OBSTACLE_CONTACT: obstacle_contact,
        CONSERVATIVE_BUFFER_BREACH: bool(conservative_buffer_breach),
        "encounter_class": classify_encounter(
            proxy_envelope_surface_clearance_m=proxy_clearance,
            geometric_body_clearance_m=body_clearance,
            thresholds=thresholds,
        ),
    }


def classify_encounter(
    *,
    proxy_envelope_surface_clearance_m: float,
    geometric_body_clearance_m: float,
    thresholds: ClearanceThresholds,
) -> str:
    """Classify a single encounter into one escalating-severity label.

    Returns:
        One of :data:`ENCOUNTER_CLASSES`.
    """
    if geometric_body_clearance_m <= thresholds.contact_threshold_m:
        return "pedestrian_contact"
    if proxy_envelope_surface_clearance_m < 0.0:
        return "proxy_envelope_overlap"
    # Bands are ordered by proximity: the near-miss threshold is the tighter zone and is
    # checked before the (typically wider) conservative buffer.
    if proxy_envelope_surface_clearance_m <= thresholds.near_miss_threshold_m:
        return "near_miss"
    if proxy_envelope_surface_clearance_m < thresholds.conservative_buffer_m:
        return "conservative_buffer_breach"
    return "clear"


@dataclass(frozen=True)
class FootprintSweepSpec:
    """Validated footprint / clearance-semantics sweep specification."""

    nominal_geometry: ClearanceGeometry
    thresholds: ClearanceThresholds
    robot_proxy_radii_m: tuple[float, ...]
    pedestrian_radii_m: tuple[float, ...]
    encounter_center_distances_m: tuple[float, ...]
    rationale: str
    # Threshold sweep — optional bounded grid for threshold-sensitivity analysis.
    # When present, the manifest records how class labels change across threshold
    # combinations at each fixed geometry cell.
    contact_thresholds_m: tuple[float, ...] | None = None
    near_miss_thresholds_m: tuple[float, ...] | None = None
    conservative_buffers_m: tuple[float, ...] | None = None
    threshold_rationale: str = ""


def load_footprint_sweep_spec(config: Mapping[str, Any]) -> FootprintSweepSpec:
    """Read and validate the footprint-sweep metadata from a fidelity config.

    Fails closed: a missing or malformed ``footprint_semantics`` block raises
    :class:`ValueError` rather than silently enumerating a degenerate sweep.

    Returns:
        Validated :class:`FootprintSweepSpec`.
    """
    if not isinstance(config, Mapping):
        raise ValueError("fidelity config must be a mapping")
    block = config.get(FOOTPRINT_CLEARANCE_CONFIG_KEY)
    if block is None:
        raise ValueError(
            f"missing required {FOOTPRINT_CLEARANCE_CONFIG_KEY!r} block: footprint clearance "
            "sweep metadata is required and this diagnostic fails closed without it"
        )
    if not isinstance(block, Mapping):
        raise ValueError(f"{FOOTPRINT_CLEARANCE_CONFIG_KEY!r} block must be a mapping")
    if block.get("schema_version") != FOOTPRINT_CLEARANCE_SCHEMA:
        raise ValueError(
            f"{FOOTPRINT_CLEARANCE_CONFIG_KEY}.schema_version must be {FOOTPRINT_CLEARANCE_SCHEMA!r}"
        )

    nominal = _require_mapping(block, "nominal_geometry")
    geometry = ClearanceGeometry(
        robot_proxy_radius_m=_require_number(nominal, "robot_proxy_radius_m"),
        pedestrian_radius_m=_require_number(nominal, "pedestrian_radius_m"),
        robot_body_radius_m=_require_number(nominal, "robot_body_radius_m"),
        pedestrian_body_radius_m=_require_number(nominal, "pedestrian_body_radius_m"),
    )
    thresholds_block = _require_mapping(block, "thresholds")
    thresholds = ClearanceThresholds(
        contact_threshold_m=_require_number(thresholds_block, "contact_threshold_m"),
        near_miss_threshold_m=_require_number(thresholds_block, "near_miss_threshold_m"),
        conservative_buffer_m=_require_number(thresholds_block, "conservative_buffer_m"),
    )
    sweep = _require_mapping(block, "sweep")
    threshold_sweep = block.get("threshold_sweep")
    if isinstance(threshold_sweep, Mapping):
        if not threshold_sweep:
            raise ValueError(
                "threshold_sweep block is present but empty: provide threshold lists or omit the block"
            )
        contact = _require_optional_number_list(threshold_sweep, "contact_threshold_m")
        near_miss = _require_optional_number_list(threshold_sweep, "near_miss_threshold_m")
        conservative = _require_optional_number_list(threshold_sweep, "conservative_buffer_m")
        if contact is not None:
            _validate_thresholds_not_empty(contact, near_miss, conservative)
        threshold_rationale = str(threshold_sweep.get("rationale", ""))
        # Validate monotonic ordering: every contact <= every near_miss <= every conservative
        if contact is not None:
            _validate_threshold_monotonic(contact, near_miss, conservative, thresholds)
    else:
        contact = None
        near_miss = None
        conservative = None
        threshold_rationale = ""
    return FootprintSweepSpec(
        nominal_geometry=geometry,
        thresholds=thresholds,
        robot_proxy_radii_m=_require_number_list(sweep, "robot_proxy_radius_m"),
        pedestrian_radii_m=_require_number_list(sweep, "pedestrian_radius_m"),
        encounter_center_distances_m=_require_number_list(sweep, "encounter_center_distances_m"),
        rationale=str(block.get("rationale", "")),
        contact_thresholds_m=contact,
        near_miss_thresholds_m=near_miss,
        conservative_buffers_m=conservative,
        threshold_rationale=threshold_rationale,
    )


def enumerate_footprint_sweep(spec: FootprintSweepSpec) -> list[dict[str, Any]]:
    """Enumerate every (robot proxy radius x pedestrian radius x distance) cell.

    Body radii are held at the nominal geometry so the sweep isolates the effect of the
    *planning* proxy footprint on the encounter classification.

    Returns:
        Deterministic list of cells, each with its clearance evaluation.
    """
    cells: list[dict[str, Any]] = []
    for robot_proxy in spec.robot_proxy_radii_m:
        for pedestrian_radius in spec.pedestrian_radii_m:
            geometry = ClearanceGeometry(
                robot_proxy_radius_m=robot_proxy,
                pedestrian_radius_m=pedestrian_radius,
                robot_body_radius_m=spec.nominal_geometry.robot_body_radius_m,
                pedestrian_body_radius_m=spec.nominal_geometry.pedestrian_body_radius_m,
            )
            for distance in spec.encounter_center_distances_m:
                evaluation = evaluate_clearance(
                    geometry,
                    spec.thresholds,
                    center_to_center_distance_m=distance,
                )
                cells.append(
                    {
                        "robot_proxy_radius_m": robot_proxy,
                        "pedestrian_radius_m": pedestrian_radius,
                        "center_to_center_distance_m": distance,
                        "evaluation": evaluation,
                    }
                )
    return cells


def build_collision_threshold_sensitivity_table(spec: FootprintSweepSpec) -> list[dict[str, Any]]:
    """Group sweep cells by (pedestrian radius, distance) and flag proxy sensitivity.

    For each fixed pedestrian radius and encounter distance, the table records how the
    encounter classification changes across the swept robot-proxy radii. A row is
    ``proxy_radius_sensitive`` when at least two proxy radii produce different classes:
    that is the methodological result the maintainer scope addition asks for, since a
    sensitive row means a collision/near-miss conclusion depends on the proxy footprint
    choice rather than on physical contact.

    Returns:
        Deterministic list of table rows.
    """
    rows: list[dict[str, Any]] = []
    for pedestrian_radius in spec.pedestrian_radii_m:
        for distance in spec.encounter_center_distances_m:
            by_proxy: list[dict[str, Any]] = []
            for robot_proxy in spec.robot_proxy_radii_m:
                geometry = ClearanceGeometry(
                    robot_proxy_radius_m=robot_proxy,
                    pedestrian_radius_m=pedestrian_radius,
                    robot_body_radius_m=spec.nominal_geometry.robot_body_radius_m,
                    pedestrian_body_radius_m=spec.nominal_geometry.pedestrian_body_radius_m,
                )
                evaluation = evaluate_clearance(
                    geometry, spec.thresholds, center_to_center_distance_m=distance
                )
                by_proxy.append(
                    {
                        "robot_proxy_radius_m": robot_proxy,
                        "encounter_class": evaluation["encounter_class"],
                        "proxy_envelope_surface_clearance_m": evaluation[
                            PROXY_ENVELOPE_SURFACE_CLEARANCE
                        ],
                    }
                )
            classes = {entry["encounter_class"] for entry in by_proxy}
            rows.append(
                {
                    "pedestrian_radius_m": pedestrian_radius,
                    "center_to_center_distance_m": distance,
                    "geometric_body_clearance_m": (
                        distance
                        - (
                            spec.nominal_geometry.robot_body_radius_m
                            + spec.nominal_geometry.pedestrian_body_radius_m
                        )
                    ),
                    "classes_by_proxy_radius": by_proxy,
                    "distinct_class_count": len(classes),
                    "proxy_radius_sensitive": len(classes) > 1,
                }
            )
    return rows


def build_footprint_clearance_manifest(
    config: Mapping[str, Any],
    *,
    config_path: str,
    git_head: str = "unknown",
) -> dict[str, Any]:
    """Build the deterministic footprint / clearance-semantics diagnostic manifest.

    Returns:
        JSON-serializable manifest with definitions, sweep cells, threshold-sensitivity
        table, the required-output contract, and the diagnostic claim boundary.
    """
    spec = load_footprint_sweep_spec(config)
    cells = enumerate_footprint_sweep(spec)
    table = build_collision_threshold_sensitivity_table(spec)
    sensitive_rows = [row for row in table if row["proxy_radius_sensitive"]]
    threshold_rows = enumerate_threshold_sensitivity(spec)
    return {
        "schema_version": FOOTPRINT_CLEARANCE_SCHEMA,
        "issue": int(config.get("issue", 3207)),
        "study_id": str(config.get("study_id", "issue_3207_fidelity_sensitivity_v1")),
        "status": "footprint_clearance_diagnostic_only",
        "dry_run": True,
        "evidence_status": "not_benchmark_evidence",
        "claim_boundary": CLAIM_BOUNDARY,
        "config_path": config_path,
        "git_head": git_head,
        "rationale": spec.rationale,
        "clearance_quantity_definitions": copy.deepcopy(CLEARANCE_QUANTITY_DEFINITIONS),
        "encounter_classes": list(ENCOUNTER_CLASSES),
        "nominal_geometry": {
            "robot_proxy_radius_m": spec.nominal_geometry.robot_proxy_radius_m,
            "pedestrian_radius_m": spec.nominal_geometry.pedestrian_radius_m,
            "robot_body_radius_m": spec.nominal_geometry.robot_body_radius_m,
            "pedestrian_body_radius_m": spec.nominal_geometry.pedestrian_body_radius_m,
        },
        "thresholds": {
            "contact_threshold_m": spec.thresholds.contact_threshold_m,
            "near_miss_threshold_m": spec.thresholds.near_miss_threshold_m,
            "conservative_buffer_m": spec.thresholds.conservative_buffer_m,
        },
        "sweep": {
            "robot_proxy_radius_m": list(spec.robot_proxy_radii_m),
            "pedestrian_radius_m": list(spec.pedestrian_radii_m),
            "encounter_center_distances_m": list(spec.encounter_center_distances_m),
        },
        "cell_count": len(cells),
        "cells": cells,
        "threshold_sensitivity_table": table,
        "proxy_radius_sensitive_row_count": len(sensitive_rows),
        "threshold_sweep_rows": threshold_rows,
        "threshold_sensitive_row_count": (
            sum(1 for r in threshold_rows if r["threshold_sensitive"])
            if threshold_rows is not None
            else None
        ),
        "required_outputs": [
            "robot_proxy_radius_m",
            "pedestrian_radius_m",
            "center_to_center_distance_m",
            PROXY_ENVELOPE_SURFACE_CLEARANCE,
            GEOMETRIC_BODY_CLEARANCE,
            "encounter_class",
            "proxy_radius_sensitive",
            "threshold_sensitive",
        ],
    }


def write_footprint_clearance_manifest(manifest: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write a deterministic JSON footprint clearance diagnostic manifest.

    Returns:
        Path to the written JSON manifest.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "footprint_clearance_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def enumerate_threshold_sensitivity(spec: FootprintSweepSpec) -> list[dict[str, Any]] | None:
    """Enumerate threshold-sensitivity rows for every (contact x near-miss x buffer) combination.

    For each fixed geometry cell from the footprint sweep, tests how the encounter classification
    changes across threshold combinations. A row is marked ``threshold_sensitive`` when at least
    two threshold sets produce different classes at the same geometry. This isolates the question:
    does a collision/near-miss conclusion depend on the *threshold boundary choice* rather than
    the physical geometry?

    Returns:
        Deterministic list of threshold-sensitivity rows, or ``None`` when the spec carries no
        threshold sweep.
    """
    if spec.contact_thresholds_m is None:
        return None
    rows: list[dict[str, Any]] = []
    for robot_proxy in spec.robot_proxy_radii_m:
        for pedestrian_radius in spec.pedestrian_radii_m:
            for distance in spec.encounter_center_distances_m:
                geometry = ClearanceGeometry(
                    robot_proxy_radius_m=robot_proxy,
                    pedestrian_radius_m=pedestrian_radius,
                    robot_body_radius_m=spec.nominal_geometry.robot_body_radius_m,
                    pedestrian_body_radius_m=spec.nominal_geometry.pedestrian_body_radius_m,
                )
                body_clearance = distance - (
                    geometry.robot_body_radius_m + geometry.pedestrian_body_radius_m
                )
                by_threshold: list[dict[str, Any]] = []
                for contact_t in spec.contact_thresholds_m:
                    for near_miss_t in spec.near_miss_thresholds_m:
                        for buffer_t in spec.conservative_buffers_m:
                            # Skip threshold combos that violate monotonic ordering.
                            # The sweep defines independent axes for each threshold,
                            # but only valid (contact <= near_miss <= buffer) combos
                            # count for the sensitivity analysis.
                            if contact_t > near_miss_t or near_miss_t > buffer_t:
                                continue
                            tr = ClearanceThresholds(
                                contact_threshold_m=contact_t,
                                near_miss_threshold_m=near_miss_t,
                                conservative_buffer_m=buffer_t,
                            )
                            evaluation = evaluate_clearance(
                                geometry, tr, center_to_center_distance_m=distance
                            )
                            by_threshold.append(
                                {
                                    "contact_threshold_m": contact_t,
                                    "near_miss_threshold_m": near_miss_t,
                                    "conservative_buffer_m": buffer_t,
                                    "encounter_class": evaluation["encounter_class"],
                                    "proxy_envelope_surface_clearance_m": evaluation[
                                        PROXY_ENVELOPE_SURFACE_CLEARANCE
                                    ],
                                    "geometric_body_clearance_m": evaluation[
                                        GEOMETRIC_BODY_CLEARANCE
                                    ],
                                }
                            )
                classes = {entry["encounter_class"] for entry in by_threshold}
                rows.append(
                    {
                        "robot_proxy_radius_m": robot_proxy,
                        "pedestrian_radius_m": pedestrian_radius,
                        "center_to_center_distance_m": distance,
                        "geometric_body_clearance_m": body_clearance,
                        "classes_by_threshold": by_threshold,
                        "distinct_class_count": len(classes),
                        "threshold_sensitive": len(classes) > 1,
                    }
                )
    return rows


def _validate_thresholds_not_empty(
    contact: tuple[float, ...],
    near_miss: tuple[float, ...],
    conservative: tuple[float, ...],
) -> None:
    """All three threshold lists must be present when any is defined."""
    if near_miss is None or conservative is None:
        missing = []
        if near_miss is None:
            missing.append("near_miss_threshold_m")
        if conservative is None:
            missing.append("conservative_buffer_m")
        raise ValueError(f"threshold_sweep requires all three lists; missing: {', '.join(missing)}")


def _validate_threshold_monotonic(
    contact: tuple[float, ...],
    near_miss: tuple[float, ...] | None,
    conservative: tuple[float, ...] | None,
    nominal: ClearanceThresholds,
) -> None:
    """Threshold sweep grids must preserve monotonic ordering: max(contact) <= min(near_miss)
    <= min(conservative). Fails closed on violation."""
    if (
        not contact
        or (near_miss is not None and not near_miss)
        or (conservative is not None and not conservative)
    ):
        raise ValueError("threshold_sweep lists must be non-empty")
    max_contact = max(contact)
    if near_miss is not None:
        min_near = min(near_miss)
        if max_contact > min_near:
            raise ValueError(
                f"threshold_sweep ordering violated: max(contact_threshold_m)={max_contact} "
                f"> min(near_miss_threshold_m)={min_near}"
            )
    if conservative is not None:
        min_cons = min(conservative)
        ref_min = min(near_miss) if near_miss is not None else nominal.near_miss_threshold_m
        if ref_min > min_cons:
            raise ValueError(
                f"threshold_sweep ordering violated: min(near_miss_threshold_m)={ref_min} "
                f"> min(conservative_buffer_m)={min_cons}"
            )


def _require_finite_non_negative(value: Any, *, key: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be finite")
    if numeric < 0.0:
        raise ValueError(f"{key} must be non-negative")
    return numeric


def _require_mapping(block: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = block.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{FOOTPRINT_CLEARANCE_CONFIG_KEY}.{key} must be a mapping")
    return value


def _require_optional_number_list(block: Mapping[str, Any], key: str) -> tuple[float, ...] | None:
    """Return threshold list if key is present and valid, else None."""
    value = block.get(key)
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(
            f"threshold_sweep.{key} must be a non-empty list of finite non-negative numbers"
        )
    return tuple(_require_finite_non_negative(item, key=f"{key}[]") for item in value)


def _require_number(block: Mapping[str, Any], key: str) -> float:
    if key not in block:
        raise ValueError(f"missing required numeric field {key!r}")
    return _require_finite_non_negative(block[key], key=key)


def _require_number_list(block: Mapping[str, Any], key: str) -> tuple[float, ...]:
    value = block.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{FOOTPRINT_CLEARANCE_CONFIG_KEY}.sweep.{key} must be a non-empty list")
    return tuple(_require_finite_non_negative(item, key=f"{key}[]") for item in value)


__all__ = [
    "CENTER_TO_CENTER_DISTANCE",
    "CLAIM_BOUNDARY",
    "CLEARANCE_QUANTITY_DEFINITIONS",
    "CONSERVATIVE_BUFFER_BREACH",
    "ENCOUNTER_CLASSES",
    "FOOTPRINT_CLEARANCE_CONFIG_KEY",
    "FOOTPRINT_CLEARANCE_SCHEMA",
    "GEOMETRIC_BODY_CLEARANCE",
    "OBSTACLE_CONTACT",
    "PEDESTRIAN_CONTACT",
    "PROXY_ENVELOPE_SURFACE_CLEARANCE",
    "ClearanceGeometry",
    "ClearanceThresholds",
    "FootprintSweepSpec",
    "build_collision_threshold_sensitivity_table",
    "build_footprint_clearance_manifest",
    "classify_encounter",
    "enumerate_footprint_sweep",
    "enumerate_threshold_sensitivity",
    "evaluate_clearance",
    "load_footprint_sweep_spec",
    "write_footprint_clearance_manifest",
]
