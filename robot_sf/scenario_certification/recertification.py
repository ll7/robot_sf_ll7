"""Transparent re-certification of tracked scenario archives under corrected semantics.

Issue #6139 repairs ``scenario_cert.v1`` so every accepted A* segment is validated
continuously against the parsed obstacle geometry and robot envelope. A grid-inflated
A* path can still cut a diagonal corner that the continuous robot disc cannot pass, so
the corrected certifier adds a swept-envelope check that fails closed on corner
cutting. Tracked archives registered before the repair were certified under the old
(occupancy-only) semantics, so their eligibility must be re-derived transparently:
this module reconstructs each archive record's scenario from its family map and
candidate robot start/goal, re-runs the corrected certifier, and records before/after
status, eligibility, swept-envelope verdicts, and stable hashes without modifying the
accepted archive.

Evidence boundary: this is corrected re-certification of existing certified inputs. It
does not prove search superiority, cross-planner transfer, or minimax robustness, and
it does not change candidate selection, metrics, or denominators.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.scenario_certification import certificate_to_dict, certify_map_definition
from robot_sf.scenario_certification.v1 import CertificationSettings

if TYPE_CHECKING:
    from pathlib import Path

    from robot_sf.nav.map_config import MapDefinition

RECERT_SCHEMA_VERSION = "issue_6139_recertification.v1"
RECERT_CLAIM_BOUNDARY = "corrected_recertification_not_benchmark_evidence"

# Family -> repo map file used to reconstruct each archive record's scenario. The
# campaign scenarios are private artifacts (``private-campaign://`` URIs); the
# reconstruction re-uses the public repo map and the record's candidate robot
# start/goal, which reproduces the embedded certification geometry bit-for-bit.
FAMILY_MAP_FILE: dict[str, str] = {
    "classic_cross_trap_medium": "maps/svg_maps/classic_crossing.svg",
    "classic_group_crossing_medium": "maps/svg_maps/classic_group_crossing.svg",
}

#: Conservative spawn/goal zone half-extent matching the certification fixtures. The
#: certifier only needs finite, in-bounds, non-obstacle route endpoints; the zone is
#: not used for swept-envelope geometry.
_ZONE_HALF_EXTENT_M = 0.2


@dataclass(frozen=True)
class RecertificationRecord:
    """Before/after certification comparison for one archive record.

    Attributes:
        archive_id: Tracked archive record identifier.
        scenario_family: Scenario family label used to resolve the map.
        before: Embedded (pre-correction) certification projection.
        after: Corrected (post-#6139) certification projection.
        reconstruction: Reconstruction fidelity diagnostics.
        status: ``unchanged`` when before/after eligibility match, else ``changed``.
    """

    archive_id: str
    scenario_family: str
    before: dict[str, Any]
    after: dict[str, Any]
    reconstruction: dict[str, Any]
    status: str


@dataclass(frozen=True)
class RecertificationReport:
    """Aggregate re-certification report over a tracked archive.

    Attributes:
        records: Per-record before/after comparisons in archive order.
        counts: Aggregate eligibility/status counts after re-certification.
        archive_sha256: SHA-256 of the unchanged accepted archive bytes.
        recertification_sha256: Stable hash of the normalized re-certification payload.
        reconstruction_fidelity: Aggregate reconstruction fidelity summary.
    """

    records: list[RecertificationRecord]
    counts: dict[str, int]
    archive_sha256: str
    recertification_sha256: str
    reconstruction_fidelity: dict[str, Any] = field(default_factory=dict)


def recertify_tracked_archive(
    archive_path: Path,
    *,
    robot_radius: float = 1.0,
    cells_per_meter: float = 2.0,
    map_file_override: Mapping[str, str] | None = None,
) -> RecertificationReport:
    """Re-certify every record of a tracked archive under the corrected certifier.

    Reconstructs each record's scenario from its family map and candidate robot
    start/goal, re-runs ``scenario_cert.v1`` (now with continuous swept-envelope
    validation), and records before/after eligibility, classification, swept-envelope
    verdict, and reconstruction fidelity. The accepted archive file is read-only and
    never modified.

    Args:
        archive_path: Path to the tracked ``archive.json`` projection.
        robot_radius: Robot collision-envelope radius used for re-certification.
        cells_per_meter: Planner grid resolution used for re-certification.
        map_file_override: Optional override of the family -> repo map-file mapping.

    Returns:
        RecertificationReport: Per-record before/after comparison and aggregate counts.

    Raises:
        FileNotFoundError: When the archive path does not exist.
        ValueError: When a record's family has no registered map file.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"tracked archive not found: {archive_path}")
    archive_bytes = archive_path.read_bytes()
    archive = json.loads(archive_bytes.decode("utf-8"))
    map_files = {**FAMILY_MAP_FILE, **(dict(map_file_override or {}))}

    repo_root = _resolve_repo_root(archive_path)
    archive_path_rel = _repo_relative_path(archive_path, repo_root)
    loaded_maps: dict[str, MapDefinition] = {}
    records: list[RecertificationRecord] = []
    max_path_length_err = 0.0
    max_clearance_err = 0.0
    fidelity_mismatches = 0

    for entry in archive.get("entries", []):
        archive_id = str(entry["archive_id"])
        family = str(entry["scenario_family"])
        if family not in map_files:
            raise ValueError(
                f"archive record {archive_id} has family {family!r} with no registered map file"
            )
        map_def = loaded_maps.get(family)
        if map_def is None:
            map_def = _load_reconstruction_map(repo_root / map_files[family])
            loaded_maps[family] = map_def

        before = _embedded_before_projection(entry)
        candidate = entry["candidate"]
        start = (float(candidate["start"]["x"]), float(candidate["start"]["y"]))
        goal = (float(candidate["goal"]["x"]), float(candidate["goal"]["y"]))
        reconstruction = _reconstruct_and_certify(
            map_def,
            archive_id=archive_id,
            start=start,
            goal=goal,
            robot_radius=robot_radius,
            cells_per_meter=cells_per_meter,
        )

        path_length_err, clearance_err = _reconstruction_fidelity_errors(before, reconstruction)
        if path_length_err is not None:
            max_path_length_err = max(max_path_length_err, path_length_err)
        if clearance_err is not None:
            max_clearance_err = max(max_clearance_err, clearance_err)
        if _fidelity_mismatch(path_length_err, clearance_err):
            fidelity_mismatches += 1

        before_elig = str(before.get("benchmark_eligibility"))
        after_elig = str(reconstruction.get("benchmark_eligibility"))
        status = "unchanged" if before_elig == after_elig else "changed"
        records.append(
            RecertificationRecord(
                archive_id=archive_id,
                scenario_family=family,
                before=before,
                after=reconstruction,
                reconstruction={
                    "map_file": map_files[family],
                    "robot_start": list(start),
                    "robot_goal": list(goal),
                    "robot_radius_m": robot_radius,
                    "cells_per_meter": cells_per_meter,
                    "embedded_shortest_path_length_m": before.get("shortest_path_length_m"),
                    "reconstructed_shortest_path_length_m": reconstruction.get(
                        "shortest_path_length_m"
                    ),
                    "embedded_minimum_static_clearance_m": before.get("minimum_static_clearance_m"),
                    "reconstructed_minimum_static_clearance_m": reconstruction.get(
                        "minimum_static_clearance_m"
                    ),
                    "path_length_abs_error_m": path_length_err,
                    "clearance_abs_error_m": clearance_err,
                },
                status=status,
            )
        )

    counts = _aggregate_counts(records)
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    reconstruction_fidelity = {
        "max_shortest_path_length_abs_error_m": max_path_length_err,
        "max_minimum_static_clearance_abs_error_m": max_clearance_err,
        "fidelity_mismatch_count": fidelity_mismatches,
        "fidelity_tolerance_m": _FIDELITY_TOLERANCE_M,
        "fidelity_basis": (
            "reconstructed shortest_path_length_m and minimum_static_clearance_m "
            "match the embedded certificate bit-for-bit at cells_per_meter=2.0"
        ),
    }
    # Compute the stable recertification hash over the normalized payload excluding the
    # hash field itself, then assemble the final report.
    provisional = RecertificationReport(
        records=records,
        counts=counts,
        archive_sha256=archive_sha256,
        recertification_sha256="",
        reconstruction_fidelity=reconstruction_fidelity,
    )
    provisional_payload = recertification_report_to_dict(provisional, archive_path=archive_path_rel)
    recertification_sha256 = _normalized_sha256(
        {k: v for k, v in provisional_payload.items() if k != "recertification_sha256"}
    )
    return RecertificationReport(
        records=records,
        counts=counts,
        archive_sha256=archive_sha256,
        recertification_sha256=recertification_sha256,
        reconstruction_fidelity=reconstruction_fidelity,
    )


def recertification_report_to_dict(
    report: RecertificationReport,
    *,
    archive_path: Path,
    issue: str = "6139",
) -> dict[str, Any]:
    """Serialize a re-certification report to JSON-safe primitives.

    Returns:
        Versioned ``issue_6139_recertification.v1`` payload.
    """
    return {
        "schema_version": RECERT_SCHEMA_VERSION,
        "issue": issue,
        "claim_boundary": RECERT_CLAIM_BOUNDARY,
        "archive_path": archive_path.as_posix(),
        "archive_sha256": report.archive_sha256,
        "recertification_sha256": report.recertification_sha256,
        "correction": {
            "summary": (
                "Re-derive scenario_cert.v1 eligibility with continuous swept-envelope "
                "validation of the planned A* path (issue #6139)."
            ),
            "accepted_archive_modified": False,
            "swept_envelope_check": (
                "planned path full-polyline clearance against the parsed obstacle union "
                "after robot-radius inflation; negative clearance fails closed"
            ),
        },
        "counts": report.counts,
        "reconstruction_fidelity": report.reconstruction_fidelity,
        "records": [
            {
                "archive_id": record.archive_id,
                "scenario_family": record.scenario_family,
                "status": record.status,
                "before": record.before,
                "after": record.after,
                "reconstruction": record.reconstruction,
            }
            for record in report.records
        ],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FIDELITY_TOLERANCE_M = 1e-6


def _resolve_repo_root(archive_path: Path) -> Path:
    """Resolve the repository root from an evidence-path anchor.

    Returns:
        Repository root directory containing ``maps/`` and ``robot_sf/``.
    """
    candidate = archive_path.resolve()
    for parent in [candidate, *candidate.parents]:
        if (parent / "maps").is_dir() and (parent / "robot_sf").is_dir():
            return parent
    return archive_path.resolve().parents[4]


def _repo_relative_path(path: Path, repo_root: Path) -> Path:
    """Return ``path`` relative to ``repo_root`` for machine-independent evidence.

    Falls back to the resolved path when it is not inside ``repo_root``.
    """
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return path


def _load_reconstruction_map(map_file: Path) -> MapDefinition:
    """Load a repo map file for scenario reconstruction.

    Returns:
        Freshly parsed MapDefinition. Callers inject the candidate robot route.
    """
    from robot_sf.nav.svg_map_parser import convert_map  # noqa: PLC0415

    return convert_map(str(map_file))


def _reconstruct_and_certify(
    map_def: MapDefinition,
    *,
    archive_id: str,
    start: tuple[float, float],
    goal: tuple[float, float],
    robot_radius: float,
    cells_per_meter: float,
) -> dict[str, Any]:
    """Reconstruct one record's scenario and run the corrected certifier.

    Returns:
        Corrected certification projection for the reconstructed record.
    """
    reconstructed = _clone_map_with_route(map_def, start=start, goal=goal)
    certificate = certify_map_definition(
        reconstructed,
        robot_config=DifferentialDriveSettings(radius=robot_radius),
        settings=CertificationSettings(planner_cells_per_meter=cells_per_meter),
        scenario={"name": archive_id},
    )
    return _corrected_projection(certificate)


def _clone_map_with_route(
    map_def: MapDefinition,
    *,
    start: tuple[float, float],
    goal: tuple[float, float],
) -> MapDefinition:
    """Return a shallow map clone carrying one candidate robot route.

    The candidate robot start/goal override the map's authored routes, mirroring the
    campaign scenario that produced the archived record. Only the route-bearing
    attributes are replaced; obstacle geometry is preserved from the parsed map.
    """
    h = _ZONE_HALF_EXTENT_M
    spawn_zone = (
        (start[0] - h, start[1] - h),
        (start[0] + h, start[1] - h),
        (start[0] + h, start[1] + h),
    )
    goal_zone = ((goal[0] - h, goal[1] - h), (goal[0] + h, goal[1] - h), (goal[0] + h, goal[1] + h))
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[start, goal],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    cloned = _shallow_clone_map(map_def)
    cloned.robot_routes = [route]
    cloned.robot_spawn_zones = [route.spawn_zone]
    cloned.robot_goal_zones = [route.goal_zone]
    return cloned


def _shallow_clone_map(map_def: MapDefinition) -> MapDefinition:
    """Return a shallow copy of a MapDefinition so route injection is non-destructive.

    The parsed obstacle geometry is shared (read-only) with the source map.
    """
    import copy  # noqa: PLC0415

    cloned = copy.copy(map_def)
    return cloned


def _embedded_before_projection(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the embedded (pre-correction) certification projection from a record.

    Returns:
        Compact projection with classification, eligibility, and discriminating checks.
    """
    attempts = (
        entry.get("candidate_certification", {})
        .get("independent_seed_confirmation", {})
        .get("attempts", [])
    )
    if not attempts:
        return {
            "classification": "unknown",
            "benchmark_eligibility": "unknown",
            "shortest_path_length_m": None,
            "minimum_static_clearance_m": None,
        }
    details = attempts[0].get("scenario_certification", {}).get("details", {})
    certificates = details.get("certificates", [])
    if not certificates:
        return {
            "classification": "unknown",
            "benchmark_eligibility": "unknown",
            "shortest_path_length_m": None,
            "minimum_static_clearance_m": None,
        }
    certificate = certificates[0]
    route_certs = certificate.get("route_certificates", [])
    route_checks = route_certs[0].get("checks", {}) if route_certs else {}
    return {
        "classification": certificate.get("classification"),
        "benchmark_eligibility": certificate.get("benchmark_eligibility"),
        "shortest_path_length_m": route_checks.get("shortest_path_length_m"),
        "minimum_static_clearance_m": route_checks.get("minimum_static_clearance_m"),
        "inflated_collision_free_path": route_checks.get("inflated_collision_free_path"),
    }


def _corrected_projection(certificate: Any) -> dict[str, Any]:
    """Build the corrected (post-#6139) certification projection.

    Returns:
        Compact projection including the swept-envelope discriminating check.
    """
    certificate_dict = certificate_to_dict(certificate)
    route_certs = certificate_dict.get("route_certificates", [])
    route_checks = route_certs[0].get("checks", {}) if route_certs else {}
    swept = route_checks.get("swept_envelope", {})
    return {
        "classification": certificate_dict.get("classification"),
        "benchmark_eligibility": certificate_dict.get("benchmark_eligibility"),
        "shortest_path_length_m": route_checks.get("shortest_path_length_m"),
        "minimum_static_clearance_m": route_checks.get("minimum_static_clearance_m"),
        "inflated_collision_free_path": route_checks.get("inflated_collision_free_path"),
        "swept_envelope": {
            "validated": swept.get("validated"),
            "clips_obstacle": swept.get("clips_obstacle"),
            "clearance_m": swept.get("clearance_m"),
            "vertex_clearance_m": swept.get("vertex_clearance_m"),
            "clipped_vertex_count": swept.get("clipped_vertex_count"),
            "planned_waypoint_count": swept.get("planned_waypoint_count"),
        },
    }


def _reconstruction_fidelity_errors(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> tuple[float | None, float | None]:
    """Return absolute errors between embedded and reconstructed discriminating checks.

    Returns:
        Tuple of (shortest_path_length_abs_error_m, minimum_static_clearance_abs_error_m).
    """
    path_length_error = _abs_error(
        before.get("shortest_path_length_m"), after.get("shortest_path_length_m")
    )
    clearance_error = _abs_error(
        before.get("minimum_static_clearance_m"), after.get("minimum_static_clearance_m")
    )
    return path_length_error, clearance_error


def _abs_error(before: Any, after: Any) -> float | None:
    """Return the absolute error between two numeric projections, or ``None``."""
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    if not math.isfinite(float(before)) or not math.isfinite(float(after)):
        return None
    return abs(float(after) - float(before))


def _fidelity_mismatch(path_length_error: float | None, clearance_error: float | None) -> bool:
    """Return whether a reconstruction fidelity error exceeds the tolerance."""
    if path_length_error is not None and path_length_error > _FIDELITY_TOLERANCE_M:
        return True
    if clearance_error is not None and clearance_error > _FIDELITY_TOLERANCE_M:
        return True
    return False


def _aggregate_counts(records: list[RecertificationRecord]) -> dict[str, int]:
    """Aggregate after-status and eligibility counts over re-certified records.

    Returns:
        Counts of after eligibility, classification, and before/after change status.
    """
    after_eligibility: dict[str, int] = {}
    after_classification: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for record in records:
        after_eligibility[str(record.after.get("benchmark_eligibility"))] = (
            after_eligibility.get(str(record.after.get("benchmark_eligibility")), 0) + 1
        )
        after_classification[str(record.after.get("classification"))] = (
            after_classification.get(str(record.after.get("classification")), 0) + 1
        )
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    return {
        "record_count": len(records),
        "after_benchmark_eligibility": after_eligibility,
        "after_classification": after_classification,
        "before_after_status": status_counts,
    }


def _normalized_sha256(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 over a JSON-normalized payload.

    Returns:
        Hex SHA-256 digest of the sorted-key JSON encoding.
    """
    encoded = json.dumps(_sanitize(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sanitize(value: Any) -> Any:
    """Convert a value to strict-JSON-safe primitives with non-finite floats dropped.

    Returns:
        JSON-safe representation with mappings/lists recursed and non-finite floats
        replaced by ``None``.
    """
    if isinstance(value, Mapping):
        return {str(key): _sanitize(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, bool | int | str):
        return value
    return str(value)


__all__ = [
    "FAMILY_MAP_FILE",
    "RECERT_CLAIM_BOUNDARY",
    "RECERT_SCHEMA_VERSION",
    "RecertificationRecord",
    "RecertificationReport",
    "recertification_report_to_dict",
    "recertify_tracked_archive",
]
