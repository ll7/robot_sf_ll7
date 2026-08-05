"""Opt-in diagnostic surface for a closing-speed / time-to-collision (TTC) aware near-miss.

Background
----------
The canonical near-miss metric in :mod:`robot_sf.benchmark.metrics` is purely
distance-based: it counts steps whose minimum robot-pedestrian clearance falls
in ``[0, D_NEAR)``. That static proximity test treats a slow drift to ``D_NEAR``
the same as a fast head-on approach that decelerates near ``D_NEAR``, and it
never registers converging-but-not-yet-close encounters with a high closing
speed / small TTC -- arguably the more dangerous ones. GitHub issue #3700
proposes a closing-speed / TTC-aware variant alongside the distance metric.

Status: diagnostic-only, opt-in, additive
-----------------------------------------
This module stages an *opt-in, additive, diagnostic* surface only. It does
**not**:

- replace or modify the canonical distance-based ``near_misses`` metric,
- wire anything into SNQI or any scoring / ranking path,
- assert a calibrated threshold or any safety result.

The TTC threshold exposed here is an explicit, **uncalibrated diagnostic
placeholder** (the choice of ``t_thr`` and the TTC-count vs. severity-weighting
variant is ``decision-required`` per issue #3700). Treat outputs as
diagnostic-only, never as benchmark evidence.

Fail-closed input contract
--------------------------
A TTC-aware near-miss needs per-step relative *positions and velocities*, which
in turn require a valid timestep ``dt`` and at least two frames (pedestrian
velocities are derived by finite difference). :func:`near_miss_ttc_input_readiness`
validates those timing/velocity inputs and reports, fail-closed, exactly which
requirement is missing. :func:`compute_ttc_near_miss_diagnostic` refuses to emit
numbers (raises :class:`NearMissTtcInputError`) when the inputs are not ready,
rather than silently returning zeros that would read as "no near misses".

TTC convention
--------------
The closing geometry mirrors the existing :func:`robot_sf.benchmark.metrics.time_to_collision_min`
metric so this diagnostic stays consistent with the repository's TTC definition:

- relative velocity ``v_rel = v_robot - v_ped``,
- a pair is *approaching* when ``dot(v_rel, d_vec) > 0`` with ``d_vec`` pointing
  from the robot to the pedestrian (centre-to-centre distance decreasing),
- ``TTC = ||d_vec|| / ||v_rel||`` for approaching pairs, ``+inf`` otherwise,
- closing speed is the component of ``v_rel`` along the line of approach,
  ``dot(v_rel, d_vec) / ||d_vec||`` (positive when approaching).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

# Reuse the canonical pedestrian-velocity primitive so this diagnostic does not
# fork the finite-difference convention used by the benchmark metrics.
from robot_sf.benchmark.metrics import _compute_ped_velocities

if TYPE_CHECKING:
    from robot_sf.benchmark.metrics import EpisodeData
from robot_sf.errors import RobotSfError

# Uncalibrated diagnostic placeholder (decision-required, issue #3700). This is
# NOT a benchmark-calibrated threshold; it only gives the diagnostic a concrete
# default so the surface can be inspected. Callers should pass an explicit
# ``t_thr`` once a calibrated value is chosen.
DIAGNOSTIC_TTC_THRESHOLD_S: float = 2.0

# Additive encounter aggregation. This is deliberately separate from the
# existing timestep diagnostic above; changing this constant must not change
# any legacy ``near_miss_ttc__*`` output.
NEAR_MISS_ENCOUNTER_SCHEMA_VERSION = "near_miss_encounter.v1"
NEAR_MISS_ENCOUNTER_PROFILE_SCHEMA_VERSION = "NearMissEncounterProfile.v1"
_ENCOUNTER_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Minimum relative speed (m/s) treated as "moving" when projecting closing speed
# / computing TTC. Matches the epsilon used by ``time_to_collision_min``.
_MIN_RELATIVE_SPEED: float = 1e-9

# Minimum centre-to-centre distance (m) below which the line-of-approach
# direction is numerically undefined; closing speed is reported as 0.0 there.
_MIN_APPROACH_DISTANCE: float = 1e-9


class NearMissTtcInputError(RobotSfError, RuntimeError):
    """Raised when TTC near-miss inputs fail the fail-closed readiness contract.

    Carries the structured readiness report so callers can surface exactly which
    timing/velocity requirement was missing instead of a bare message.
    """

    def __init__(self, message: str, *, readiness: NearMissTtcReadiness | None = None) -> None:
        """Store the actionable message plus the structured readiness report."""
        super().__init__(message)
        self.readiness = readiness


class NearMissEncounterInputError(RobotSfError, ValueError):
    """Raised when encounter aggregation cannot establish an auditable trace."""


@dataclass(frozen=True)
class NearMissEncounterProfile:
    """Explicit, versioned thresholds and continuity rule for encounters.

    A profile is required by :func:`build_near_miss_encounter_report`. The
    function never supplies a threshold default because a threshold choice is
    a scientific decision outside this diagnostic surface.
    """

    profile_id: str
    qualification_rule: str
    continuity_gap_s: float
    distance_threshold_m: float | None = None
    ttc_threshold_s: float | None = None

    def __post_init__(self) -> None:
        """Validate the explicit profile contract."""
        profile_id = str(self.profile_id).strip()
        if not profile_id:
            raise NearMissEncounterInputError("profile_id must be non-empty")
        allowed_rules = {"distance", "ttc", "distance_or_ttc"}
        if self.qualification_rule not in allowed_rules:
            raise NearMissEncounterInputError(
                "qualification_rule must be one of distance, ttc, distance_or_ttc"
            )
        if self.distance_threshold_m is None and self.ttc_threshold_s is None:
            raise NearMissEncounterInputError(
                "profile must declare distance_threshold_m or ttc_threshold_s"
            )
        if self.qualification_rule == "distance" and self.distance_threshold_m is None:
            raise NearMissEncounterInputError(
                "distance qualification requires distance_threshold_m"
            )
        if self.qualification_rule == "ttc" and self.ttc_threshold_s is None:
            raise NearMissEncounterInputError("ttc qualification requires ttc_threshold_s")
        _require_positive_finite(self.continuity_gap_s, "continuity_gap_s")
        if self.distance_threshold_m is not None:
            _require_positive_finite(self.distance_threshold_m, "distance_threshold_m")
        if self.ttc_threshold_s is not None:
            _require_positive_finite(self.ttc_threshold_s, "ttc_threshold_s")

    def to_dict(self) -> dict[str, object]:
        """Return the profile as a JSON-safe mapping."""
        return {
            "schema_version": NEAR_MISS_ENCOUNTER_PROFILE_SCHEMA_VERSION,
            "profile_id": self.profile_id.strip(),
            "qualification_rule": self.qualification_rule,
            "continuity_gap_s": float(self.continuity_gap_s),
            "distance_threshold_m": (
                float(self.distance_threshold_m) if self.distance_threshold_m is not None else None
            ),
            "ttc_threshold_s": (
                float(self.ttc_threshold_s) if self.ttc_threshold_s is not None else None
            ),
            "units": {"distance": "m", "time": "s", "speed": "m/s"},
        }


@dataclass(frozen=True)
class NearMissEncounterSample:
    """One actor trace sample used by the additive encounter aggregator.

    ``clearance_m``, ``ttc_s``, ``closing_speed_mps``, and ``pet_s`` are
    optional because existing traces do not always provide every diagnostic
    field. Missing optional values are retained as ``None`` with an explicit
    missingness entry in the report. ``actor_id`` and ``timestamp_s`` are
    mandatory for deterministic segmentation.
    """

    actor_id: str
    timestamp_s: float
    clearance_m: float | None = None
    ttc_s: float | None = None
    closing_speed_mps: float | None = None
    pet_s: float | None = None
    contact: bool | None = None
    exposure_valid: bool = True
    dt_s: float | None = None
    unavailable_fields: tuple[str, ...] = field(default_factory=tuple)


def build_near_miss_encounter_report(
    samples: Sequence[NearMissEncounterSample | Mapping[str, object]],
    *,
    profile: NearMissEncounterProfile,
    source_commit: str,
    release_id: str,
    bundle_id: str,
    input_checksums: Mapping[str, str],
) -> dict[str, object]:
    """Build deterministic, diagnostic-only encounter records.

    Samples are grouped by actor and segmented using the caller-declared
    profile. A contact sample terminates the current encounter and is not
    silently converted into a near-miss sample. Missing optional metrics are
    retained as ``None`` and listed in ``missingness``.

    The function does not call or modify the canonical timestep metrics. It
    also does not infer a threshold, merge across actors, or fill missing
    velocity/TTC/PET values.

    Raises:
        NearMissEncounterInputError: If actor identity, timestamps, or
            provenance cannot be validated, or if duplicate actor timestamps
            make the ordering ambiguous.

    Returns:
        A JSON-safe diagnostic report with encounters, denominator, exclusions,
        missingness, and provenance.
    """
    if not isinstance(profile, NearMissEncounterProfile):
        raise NearMissEncounterInputError("profile must be NearMissEncounterProfile.v1")
    provenance = _normalise_encounter_provenance(
        source_commit=source_commit,
        release_id=release_id,
        bundle_id=bundle_id,
        input_checksums=input_checksums,
    )
    if not samples:
        raise NearMissEncounterInputError("at least one encounter sample is required")

    normalized = [
        _normalise_encounter_sample(value, index=index) for index, value in enumerate(samples)
    ]
    grouped: dict[str, list[NearMissEncounterSample]] = {}
    for sample in normalized:
        grouped.setdefault(sample.actor_id, []).append(sample)
    ordered_by_actor = {
        actor_id: _order_actor_samples(actor_id, actor_samples)
        for actor_id, actor_samples in sorted(grouped.items())
    }

    encounters: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    missingness: Counter[str] = Counter()
    qualifying_sample_count = 0
    valid_exposure_duration_s = 0.0
    for actor_id, actor_samples in ordered_by_actor.items():
        valid_exposure_duration_s += _valid_exposure_duration(
            actor_samples, continuity_gap_s=profile.continuity_gap_s
        )
        actor_encounters, actor_exclusions, actor_missingness, actor_qualifying = (
            _segment_actor_encounters(
                actor_id,
                actor_samples,
                profile=profile,
                encounter_offset=len([item for item in encounters if item["actor_id"] == actor_id]),
            )
        )
        encounters.extend(actor_encounters)
        exclusions.extend(actor_exclusions)
        missingness.update(actor_missingness)
        qualifying_sample_count += actor_qualifying

    required_exclusions = sum(
        1 for item in exclusions if item["reason"] == "missing_qualification_fields"
    )
    status = (
        "complete"
        if encounters
        else "unavailable"
        if required_exclusions
        else "no-qualifying-samples"
    )
    exclusion_counts = Counter(str(item["reason"]) for item in exclusions)
    return {
        "schema_version": NEAR_MISS_ENCOUNTER_SCHEMA_VERSION,
        "status": status,
        "evidence_status": "diagnostic-only",
        "claim_boundary": (
            "Temporal grouping of already-qualified trace samples. This is not a calibrated "
            "near-miss risk measure, independent safety event, collision probability, or "
            "real-world safety evidence."
        ),
        "profile": profile.to_dict(),
        "units": {
            "time": "s",
            "distance": "m",
            "speed": "m/s",
            "encounter_duration": "s",
            "valid_exposure_duration": "s",
        },
        "denominator": {
            "sample_unit": "trace_sample",
            "encounter_unit": "encounter",
            "input_sample_count": len(normalized),
            "actor_count": len(ordered_by_actor),
            "qualifying_sample_count": qualifying_sample_count,
            "encounter_count": len(encounters),
            "valid_exposure_duration_s": valid_exposure_duration_s,
        },
        "encounters": encounters,
        "exclusions": sorted(
            exclusions,
            key=lambda item: (
                str(item["actor_id"]),
                float(item["timestamp_s"]),
                int(item["source_index"]),
            ),
        ),
        "missingness": {
            "field_counts": dict(sorted(missingness.items())),
            "sample_exclusion_counts": dict(sorted(exclusion_counts.items())),
        },
        "provenance": provenance,
    }


def write_near_miss_encounter_report(
    report: Mapping[str, object],
    path: str,
) -> str:
    """Write a deterministic encounter report JSON file and return its path.

    Returns:
        The string path written.
    """
    target = str(path)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return target


def _normalise_encounter_sample(
    value: NearMissEncounterSample | Mapping[str, object],
    *,
    index: int,
) -> NearMissEncounterSample:
    """Normalize one public sample input and retain unavailable fields.

    Returns:
        A normalized sample with explicit unavailable-field names.
    """
    if isinstance(value, NearMissEncounterSample):
        raw: Mapping[str, object] = {
            "actor_id": value.actor_id,
            "timestamp_s": value.timestamp_s,
            "clearance_m": value.clearance_m,
            "ttc_s": value.ttc_s,
            "closing_speed_mps": value.closing_speed_mps,
            "pet_s": value.pet_s,
            "contact": value.contact,
            "exposure_valid": value.exposure_valid,
            "dt_s": value.dt_s,
        }
        inherited_unavailable = set(value.unavailable_fields)
    elif isinstance(value, Mapping):
        raw = value
        inherited_unavailable = set()
    else:
        raise NearMissEncounterInputError(f"sample[{index}] must be a mapping")

    actor_id = _required_sample_text(raw.get("actor_id"), f"sample[{index}].actor_id")
    timestamp_s = _required_sample_float(raw.get("timestamp_s"), f"sample[{index}].timestamp_s")
    unavailable = inherited_unavailable
    clearance_m = _optional_sample_float(
        raw.get("clearance_m"), "clearance_m", unavailable, nonnegative=True
    )
    ttc_s = _optional_sample_float(raw.get("ttc_s"), "ttc_s", unavailable, nonnegative=True)
    closing_speed_mps = _optional_sample_float(
        raw.get("closing_speed_mps"), "closing_speed_mps", unavailable, nonnegative=True
    )
    pet_s = _optional_sample_float(raw.get("pet_s"), "pet_s", unavailable, nonnegative=True)
    contact = _optional_sample_bool(raw.get("contact"), "contact", unavailable)
    exposure_valid = _sample_exposure_valid(raw.get("exposure_valid", True), unavailable)
    dt_s = _optional_sample_float(raw.get("dt_s"), "dt_s", unavailable, nonnegative=False)
    if raw.get("dt_s") is not None and dt_s is None:
        exposure_valid = False
    if dt_s is not None and dt_s <= 0.0:
        unavailable.add("dt_s")
        dt_s = None
        exposure_valid = False

    return NearMissEncounterSample(
        actor_id=actor_id,
        timestamp_s=timestamp_s,
        clearance_m=clearance_m,
        ttc_s=ttc_s,
        closing_speed_mps=closing_speed_mps,
        pet_s=pet_s,
        contact=contact,
        exposure_valid=exposure_valid,
        dt_s=dt_s,
        unavailable_fields=tuple(sorted(unavailable)),
    )


def _order_actor_samples(
    actor_id: str,
    samples: Sequence[NearMissEncounterSample],
) -> list[NearMissEncounterSample]:
    """Sort an actor trace by time and reject ambiguous timestamps.

    Returns:
        Strictly time-ordered actor samples.
    """
    ordered = sorted(samples, key=lambda sample: sample.timestamp_s)
    previous: float | None = None
    for sample in ordered:
        if previous is not None and sample.timestamp_s <= previous:
            raise NearMissEncounterInputError(
                f"actor {actor_id!r} has duplicate or non-increasing timestamps"
            )
        previous = sample.timestamp_s
    return ordered


def _valid_exposure_duration(
    samples: Sequence[NearMissEncounterSample],
    *,
    continuity_gap_s: float,
) -> float:
    """Sum observed valid trace intervals without crossing continuity gaps.

    Returns:
        The observed duration in seconds.
    """
    duration = 0.0
    for previous, current in pairwise(samples):
        delta = current.timestamp_s - previous.timestamp_s
        if previous.exposure_valid and current.exposure_valid and 0.0 < delta <= continuity_gap_s:
            duration += delta
    return duration


def _segment_actor_encounters(
    actor_id: str,
    samples: Sequence[NearMissEncounterSample],
    *,
    profile: NearMissEncounterProfile,
    encounter_offset: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], Counter[str], int]:
    """Segment one ordered actor trace into encounters.

    Returns:
        Encounters, explicit sample exclusions, missingness counts, and the
        number of qualifying samples.
    """
    encounters: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    missingness: Counter[str] = Counter()
    active: dict[str, object] | None = None
    qualifying_count = 0
    encounter_index = encounter_offset

    for source_index, sample in enumerate(samples):
        missingness.update(sample.unavailable_fields)
        active = _close_on_gap(active, sample, profile=profile, encounters=encounters)
        active, encounter_index, added_qualifying = _process_actor_sample(
            actor_id,
            sample,
            source_index=source_index,
            profile=profile,
            encounter_index=encounter_index,
            active=active,
            encounters=encounters,
            exclusions=exclusions,
        )
        qualifying_count += added_qualifying

    if active is not None:
        encounters.append(_finish_encounter(active, termination_reason="trace_end"))
    return encounters, exclusions, missingness, qualifying_count


def _close_on_gap(
    active: dict[str, object] | None,
    sample: NearMissEncounterSample,
    *,
    profile: NearMissEncounterProfile,
    encounters: list[dict[str, object]],
) -> dict[str, object] | None:
    """Close an active encounter when the continuity gap is exceeded.

    Returns:
        The unchanged active state, or ``None`` after a gap closure.
    """
    if (
        active is not None
        and sample.timestamp_s - float(active["last_timestamp_s"]) > profile.continuity_gap_s
    ):
        encounters.append(_finish_encounter(active, termination_reason="gap"))
        return None
    return active


def _process_actor_sample(
    actor_id: str,
    sample: NearMissEncounterSample,
    *,
    source_index: int,
    profile: NearMissEncounterProfile,
    encounter_index: int,
    active: dict[str, object] | None,
    encounters: list[dict[str, object]],
    exclusions: list[dict[str, object]],
) -> tuple[dict[str, object] | None, int, int]:
    """Process one sample and return updated state and qualifying count.

    Returns:
        Updated active encounter, next encounter index, and either zero or one
        qualifying sample added by this call.
    """
    if not sample.exposure_valid:
        if active is not None:
            encounters.append(_finish_encounter(active, termination_reason="invalid_exposure"))
        _append_sample_exclusion(
            exclusions,
            sample,
            source_index=source_index,
            reason="invalid_exposure",
        )
        return None, encounter_index, 0
    if sample.contact is True:
        if active is not None:
            encounters.append(
                _finish_encounter(
                    active,
                    termination_reason="contact",
                    contact_time_s=sample.timestamp_s,
                )
            )
        else:
            _append_sample_exclusion(
                exclusions,
                sample,
                source_index=source_index,
                reason="contact_without_active_encounter",
            )
        return None, encounter_index, 0

    qualifies, reason = _sample_qualifies(sample, profile)
    if not qualifies:
        if active is not None:
            encounters.append(
                _finish_encounter(active, termination_reason=reason or "non_qualifying")
            )
        if reason == "missing_qualification_fields":
            _append_sample_exclusion(
                exclusions,
                sample,
                source_index=source_index,
                reason=reason,
            )
        return None, encounter_index, 0

    if active is None:
        encounter_index += 1
        active = _start_encounter(
            actor_id,
            encounter_index,
            sample,
            source_index=source_index,
        )
        return active, encounter_index, 1
    delta = sample.timestamp_s - float(active["last_timestamp_s"])
    if not 0.0 < delta <= profile.continuity_gap_s:
        encounters.append(_finish_encounter(active, termination_reason="gap"))
        encounter_index += 1
        active = _start_encounter(
            actor_id,
            encounter_index,
            sample,
            source_index=source_index,
        )
        return active, encounter_index, 1
    _extend_encounter(active, sample, delta=delta, source_index=source_index)
    return active, encounter_index, 1


def _sample_qualifies(
    sample: NearMissEncounterSample,
    profile: NearMissEncounterProfile,
) -> tuple[bool, str | None]:
    """Apply only the explicitly selected profile rule to one sample.

    Returns:
        A qualification flag and an explicit exclusion reason when it does not
        qualify.
    """
    checks: list[bool] = []
    required_missing = 0
    if profile.qualification_rule in {"distance", "distance_or_ttc"}:
        if profile.distance_threshold_m is None:
            pass
        elif sample.clearance_m is None:
            required_missing += 1
        else:
            checks.append(sample.clearance_m < profile.distance_threshold_m)
    if profile.qualification_rule in {"ttc", "distance_or_ttc"}:
        if profile.ttc_threshold_s is None:
            pass
        elif sample.ttc_s is None:
            required_missing += 1
        else:
            checks.append(sample.ttc_s < profile.ttc_threshold_s)
    if any(checks):
        return True, None
    if not checks or required_missing == len(checks) + required_missing:
        return False, "missing_qualification_fields"
    return False, "non_qualifying"


def _start_encounter(
    actor_id: str,
    encounter_index: int,
    sample: NearMissEncounterSample,
    *,
    source_index: int,
) -> dict[str, object]:
    """Initialize internal encounter state from one qualifying sample.

    Returns:
        Mutable internal state for the active encounter.
    """
    return {
        "schema_version": NEAR_MISS_ENCOUNTER_SCHEMA_VERSION,
        "encounter_id": f"{actor_id}:encounter-{encounter_index:04d}",
        "actor_id": actor_id,
        "profile_id": None,
        "samples": [sample],
        "source_indices": [source_index],
        "start_time_s": sample.timestamp_s,
        "last_timestamp_s": sample.timestamp_s,
        "valid_exposure_duration_s": 0.0,
        "unavailable_fields": set(sample.unavailable_fields),
        "contact_observation_unavailable": sample.contact is None,
    }


def _extend_encounter(
    active: dict[str, object],
    sample: NearMissEncounterSample,
    *,
    delta: float,
    source_index: int,
) -> None:
    """Add one contiguous qualifying sample to internal state."""
    active["samples"].append(sample)
    active["source_indices"].append(source_index)
    active["last_timestamp_s"] = sample.timestamp_s
    active["valid_exposure_duration_s"] += delta
    active["unavailable_fields"].update(sample.unavailable_fields)
    active["contact_observation_unavailable"] = bool(
        active["contact_observation_unavailable"] or sample.contact is None
    )


def _finish_encounter(
    active: dict[str, object],
    *,
    termination_reason: str,
    contact_time_s: float | None = None,
) -> dict[str, object]:
    """Convert internal encounter state into a JSON-safe record.

    Returns:
        A versioned encounter record.
    """
    samples = active["samples"]
    start_time_s = float(active["start_time_s"])
    end_time_s = float(active["last_timestamp_s"])
    clearances = [sample.clearance_m for sample in samples if sample.clearance_m is not None]
    ttcs = [sample.ttc_s for sample in samples if sample.ttc_s is not None]
    closing_speeds = [
        sample.closing_speed_mps for sample in samples if sample.closing_speed_mps is not None
    ]
    pets = [sample.pet_s for sample in samples if sample.pet_s is not None]
    contact_terminated = termination_reason == "contact"
    contact_status = (
        "observed"
        if contact_terminated
        else "unavailable"
        if active["contact_observation_unavailable"]
        else "not-observed"
    )
    return {
        "schema_version": NEAR_MISS_ENCOUNTER_SCHEMA_VERSION,
        "encounter_id": active["encounter_id"],
        "actor_id": active["actor_id"],
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "duration_s": end_time_s - start_time_s,
        "minimum_clearance_m": min(clearances) if clearances else None,
        "minimum_ttc_s": min(ttcs) if ttcs else None,
        "maximum_closing_speed_mps": max(closing_speeds) if closing_speeds else None,
        "minimum_pet_s": min(pets) if pets else None,
        "sample_count": len(samples),
        "valid_exposure_duration_s": float(active["valid_exposure_duration_s"]),
        "termination_reason": termination_reason,
        "contact_terminated": contact_terminated,
        "contact_status": contact_status,
        "contact_time_s": contact_time_s,
        "unavailable_fields": sorted(active["unavailable_fields"]),
        "evidence_status": "diagnostic-only",
    }


def _append_sample_exclusion(
    exclusions: list[dict[str, object]],
    sample: NearMissEncounterSample,
    *,
    source_index: int,
    reason: str,
) -> None:
    """Append one deterministic, explicit sample exclusion."""
    exclusions.append(
        {
            "actor_id": sample.actor_id,
            "timestamp_s": sample.timestamp_s,
            "source_index": source_index,
            "reason": reason,
            "unavailable_fields": list(sample.unavailable_fields),
        }
    )


def _normalise_encounter_provenance(
    *,
    source_commit: str,
    release_id: str,
    bundle_id: str,
    input_checksums: Mapping[str, str],
) -> dict[str, object]:
    """Validate and normalize provenance required by the report.

    Returns:
        A normalized provenance mapping with an input-checksum digest.
    """
    identities = {
        "source_commit": source_commit,
        "release_id": release_id,
        "bundle_id": bundle_id,
    }
    normalized: dict[str, object] = {}
    for field_name, value in identities.items():
        text = str(value).strip()
        if not text:
            raise NearMissEncounterInputError(f"{field_name} must be non-empty")
        normalized[field_name] = text
    if not input_checksums:
        raise NearMissEncounterInputError("input_checksums must not be empty")
    checksums: dict[str, str] = {}
    for name, checksum in sorted(input_checksums.items()):
        name_text = str(name).strip()
        checksum_text = str(checksum).strip()
        if not name_text or _ENCOUNTER_SHA256_RE.fullmatch(checksum_text) is None:
            raise NearMissEncounterInputError(f"invalid input checksum for {name!r}")
        checksums[name_text] = checksum_text
    normalized["input_checksums"] = checksums
    normalized["input_checksum_digest"] = _encounter_json_digest(checksums)
    return normalized


def _required_sample_text(value: object, field_name: str) -> str:
    """Require a non-empty sample identity field.

    Returns:
        The stripped identity text.
    """
    text = str(value).strip() if value is not None else ""
    if not text:
        raise NearMissEncounterInputError(f"{field_name} must be non-empty")
    return text


def _required_sample_float(value: object, field_name: str) -> float:
    """Require a finite sample timestamp.

    Returns:
        The finite timestamp value.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NearMissEncounterInputError(f"{field_name} must be finite") from exc
    if not np.isfinite(number):
        raise NearMissEncounterInputError(f"{field_name} must be finite")
    return number


def _optional_sample_float(
    value: object,
    field_name: str,
    unavailable: set[str],
    *,
    nonnegative: bool,
) -> float | None:
    """Parse an optional finite measurement, recording unavailable values.

    Returns:
        The finite measurement, or ``None`` when unavailable.
    """
    if value is None:
        unavailable.add(field_name)
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        unavailable.add(field_name)
        return None
    if not np.isfinite(number) or (nonnegative and number < 0.0):
        unavailable.add(field_name)
        return None
    return number


def _optional_sample_bool(value: object, field_name: str, unavailable: set[str]) -> bool | None:
    """Parse an optional boolean observation.

    Returns:
        The boolean observation, or ``None`` when unavailable.
    """
    if value is None:
        unavailable.add(field_name)
        return None
    if not isinstance(value, bool):
        unavailable.add(field_name)
        return None
    return value


def _sample_exposure_valid(value: object, unavailable: set[str]) -> bool:
    """Parse the exposure validity flag without treating invalid data as valid.

    Returns:
        ``True`` only for an explicit valid flag.
    """
    if isinstance(value, bool):
        return value
    unavailable.add("exposure_valid")
    return False


def _require_positive_finite(value: float, field_name: str) -> None:
    """Require a finite positive profile quantity."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NearMissEncounterInputError(f"{field_name} must be finite and positive") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise NearMissEncounterInputError(f"{field_name} must be finite and positive")


def _encounter_json_digest(value: object) -> str:
    """Return a deterministic SHA-256 digest for JSON-safe provenance data."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class NearMissTtcReadiness:
    """Structured fail-closed readiness report for TTC near-miss inputs.

    Attributes
    ----------
    ready : bool
        True only when every timing/velocity input required to compute a
        TTC-aware near-miss is present and well-formed.
    reasons : tuple[str, ...]
        Human-readable reasons the inputs are not ready; empty when ``ready``.
    n_steps : int
        Number of trajectory frames (``T``) detected, or 0 when undetectable.
    n_peds : int
        Number of pedestrians (``K``) detected, or 0 when undetectable.
    dt : float
        Timestep value inspected (may be non-finite/non-positive when invalid).
    """

    ready: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    n_steps: int = 0
    n_peds: int = 0
    dt: float = float("nan")


@dataclass(frozen=True)
class NearMissTtcDecisionPacket:
    """Read-only TTC near-miss diagnostic decision packet.

    The packet summarizes whether the existing opt-in diagnostic can be
    evaluated for a trajectory and what the result can and cannot support. It
    intentionally carries no threshold recommendation and does not replace the
    canonical distance-based near-miss metric.
    """

    issue: str
    evidence_status: str
    diagnostic_status: str
    available_inputs: tuple[str, ...]
    unsupported_cases: tuple[str, ...]
    cannot_claim: tuple[str, ...]
    readiness: NearMissTtcReadiness
    diagnostic: dict[str, float | str] = field(default_factory=dict)
    threshold_s: float = DIAGNOSTIC_TTC_THRESHOLD_S

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe packet representation."""
        return {
            "issue": self.issue,
            "evidence_status": self.evidence_status,
            "diagnostic_status": self.diagnostic_status,
            "available_inputs": list(self.available_inputs),
            "unsupported_cases": list(self.unsupported_cases),
            "cannot_claim": list(self.cannot_claim),
            "readiness": _json_safe_value(asdict(self.readiness)),
            "diagnostic": _json_safe_value(dict(self.diagnostic)),
            "threshold_s": _json_safe_value(self.threshold_s),
        }


def _as_float_array(value: object) -> np.ndarray | None:
    """Return ``value`` as a float ndarray, or ``None`` when it cannot be coerced."""
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None


def _check_dt(data: EpisodeData, reasons: list[str]) -> float:
    """Validate the ``dt`` timing field, appending reasons; return the value seen.

    Returns:
        The ``dt`` value inspected, or NaN when it is missing/not a number.
    """
    try:
        dt = float(data.dt)
    except (TypeError, ValueError):
        reasons.append("dt is missing or not a number")
        return float("nan")
    if not np.isfinite(dt):
        reasons.append(f"dt must be finite (got {dt!r})")
    elif dt <= 0.0:
        reasons.append(f"dt must be strictly positive (got {dt!r})")
    return dt


def _check_robot_pos(robot_pos: np.ndarray | None, reasons: list[str]) -> int:
    """Validate ``robot_pos`` shape/length, appending reasons; return frame count.

    Returns:
        Detected number of frames ``T``, or 0 when the shape is invalid.
    """
    if robot_pos is None or robot_pos.ndim != 2 or robot_pos.shape[1] != 2:
        shape = None if robot_pos is None else robot_pos.shape
        reasons.append(f"robot_pos must be a (T, 2) array (got shape {shape})")
        return 0
    n_steps = int(robot_pos.shape[0])
    if n_steps < 2:
        reasons.append(f"robot_pos needs >= 2 frames to derive velocities (got T={n_steps})")
    return n_steps


def _check_robot_vel(
    robot_vel: np.ndarray | None, robot_pos: np.ndarray | None, reasons: list[str]
) -> None:
    """Validate ``robot_vel`` shape and frame consistency with ``robot_pos``."""
    if robot_vel is None or robot_vel.ndim != 2 or robot_vel.shape[1] != 2:
        shape = None if robot_vel is None else robot_vel.shape
        reasons.append(f"robot_vel must be a (T, 2) array (got shape {shape})")
        return
    if robot_pos is not None and robot_pos.ndim == 2 and robot_vel.shape[0] != robot_pos.shape[0]:
        reasons.append(
            "robot_vel frame count must match robot_pos "
            f"(robot_vel T={robot_vel.shape[0]} vs robot_pos T={robot_pos.shape[0]})"
        )


def _check_peds_pos(peds_pos: np.ndarray | None, n_steps: int, reasons: list[str]) -> int:
    """Validate ``peds_pos`` shape and frame consistency; return pedestrian count.

    Returns:
        Detected number of pedestrians ``K``, or 0 when the shape is invalid.
    """
    if peds_pos is None or peds_pos.ndim != 3 or peds_pos.shape[2] != 2:
        shape = None if peds_pos is None else peds_pos.shape
        reasons.append(f"peds_pos must be a (T, K, 2) array (got shape {shape})")
        return 0
    if n_steps and peds_pos.shape[0] != n_steps:
        reasons.append(
            "peds_pos frame count must match robot_pos "
            f"(peds_pos T={peds_pos.shape[0]} vs robot_pos T={n_steps})"
        )
    return int(peds_pos.shape[1])


def near_miss_ttc_input_readiness(data: EpisodeData) -> NearMissTtcReadiness:
    """Validate, fail-closed, the inputs required for a TTC-aware near-miss.

    The TTC-aware near-miss needs per-step relative positions and velocities.
    This checks the timing/velocity fields that make that derivation valid:

    - ``dt`` is finite and strictly positive (needed to derive pedestrian
      velocity from positions and to interpret TTC in seconds),
    - ``robot_pos`` is a ``(T, 2)`` array with at least two frames,
    - ``robot_vel`` is present and shaped like ``robot_pos`` (``(T, 2)``),
    - ``peds_pos`` is a ``(T, K, 2)`` array whose frame count matches the robot.

    A pedestrian-free episode (``K == 0``) is still *ready*: the inputs are valid
    and the diagnostic simply has no pairs to evaluate. Readiness is about the
    timing/velocity contract, not about whether any near miss occurred.

    Returns
    -------
    NearMissTtcReadiness
        ``ready=True`` with empty ``reasons`` when all inputs are valid;
        otherwise ``ready=False`` listing every failed requirement.
    """
    reasons: list[str] = []

    dt = _check_dt(data, reasons)
    robot_pos = _as_float_array(getattr(data, "robot_pos", None))
    robot_vel = _as_float_array(getattr(data, "robot_vel", None))
    peds_pos = _as_float_array(getattr(data, "peds_pos", None))

    n_steps = _check_robot_pos(robot_pos, reasons)
    _check_robot_vel(robot_vel, robot_pos, reasons)
    n_peds = _check_peds_pos(peds_pos, n_steps, reasons)

    return NearMissTtcReadiness(
        ready=not reasons,
        reasons=tuple(reasons),
        n_steps=n_steps,
        n_peds=n_peds,
        dt=dt,
    )


def compute_ttc_near_miss_diagnostic(
    data: EpisodeData,
    *,
    t_thr: float = DIAGNOSTIC_TTC_THRESHOLD_S,
) -> dict[str, float | str]:
    """Compute the opt-in, diagnostic-only TTC-aware near-miss surface.

    This is an *additive diagnostic*: it never touches the canonical
    distance-based ``near_misses`` metric and is not consumed by SNQI or any
    scoring path. It fails closed -- raising :class:`NearMissTtcInputError` --
    when the timing/velocity inputs are not ready, so missing data can never be
    misread as "no near misses".

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container (positions, robot velocity, ``dt``).
    t_thr : float, optional
        TTC threshold in seconds. A step counts as a TTC near-miss when its
        minimum projected TTC over all pedestrians is below ``t_thr``. Defaults
        to the uncalibrated :data:`DIAGNOSTIC_TTC_THRESHOLD_S` placeholder;
        callers should pass an explicit value once calibrated (issue #3700).

    Returns
    -------
    dict
        Diagnostic surface under ``near_miss_ttc__*`` keys:

        - ``near_miss_ttc__status``: ``"ok"``, ``"no-pedestrians"`` or
          ``"no-approaching-pairs"``,
        - ``near_miss_ttc__threshold_s``: the ``t_thr`` used,
        - ``near_miss_ttc__count``: number of steps whose minimum TTC < ``t_thr``,
        - ``near_miss_ttc__min_ttc_s``: smallest projected TTC (NaN if none),
        - ``near_miss_ttc__max_closing_speed_mps``: largest closing speed over
          approaching pairs (NaN if none),
        - ``near_miss_ttc__approaching_steps``: steps with >= 1 approaching pair,
        - ``near_miss_ttc__n_steps``: trajectory frames evaluated.

    Raises
    ------
    NearMissTtcInputError
        If the readiness contract fails, or ``t_thr`` is not finite and positive.
    """
    readiness = near_miss_ttc_input_readiness(data)
    if not readiness.ready:
        raise NearMissTtcInputError(
            "TTC near-miss inputs are not ready: " + "; ".join(readiness.reasons),
            readiness=readiness,
        )

    try:
        t_thr_value = float(t_thr)
    except (TypeError, ValueError) as exc:
        raise NearMissTtcInputError(f"t_thr must be a number (got {t_thr!r})") from exc
    if not np.isfinite(t_thr_value) or t_thr_value <= 0.0:
        raise NearMissTtcInputError(
            f"t_thr must be finite and strictly positive (got {t_thr_value!r})"
        )

    n_steps = readiness.n_steps
    base_result: dict[str, float | str] = {
        "near_miss_ttc__threshold_s": t_thr_value,
        "near_miss_ttc__count": 0.0,
        "near_miss_ttc__min_ttc_s": float("nan"),
        "near_miss_ttc__max_closing_speed_mps": float("nan"),
        "near_miss_ttc__approaching_steps": 0.0,
        "near_miss_ttc__n_steps": float(n_steps),
    }

    # No pedestrians: inputs are valid but there are no pairs to evaluate.
    if readiness.n_peds == 0:
        base_result["near_miss_ttc__status"] = "no-pedestrians"
        return base_result

    peds_pos = np.asarray(data.peds_pos, dtype=float)
    robot_pos = np.asarray(data.robot_pos, dtype=float)
    robot_vel = np.asarray(data.robot_vel, dtype=float)
    dt = readiness.dt

    ped_vels = _compute_ped_velocities(peds_pos, dt)  # (T-1, K, 2)
    if ped_vels.shape[0] == 0:
        base_result["near_miss_ttc__status"] = "no-approaching-pairs"
        return base_result

    # Align robot arrays to the (T-1) finite-difference grid, matching the
    # convention in ``time_to_collision_min``.
    robot_vel_aligned = robot_vel[1:]
    robot_pos_aligned = robot_pos[1:]
    peds_pos_aligned = peds_pos[1:]

    v_rel = robot_vel_aligned[:, None, :] - ped_vels  # (T-1, K, 2)
    d_vec = peds_pos_aligned - robot_pos_aligned[:, None, :]  # robot -> ped

    # dot(v_rel, d_vec) > 0 => centre-to-centre distance decreasing (approaching).
    dot_product = np.einsum("ijk,ijk->ij", v_rel, d_vec)
    v_rel_mag = np.linalg.norm(v_rel, axis=2)
    d_mag = np.linalg.norm(d_vec, axis=2)

    approaching = dot_product > 0.0
    valid = approaching & (v_rel_mag > _MIN_RELATIVE_SPEED)

    # Projected TTC for approaching, moving pairs; +inf elsewhere so a per-step
    # min over pedestrians ignores diverging/static pairs.
    ttc_matrix = np.full_like(d_mag, np.inf)
    ttc_matrix[valid] = d_mag[valid] / v_rel_mag[valid]

    # Closing speed along the line of approach (severity proxy), only where the
    # approach direction is numerically defined.
    closing_speed = np.zeros_like(d_mag)
    speed_defined = valid & (d_mag > _MIN_APPROACH_DISTANCE)
    closing_speed[speed_defined] = dot_product[speed_defined] / d_mag[speed_defined]

    step_min_ttc = ttc_matrix.min(axis=1)  # (T-1,)
    ttc_near_miss_steps = int(np.count_nonzero(step_min_ttc < t_thr_value))
    approaching_steps = int(np.count_nonzero(approaching.any(axis=1)))

    finite_ttc = ttc_matrix[np.isfinite(ttc_matrix)]
    min_ttc = float(finite_ttc.min()) if finite_ttc.size else float("nan")
    max_closing = float(closing_speed.max()) if np.any(speed_defined) else float("nan")

    base_result["near_miss_ttc__count"] = float(ttc_near_miss_steps)
    base_result["near_miss_ttc__min_ttc_s"] = min_ttc
    base_result["near_miss_ttc__max_closing_speed_mps"] = max_closing
    base_result["near_miss_ttc__approaching_steps"] = float(approaching_steps)
    base_result["near_miss_ttc__status"] = "ok" if approaching_steps else "no-approaching-pairs"
    return base_result


def build_ttc_near_miss_decision_packet(
    data: EpisodeData,
    *,
    t_thr: float = DIAGNOSTIC_TTC_THRESHOLD_S,
    issue: str = "#3808",
) -> NearMissTtcDecisionPacket:
    """Build a read-only TTC near-miss diagnostic decision packet.

    The packet consumes the issue #3700 diagnostic surface without mutating the
    trajectory or canonical benchmark metrics. Unsupported inputs are reported
    fail-closed instead of converted to a zero near-miss result.

    Returns
    -------
    NearMissTtcDecisionPacket
        Structured diagnostic-only packet for review handoff.
    """

    readiness = near_miss_ttc_input_readiness(data)
    available_inputs = _describe_available_ttc_inputs(readiness)
    cannot_claim = (
        "no canonical near-miss metric replacement",
        "no calibrated TTC threshold or severity weighting choice",
        "no planner comparison, benchmark ranking, or paper/dissertation claim",
    )

    if not readiness.ready:
        return NearMissTtcDecisionPacket(
            issue=issue,
            evidence_status="diagnostic-only",
            diagnostic_status="unsupported-inputs",
            available_inputs=available_inputs,
            unsupported_cases=tuple(readiness.reasons),
            cannot_claim=cannot_claim,
            readiness=readiness,
            threshold_s=t_thr,
        )

    diagnostic = compute_ttc_near_miss_diagnostic(data, t_thr=t_thr)
    status = str(diagnostic["near_miss_ttc__status"])
    unsupported_cases = _describe_unsupported_ttc_cases(status)

    return NearMissTtcDecisionPacket(
        issue=issue,
        evidence_status="diagnostic-only",
        diagnostic_status=status,
        available_inputs=available_inputs,
        unsupported_cases=unsupported_cases,
        cannot_claim=cannot_claim,
        readiness=readiness,
        diagnostic=diagnostic,
        threshold_s=float(diagnostic["near_miss_ttc__threshold_s"]),
    )


def render_ttc_near_miss_decision_packet_markdown(packet: NearMissTtcDecisionPacket) -> str:
    """Render a compact Markdown decision packet for review or issue handoff.

    Returns
    -------
    str
        Markdown packet text.
    """

    lines = [
        "# TTC Near-Miss Diagnostic Decision Packet",
        "",
        f"- Issue: `{packet.issue}`",
        f"- Evidence status: `{packet.evidence_status}`",
        f"- Diagnostic status: `{packet.diagnostic_status}`",
        f"- Threshold inspected: `{packet.threshold_s}` seconds",
        "",
        "## Available Diagnostic Inputs",
        *_bullet_lines(packet.available_inputs),
        "",
        "## Unsupported Cases",
        *_bullet_lines(packet.unsupported_cases),
        "",
        "## Cannot Claim Before Canonical Metric Change",
        *_bullet_lines(packet.cannot_claim),
    ]
    if packet.diagnostic:
        lines.extend(["", "## Diagnostic Values"])
        lines.extend(
            f"- `{key}`: `{value}`"
            for key, value in sorted(packet.diagnostic.items(), key=lambda item: item[0])
        )
    return "\n".join(lines) + "\n"


def _describe_available_ttc_inputs(readiness: NearMissTtcReadiness) -> tuple[str, ...]:
    """Describe the timing and trajectory inputs visible to the packet.

    Returns
    -------
    tuple[str, ...]
        Human-readable input availability lines.
    """

    inputs = [
        f"dt inspected: {readiness.dt}",
        f"trajectory frames inspected: {readiness.n_steps}",
        f"pedestrians inspected: {readiness.n_peds}",
    ]
    if readiness.ready:
        inputs.append("robot position, robot velocity, and pedestrian position arrays are usable")
    return tuple(inputs)


def _describe_unsupported_ttc_cases(status: str) -> tuple[str, ...]:
    """Describe cases this diagnostic cannot support as TTC near-miss evidence.

    Returns
    -------
    tuple[str, ...]
        Human-readable unsupported-case lines.
    """

    cases = [
        "static distance-only proximity remains covered only by canonical near_misses",
        "trajectory-free aggregate metrics cannot be reinterpreted as TTC evidence",
    ]
    if status == "no-pedestrians":
        cases.append("no pedestrian pairs were available for TTC evaluation")
    elif status == "no-approaching-pairs":
        cases.append("opening or non-converging pairs do not support TTC near-miss counts")
    return tuple(cases)


def _bullet_lines(items: tuple[str, ...]) -> list[str]:
    """Format tuple content as Markdown bullets.

    Returns
    -------
    list[str]
        Markdown bullet lines.
    """

    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]


def _json_safe_value(value: object) -> object:
    """Convert packet values to strict JSON-compatible primitives.

    Returns
    -------
    object
        Value with non-finite floats replaced by ``None``.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        float_value = float(value)
        return float_value if np.isfinite(float_value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    return value


__all__ = [
    "DIAGNOSTIC_TTC_THRESHOLD_S",
    "NEAR_MISS_ENCOUNTER_PROFILE_SCHEMA_VERSION",
    "NEAR_MISS_ENCOUNTER_SCHEMA_VERSION",
    "NearMissEncounterInputError",
    "NearMissEncounterProfile",
    "NearMissEncounterSample",
    "NearMissTtcDecisionPacket",
    "NearMissTtcInputError",
    "NearMissTtcReadiness",
    "build_near_miss_encounter_report",
    "build_ttc_near_miss_decision_packet",
    "compute_ttc_near_miss_diagnostic",
    "near_miss_ttc_input_readiness",
    "render_ttc_near_miss_decision_packet_markdown",
    "write_near_miss_encounter_report",
]
