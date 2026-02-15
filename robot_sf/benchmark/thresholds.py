"""Canonical threshold metadata helpers for benchmark metrics and reporting.

This module centralizes threshold-profile serialization so episode records and
aggregation reports can declare exactly which metric thresholds were used.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from robot_sf.benchmark.constants import COLLISION_DIST, COMFORT_FORCE_THRESHOLD, NEAR_MISS_DIST
from robot_sf.benchmark.errors import AggregationMetadataError

THRESHOLD_PROFILE_ID = "social_nav_thresholds_v1"


def default_threshold_profile() -> dict[str, Any]:
    """Return the canonical metric threshold profile for benchmark episodes."""
    return {
        "profile_id": THRESHOLD_PROFILE_ID,
        "collision_distance_m": float(COLLISION_DIST),
        "near_miss_distance_m": float(NEAR_MISS_DIST),
        "comfort_force_threshold": float(COMFORT_FORCE_THRESHOLD),
        "near_miss_definition": "collision_distance_m <= min_distance < near_miss_distance_m",
        "near_miss_speed_dependence": "disabled_distance_only",
        "candidate_speed_dependent_variants": [
            "relative_speed_weighted",
            "ttc_gated",
        ],
        "sources": {
            "constants": "robot_sf/benchmark/constants.py",
            "metrics": "robot_sf/benchmark/metrics.py",
            "spec": "docs/ped_metrics/metrics_spec.md",
        },
    }


def threshold_profile_signature(profile: dict[str, Any]) -> str:
    """Build a stable hash signature for a threshold profile.

    Returns:
        Hex digest of the profile payload.
    """
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_metric_parameters(*, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build metric-parameter metadata for episode/report payloads.

    Returns:
        Mapping containing threshold profile payload and deterministic signature.
    """
    threshold_profile = deepcopy(default_threshold_profile())
    if profile is not None:
        threshold_profile.update(profile)
    return {
        "threshold_profile": threshold_profile,
        "threshold_signature": threshold_profile_signature(threshold_profile),
    }


def ensure_metric_parameters(record: dict[str, Any]) -> None:
    """Attach canonical metric-parameter metadata when missing."""
    if isinstance(record.get("metric_parameters"), dict):
        return
    record["metric_parameters"] = build_metric_parameters()


def _extract_threshold_profile(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Extract profile from record, falling back to defaults when absent.

    Returns:
        Tuple of (profile, was_missing_on_record).
    """
    params = record.get("metric_parameters")
    if isinstance(params, dict):
        profile = params.get("threshold_profile")
        if isinstance(profile, dict):
            return deepcopy(profile), False
    return deepcopy(default_threshold_profile()), True


def validate_threshold_parameter_consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate threshold profile consistency across records for reporting.

    Returns:
        Aggregation-ready threshold profile metadata (profile/signature/counts).

    Raises:
        AggregationMetadataError: If multiple non-matching threshold profiles exist.
    """
    if not records:
        profile = default_threshold_profile()
        return {
            "threshold_profile": profile,
            "threshold_signature": threshold_profile_signature(profile),
            "missing_profile_records": 0,
            "explicit_profile_records": 0,
        }

    reference_profile: dict[str, Any] | None = None
    reference_episode: str | None = None
    missing_profile_records = 0
    explicit_profile_records = 0

    for rec in records:
        episode_id = rec.get("episode_id")
        episode_ref = str(episode_id) if episode_id is not None else None
        profile, is_missing = _extract_threshold_profile(rec)
        if is_missing:
            missing_profile_records += 1
        else:
            explicit_profile_records += 1

        if reference_profile is None:
            reference_profile = profile
            reference_episode = episode_ref
            continue
        if profile != reference_profile:
            raise AggregationMetadataError(
                "Episode records use inconsistent metric threshold profiles.",
                episode_id=episode_ref,
                missing_fields=("metric_parameters.threshold_profile",),
                advice=(
                    "Regenerate episodes with one threshold profile or split reports by profile. "
                    f"Reference episode: {reference_episode or 'unknown'}."
                ),
            )

    assert reference_profile is not None
    return {
        "threshold_profile": reference_profile,
        "threshold_signature": threshold_profile_signature(reference_profile),
        "missing_profile_records": missing_profile_records,
        "explicit_profile_records": explicit_profile_records,
    }


__all__ = [
    "THRESHOLD_PROFILE_ID",
    "build_metric_parameters",
    "default_threshold_profile",
    "ensure_metric_parameters",
    "threshold_profile_signature",
    "validate_threshold_parameter_consistency",
]
