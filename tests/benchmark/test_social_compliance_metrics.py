"""Tests for the diagnostic social-compliance episode and aggregate contract."""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.aggregate import compute_aggregates, flatten_metrics
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics
from robot_sf.benchmark.social_compliance import (
    SOCIAL_COMPLIANCE_SCHEMA_VERSION,
    build_social_compliance_episode_block,
)


def _episode(*, with_pedestrian: bool = True) -> EpisodeData:
    """Build a small native episode fixture."""
    peds = (
        np.asarray([[[0.5, 0.0]], [[2.0, 0.0]], [[0.5, 0.0]]], dtype=float)
        if with_pedestrian
        else np.empty((3, 0, 2), dtype=float)
    )
    return EpisodeData(
        robot_pos=np.zeros((3, 2), dtype=float),
        robot_vel=np.zeros((3, 2), dtype=float),
        robot_acc=np.zeros((3, 2), dtype=float),
        peds_pos=peds,
        ped_forces=np.ones_like(peds),
        goal=np.asarray([1.0, 0.0]),
        dt=0.5,
        reached_goal_step=2,
        robot_radius=0.1,
        ped_radius=0.1,
    )


def test_episode_block_computes_only_supported_comfort_family() -> None:
    """Native positions produce comfort exposure while reference families stay unavailable."""
    block = build_social_compliance_episode_block(_episode())

    assert block["schema_version"] == SOCIAL_COMPLIANCE_SCHEMA_VERSION
    metrics = block["metrics"]
    comfort = metrics["comfort_exposure_person_s"]
    assert comfort["status"] == "available"
    assert comfort["value"] == 1.0
    assert comfort["support_count"] == 3
    assert metrics["pedestrian_deviation_mean_m"]["status"] == "unavailable"
    assert metrics["flow_disruption_delay_s"]["status"] == "unavailable"
    assert metrics["legibility_progress_deficit_m"]["status"] == "unavailable"
    assert metrics["distributional_inconvenience_p90_p50_gap"]["status"] == "unavailable"


def test_empty_crowd_is_not_applicable_not_zero() -> None:
    """No pedestrian samples are not evidence of zero comfort exposure."""
    block = build_social_compliance_episode_block(_episode(with_pedestrian=False))

    assert block["metrics"]["comfort_exposure_person_s"]["status"] == "not_applicable"
    assert "value" not in block["metrics"]["comfort_exposure_person_s"]


def test_comfort_exposure_uses_surface_clearance() -> None:
    """Footprints count as exposed when their surfaces enter the comfort radius."""
    block = build_social_compliance_episode_block(_episode(), comfort_radius_m=0.35)

    comfort = block["metrics"]["comfort_exposure_person_s"]
    assert comfort["status"] == "available"
    assert comfort["value"] == 1.0


def test_invalid_geometry_fails_closed() -> None:
    """Invalid trajectory coordinates or radii cannot produce an available float."""
    invalid_positions = _episode()
    invalid_positions.peds_pos[1, 0, 0] = np.nan
    invalid_position_block = build_social_compliance_episode_block(invalid_positions)
    assert invalid_position_block["metrics"]["comfort_exposure_person_s"]["status"] == "unavailable"

    invalid_radius = _episode()
    invalid_radius.robot_radius = -0.1
    invalid_radius_block = build_social_compliance_episode_block(invalid_radius)
    assert invalid_radius_block["metrics"]["comfort_exposure_person_s"]["status"] == "unavailable"


def test_compute_metrics_emits_block_without_changing_existing_scalars() -> None:
    """The block is additive and existing scalar metrics remain available."""
    metrics = compute_all_metrics(_episode(), horizon=3)

    assert metrics["success"] == 1.0
    assert metrics["collisions"] == 0
    assert metrics["social_compliance"]["claim_class"] == "diagnostic_proxy"


def test_flatten_and_aggregate_preserve_status_support_and_values() -> None:
    """Aggregate output groups the side-channel under the contract namespace."""
    metrics = compute_all_metrics(_episode(), horizon=3)
    record = {
        "episode_id": "social-1",
        "scenario_id": "fixture",
        "seed": 1,
        "scenario_params": {"algo": "planner_a"},
        "metrics": metrics,
    }

    flat = flatten_metrics(record)
    assert flat["social_compliance.comfort_exposure_person_s"] == 1.0
    assert flat["social_compliance.comfort_exposure_person_s.status"] == "available"
    assert flat["social_compliance.comfort_exposure_person_s.support_count"] == 3

    aggregate = compute_aggregates([record])
    social = aggregate["planner_a"]["social_compliance"]
    comfort = social["metrics"]["comfort_exposure_person_s"]
    assert comfort["status_counts"] == {"available": 1}
    assert comfort["support_count"] == 3
    assert comfort["mean"] == 1.0


def test_aggregate_normalizes_legacy_social_rows() -> None:
    """Legacy rows without a social status remain unavailable with zero support."""
    record = {
        "episode_id": "legacy-social-1",
        "scenario_id": "fixture",
        "seed": 1,
        "scenario_params": {"algo": "planner_a"},
        "metrics": {
            "social_compliance": {
                "metrics": {
                    "comfort_exposure_person_s": {
                        "support_count": 4,
                    }
                }
            }
        },
    }
    aggregate = compute_aggregates(
        [record],
        group_by="scenario_params.algo",
    )

    comfort = aggregate["planner_a"]["social_compliance"]["metrics"]["comfort_exposure_person_s"]
    assert comfort["status_counts"] == {"unavailable": 1}
    assert comfort["support_count"] == 0
