"""Tests for group-space metadata plumbing into benchmark metrics (issue #3972).

Cover the additive integration into ``compute_all_metrics`` / post-processing,
the ``group_specs_from_map`` serializer, and the merged episode-metadata helper
used by the map runner.
"""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.group_space_metrics import group_specs_from_map
from robot_sf.benchmark.map_runner_episode import _episode_metadata_for_benchmark_metrics
from robot_sf.benchmark.metrics import (
    EpisodeData,
    compute_all_metrics,
    post_process_metrics,
)
from robot_sf.nav.map_config import SocialGroupDefinition


def _episode(robot_pos: np.ndarray, *, episode_metadata=None) -> EpisodeData:
    """Build a minimal single-pedestrian-free EpisodeData for metric tests."""
    robot_pos = np.asarray(robot_pos, dtype=float)
    peds_pos = np.zeros((robot_pos.shape[0], 1, 2))
    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=np.zeros_like(robot_pos),
        robot_acc=np.zeros_like(robot_pos),
        peds_pos=peds_pos,
        ped_forces=np.zeros_like(peds_pos),
        goal=np.array([100.0, 0.0]),
        dt=0.1,
        reached_goal_step=None,
        episode_metadata=episode_metadata,
    )


def _social_groups_metadata(centroid=(0.0, 0.0), radius=1.0):
    """Return episode metadata declaring one circular social group."""
    return {
        "social_groups": {
            "schema_version": "social-groups.v1",
            "groups": [
                {
                    "group_id": "g1",
                    "type": "conversation",
                    "members": ["ped_a"],
                    "formation": "circular",
                    "centroid": [float(centroid[0]), float(centroid[1])],
                    "radius": float(radius),
                    "o_space_polygon": None,
                }
            ],
        }
    }


def test_compute_all_metrics_without_groups_omits_group_space():
    """Default rows (no declared groups) carry no group-space keys or block."""
    robot_pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    raw = compute_all_metrics(_episode(robot_pos), horizon=10)

    assert "group_space_available" not in raw
    processed = post_process_metrics(raw, snqi_weights=None, snqi_baseline=None)
    assert "group_space" not in processed


def test_compute_all_metrics_with_groups_emits_block_and_flat_keys():
    """Declared groups produce flat keys and a schema-backed post-processed block."""
    # Robot passes through the o-space: first step is inside (dist 0), rest outside.
    robot_pos = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0]])
    raw = compute_all_metrics(
        _episode(robot_pos, episode_metadata=_social_groups_metadata()),
        horizon=10,
    )

    assert raw["group_space_available"] == 1.0
    assert raw["group_intrusion_episode_rate"] == 1.0
    assert raw["group_intrusion_time_ratio"] == 0.25
    assert raw["nearest_group_id"] == "g1"

    processed = post_process_metrics(raw, snqi_weights=None, snqi_baseline=None)
    block = processed["group_space"]
    assert block["schema_version"] == "group-space-metrics.v1"
    assert block["available"] is True
    assert block["group_count"] == 1
    assert block["metrics"]["group_intrusion_time_ratio"] == 0.25
    assert block["support"]["nearest_group_id"] == "g1"
    assert "claim_boundary" in block


def test_group_specs_from_map_serializes_definitions():
    """group_specs_from_map returns JSON-safe specs from social group objects."""

    class _StubMap:
        social_groups = [
            SocialGroupDefinition(
                group_id="conversation_a",
                type="conversation",
                members=["ped_a", "ped_b"],
                formation="circular",
                centroid=(5.0, 3.0),
                radius=1.2,
            )
        ]

    specs = group_specs_from_map(_StubMap())
    assert len(specs) == 1
    assert specs[0]["group_id"] == "conversation_a"
    assert specs[0]["centroid"] == [5.0, 3.0]
    assert specs[0]["members"] == ["ped_a", "ped_b"]


def test_episode_metadata_helper_injects_social_groups():
    """The merged helper adds social_groups from the map to episode metadata."""

    class _StubMap:
        social_groups = [
            SocialGroupDefinition(
                group_id="g1",
                type="conversation",
                members=["ped_a"],
                formation="circular",
                centroid=(1.0, 1.0),
                radius=1.0,
            )
        ]

    metadata = _episode_metadata_for_benchmark_metrics({}, _StubMap())
    assert metadata is not None
    payload = metadata["social_groups"]
    assert payload["schema_version"] == "social-groups.v1"
    assert payload["groups"][0]["group_id"] == "g1"


def test_episode_metadata_helper_returns_none_without_metadata_or_groups():
    """No signal metadata and no groups yields None (unchanged default behavior)."""

    class _EmptyMap:
        social_groups = []

    assert _episode_metadata_for_benchmark_metrics({}, _EmptyMap()) is None
    assert _episode_metadata_for_benchmark_metrics({}, None) is None
