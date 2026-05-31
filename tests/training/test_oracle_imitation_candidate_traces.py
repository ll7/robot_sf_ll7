"""Tests for oracle-imitation candidate trace launch-packet selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.training.oracle_imitation_launch_packet import load_launch_packet
from scripts.training.collect_oracle_imitation_candidate_traces import (
    _parse_episode_id,
    build_split_scenarios,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml"


def test_parse_episode_id_requires_split() -> None:
    """Episode ids must preserve the requested launch-packet split."""
    assert _parse_episode_id("train__planner_sanity_simple__seed201", split="train") == (
        "planner_sanity_simple",
        201,
    )

    with pytest.raises(ValueError, match="split mismatch"):
        _parse_episode_id("validation__planner_sanity_simple__seed101", split="train")


def test_build_split_scenarios_uses_exact_packet_rows() -> None:
    """Train trace collection should use only the packet's declared train rows."""
    packet = load_launch_packet(PACKET)

    scenarios = build_split_scenarios(packet, split="train", repo_root=REPO_ROOT)

    assert [scenario["name"] for scenario in scenarios] == [
        "planner_sanity_simple",
        "classic_head_on_corridor_low",
        "classic_crossing_low",
        "classic_doorway_low",
        "classic_overtaking_low",
        "francis2023_following_human",
    ]
    assert [scenario["seeds"] for scenario in scenarios] == [
        [201],
        [202],
        [203],
        [204],
        [205],
        [206],
    ]
    assert scenarios[0]["metadata"]["oracle_imitation_episode_id"] == (
        "train__planner_sanity_simple__seed201"
    )


def test_build_validation_split_uses_validation_subset_only() -> None:
    """Validation trace collection should not inherit the full scenario source."""
    packet = load_launch_packet(PACKET)

    scenarios = build_split_scenarios(packet, split="validation", repo_root=REPO_ROOT)

    assert [scenario["name"] for scenario in scenarios] == [
        "planner_sanity_simple",
        "classic_crossing_low",
        "classic_doorway_low",
    ]
    assert [scenario["seeds"] for scenario in scenarios] == [[101], [102], [103]]
