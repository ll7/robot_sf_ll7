"""Tests for the issue #6969 Stage A parameter screen."""

from __future__ import annotations

from robot_sf.research.lane_formation_parameter_screen import (
    PARAMETER_BOUNDS,
    build_space_filling_profiles,
    run_parameter_screen,
)
from robot_sf.research.lane_formation_reference import ReferenceProtocol


def test_space_filling_profiles_are_deterministic_bounded_and_anchor_preserving():
    first = build_space_filling_profiles(n_profiles=3, seed=17)
    second = build_space_filling_profiles(n_profiles=3, seed=17)

    assert [profile.as_dict() for profile in first] == [profile.as_dict() for profile in second]
    assert [profile.profile_id for profile in first[-2:]] == [
        "anchor_released_default",
        "anchor_literature_typical",
    ]
    for profile in first[:-2]:
        values = profile.as_dict()
        for name, (lower, upper) in PARAMETER_BOUNDS.items():
            assert lower <= values[name] <= upper


def test_parameter_screen_smoke_is_native_and_non_ranking():
    payload = run_parameter_screen(
        protocol=ReferenceProtocol(
            length_m=10.0,
            half_width_m=2.0,
            n_pedestrians=6,
            warmup_steps=3,
            observation_steps=8,
        ),
        seeds=[7],
        n_profiles=1,
        profile_seed=17,
        sampling_strides=[1],
    )

    assert payload["schema_version"] == "lane_formation_parameter_screen.v1"
    assert payload["manifest"]["stage"] == "A"
    assert payload["manifest"]["execution_policy"]["row_count"] == 3
    assert len(payload["summary"]) == 3
    assert {row["execution"]["execution_mode"] for row in payload["rows"]} == {"native"}
    assert all(
        summary["selection_policy"] == "no_response_dependent_selection_in_stage_a"
        for summary in payload["summary"]
    )
