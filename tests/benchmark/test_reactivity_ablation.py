"""Tests for the reactive-vs-replay reactivity-ablation quantifier (issue #3573)."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.reactivity_ablation import (
    REACTIVITY_ABLATION_SCHEMA,
    ReactivityContrast,
    assess_reactivity_ablation,
    reactivity_delta,
)


def _contrast(
    planner: str,
    *,
    reactive_coll: float,
    replay_coll: float,
    reactive_nm: float = 0.1,
    replay_nm: float = 0.1,
    reactive_sep: float = 0.5,
    replay_sep: float = 0.5,
) -> ReactivityContrast:
    """Build a paired reactive-vs-replay contrast for one planner."""
    return ReactivityContrast(
        planner=planner,
        reactive_collision_rate=reactive_coll,
        replay_collision_rate=replay_coll,
        reactive_near_miss_rate=reactive_nm,
        replay_near_miss_rate=replay_nm,
        reactive_min_separation_m=reactive_sep,
        replay_min_separation_m=replay_sep,
    )


def test_replay_flatters_when_it_underreports_collisions() -> None:
    """More collisions under reactive than replay means replay flatters the planner."""
    delta = reactivity_delta(_contrast("p", reactive_coll=0.20, replay_coll=0.05))

    assert delta["collision_delta"] == pytest.approx(0.15)
    assert delta["replay_flatters"] is True


def test_no_flattering_when_replay_is_not_safer() -> None:
    """If replay shows no fewer hazards and no more separation, it does not flatter."""
    delta = reactivity_delta(
        _contrast("p", reactive_coll=0.05, replay_coll=0.05, reactive_sep=0.5, replay_sep=0.5)
    )

    assert delta["replay_flatters"] is False


def test_lower_separation_under_replay_does_not_flatter() -> None:
    """Replay with *less* separation (separation_delta > 0) is not flattering on that axis."""
    delta = reactivity_delta(
        _contrast("p", reactive_coll=0.05, replay_coll=0.05, reactive_sep=0.6, replay_sep=0.4)
    )

    assert delta["min_separation_delta_m"] == pytest.approx(0.2)
    assert delta["replay_flatters"] is False


def test_ablation_reports_mean_inflation_and_flattered_planners() -> None:
    """The ablation must aggregate mean replay inflation and the flattered planners."""
    report = assess_reactivity_ablation(
        [
            _contrast("A", reactive_coll=0.20, replay_coll=0.05),  # flattered
            _contrast("B", reactive_coll=0.05, replay_coll=0.05),  # not flattered
        ]
    )

    assert report["schema_version"] == REACTIVITY_ABLATION_SCHEMA
    assert report["mean_replay_collision_inflation"] == pytest.approx((0.15 + 0.0) / 2)
    assert report["planners_flattered_by_replay"] == ["A"]


def test_rank_reactivity_sensitivity_is_detected() -> None:
    """A planner whose collision rank changes between conditions must be flagged."""
    # Reactive: A(0.10) safer than B(0.20) -> A rank 1.
    # Replay:   B(0.02) safer than A(0.08) -> B rank 1: the order flips.
    report = assess_reactivity_ablation(
        [
            _contrast("A", reactive_coll=0.10, replay_coll=0.08),
            _contrast("B", reactive_coll=0.20, replay_coll=0.02),
        ]
    )

    assert report["ranking_is_reactivity_sensitive"] is True
    assert set(report["rank_reactivity_sensitive_planners"]) == {"A", "B"}


def test_stable_ranking_is_not_flagged() -> None:
    """When the collision order is unchanged, no planner is reactivity-sensitive by rank."""
    report = assess_reactivity_ablation(
        [
            _contrast("A", reactive_coll=0.05, replay_coll=0.02),
            _contrast("B", reactive_coll=0.20, replay_coll=0.10),
        ]
    )

    assert report["ranking_is_reactivity_sensitive"] is False
    assert report["rank_reactivity_sensitive_planners"] == []


def test_empty_input_is_rejected() -> None:
    """An empty ablation cannot be summarized."""
    with pytest.raises(ValueError):
        assess_reactivity_ablation([])
