"""Cooperative multi-agent semantics for the aggregated_time metric (issue #8518)."""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.metrics import EpisodeData, aggregated_time


def _make_episode(dt: float = 0.1) -> EpisodeData:
    """Build a minimal single-robot episode container for metric tests."""
    steps = 10
    return EpisodeData(
        robot_pos=np.zeros((steps, 2)),
        robot_vel=np.zeros((steps, 2)),
        robot_acc=np.zeros((steps, 2)),
        peds_pos=np.zeros((steps, 0, 2)),
        ped_forces=np.zeros((steps, 0, 2)),
        goal=np.array([5.0, 0.0]),
        dt=dt,
        reached_goal_step=None,
    )


def test_none_preserves_single_robot_time_to_goal() -> None:
    """cooperative_agents=None must return the existing time_to_goal result."""
    episode = _make_episode()
    episode.reached_goal_step = 8

    assert aggregated_time(episode) == 8 * 0.1

    episode.reached_goal_step = None
    assert np.isnan(aggregated_time(episode))


def test_new_field_preserves_legacy_positional_optional_fields() -> None:
    """Appending the cooperative field must not shift the public dataclass layout."""
    force_grid = {"X": np.zeros((1, 1))}
    obstacles = np.zeros((1, 2))
    other_agents = np.zeros((1, 1, 2))
    metadata = {"source": "legacy"}
    episode = EpisodeData(
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        np.zeros((1, 0, 2)),
        np.zeros((1, 0, 2)),
        np.zeros(2),
        0.1,
        3,
        force_grid,
        obstacles,
        other_agents,
        0.3,
        0.4,
        metadata,
    )

    assert episode.reached_goal_step == 3
    assert episode.force_field_grid is force_grid
    assert episode.obstacles is obstacles
    assert episode.other_agents_pos is other_agents
    assert episode.robot_radius == 0.3
    assert episode.ped_radius == 0.4
    assert episode.episode_metadata is metadata
    assert episode.cooperative_goal_steps is None


def test_subset_returns_maximum_completion_time() -> None:
    """A requested subset returns the maximum per-agent completion time in seconds."""
    episode = _make_episode()
    episode.cooperative_goal_steps = {0: 10, 1: 30, 2: 20}

    assert aggregated_time(episode, cooperative_agents=[0, 2]) == 20 * 0.1
    assert aggregated_time(episode, cooperative_agents=[1]) == 30 * 0.1
    # All-agent aggregation is explicit: pass every known index.
    assert aggregated_time(episode, cooperative_agents=[0, 1, 2]) == 30 * 0.1


def test_duplicates_and_ordering_are_deterministic() -> None:
    """Duplicate indices and mapping order must not change the result."""
    episode = _make_episode()
    episode.cooperative_goal_steps = {0: 10, 1: 30, 2: 20}

    assert aggregated_time(episode, cooperative_agents=[2, 0, 2, 0]) == 20 * 0.1

    reordered = _make_episode()
    reordered.cooperative_goal_steps = {2: 20, 0: 10, 1: 30}
    assert aggregated_time(episode, cooperative_agents=[0, 1, 2]) == aggregated_time(
        reordered, cooperative_agents=[0, 1, 2]
    )


def test_empty_subset_is_unavailable() -> None:
    """An empty request aggregates over no agent and must return NaN."""
    episode = _make_episode()
    episode.cooperative_goal_steps = {0: 10}

    assert np.isnan(aggregated_time(episode, cooperative_agents=[]))


def test_missing_data_never_falls_back_to_single_agent() -> None:
    """Missing mappings or agents must be unavailable, never single-agent values."""
    episode = _make_episode()
    episode.reached_goal_step = 8

    # No mapping at all, even though the single-robot field is populated.
    assert np.isnan(aggregated_time(episode, cooperative_agents=[0]))

    episode.cooperative_goal_steps = {0: 10, 1: 30}
    # Requested agent absent from the mapping.
    assert np.isnan(aggregated_time(episode, cooperative_agents=[0, 7]))
    # Empty mapping.
    episode.cooperative_goal_steps = {}
    assert np.isnan(aggregated_time(episode, cooperative_agents=[0]))


def test_malformed_inputs_fail_closed() -> None:
    """Non-integral, boolean, or negative inputs must return NaN, not raise."""
    episode = _make_episode()
    episode.cooperative_goal_steps = {0: 10, 1: 30}

    assert np.isnan(aggregated_time(episode, cooperative_agents=[True]))
    assert np.isnan(aggregated_time(episode, cooperative_agents=[0.0]))

    malformed = _make_episode()
    malformed.cooperative_goal_steps = {0: -1, 1: 30}
    assert np.isnan(aggregated_time(malformed, cooperative_agents=[0, 1]))

    bool_step = _make_episode()
    bool_step.cooperative_goal_steps = {0: True, 1: 30}
    assert np.isnan(aggregated_time(bool_step, cooperative_agents=[0, 1]))


def test_invalid_timestep_and_agent_index_fail_closed() -> None:
    """Invalid time units and negative indices cannot produce a duration."""
    negative_dt = _make_episode(dt=-0.1)
    negative_dt.cooperative_goal_steps = {0: 10}
    assert np.isnan(aggregated_time(negative_dt, cooperative_agents=[0]))

    nonfinite_dt = _make_episode(dt=float("inf"))
    nonfinite_dt.cooperative_goal_steps = {0: 10}
    assert np.isnan(aggregated_time(nonfinite_dt, cooperative_agents=[0]))

    nonnumeric_dt = _make_episode()
    nonnumeric_dt.dt = "0.1"  # type: ignore[assignment]
    nonnumeric_dt.cooperative_goal_steps = {0: 10}
    assert np.isnan(aggregated_time(nonnumeric_dt, cooperative_agents=[0]))

    negative_index = _make_episode()
    negative_index.cooperative_goal_steps = {-1: 10}
    assert np.isnan(aggregated_time(negative_index, cooperative_agents=[-1]))
