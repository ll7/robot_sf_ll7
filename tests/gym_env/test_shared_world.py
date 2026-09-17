"""Regression tests for the shared-world contract (issue #9344).

Speed note: runner tests build a real two-robot simulator on the default map
(one simulator, shared pedestrians). Each construction costs a few seconds;
step counts stay small and deterministic via fixed seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.gym_env.shared_world import (
    SharedWorldPacketError,
    SharedWorldRunner,
    admit_shared_world,
    assign_agent_ids,
    validate_action_packet,
)
from robot_sf.gym_env.unified_config import MultiRobotConfig
from robot_sf.sim.simulator import init_simulators, split_robot_counts


def test_split_robot_counts_cover_requested_robots() -> None:
    """Per-simulator counts sum to the request, including exact multiples."""
    assert split_robot_counts(1, 16) == [1]
    assert split_robot_counts(2, 2) == [2]
    assert split_robot_counts(5, 2) == [2, 2, 1]
    assert split_robot_counts(10, 5) == [5, 5]
    assert split_robot_counts(50, 10) == [10, 10, 10, 10, 10]
    assert split_robot_counts(4, 2) == [2, 2]
    assert split_robot_counts(17, 16) == [16, 1]
    for requested, capacity in ((1, 1), (2, 2), (5, 2), (10, 5), (50, 10), (32, 16)):
        assert sum(split_robot_counts(requested, capacity)) == requested


def test_split_robot_counts_reject_non_positive() -> None:
    """Zero or negative counts fail closed instead of yielding empty worlds."""
    with pytest.raises(ValueError, match="num_robots"):
        split_robot_counts(0, 2)
    with pytest.raises(ValueError, match="num_start_pos"):
        split_robot_counts(2, 0)


def test_admission_pairs_admit_requested_count() -> None:
    """(capacity, requested) pairs admit exactly the requested robots."""
    for capacity, requested in ((1, 1), (2, 2), (5, 2), (10, 5), (50, 10)):
        receipt = admit_shared_world(
            world_id="w", requested_robots=requested, spawn_capacity=capacity
        )
        assert receipt.admitted_robots == requested
        assert receipt.spawn_capacity == capacity


def test_admission_rejects_over_capacity_without_guessing() -> None:
    """Over-capacity requests are rejected, never silently split or overlapped."""
    with pytest.raises(ValueError, match="insufficient"):
        admit_shared_world(world_id="w", requested_robots=5, spawn_capacity=2)
    with pytest.raises(ValueError, match="positive integer"):
        admit_shared_world(world_id="w", requested_robots=0, spawn_capacity=2)
    with pytest.raises(ValueError, match="world_id"):
        admit_shared_world(world_id="", requested_robots=1, spawn_capacity=2)


def test_agent_ids_are_stable_and_unique() -> None:
    """Agent IDs are deterministic in spawn order."""
    assert assign_agent_ids("w", 3) == ("agent_000", "agent_001", "agent_002")


def test_packet_validation_remaps_stable_order() -> None:
    """Packet order is irrelevant; output follows expected spawn order."""
    ids = ("agent_000", "agent_001")
    forward = validate_action_packet([("agent_000", [1.0, 0.0]), ("agent_001", [0.0, 0.5])], ids)
    reversed_packet = validate_action_packet(
        [("agent_001", [0.0, 0.5]), ("agent_000", [1.0, 0.0])], ids
    )
    assert [list(map(float, v)) for v in forward] == [[1.0, 0.0], [0.0, 0.5]]
    assert [list(map(float, v)) for v in reversed_packet] == [[1.0, 0.0], [0.0, 0.5]]


def test_packet_validation_rejects_before_mutation() -> None:
    """Missing, duplicate, foreign, malformed, and non-finite packets fail."""
    ids = ("agent_000", "agent_001")
    with pytest.raises(SharedWorldPacketError, match="missing"):
        validate_action_packet([("agent_000", [0.0, 0.0])], ids)
    with pytest.raises(SharedWorldPacketError, match="duplicate"):
        validate_action_packet([("agent_000", [0.0, 0.0]), ("agent_000", [0.0, 0.0])], ids[:1])
    with pytest.raises(SharedWorldPacketError, match="foreign"):
        validate_action_packet([("agent_000", [0.0, 0.0]), ("agent_999", [0.0, 0.0])], ids)
    with pytest.raises(SharedWorldPacketError, match="malformed"):
        validate_action_packet([("agent_000", [0.0, 0.0]), ("agent_001",)], ids)
    with pytest.raises(SharedWorldPacketError, match="finite"):
        validate_action_packet([("agent_000", [0.0, 0.0]), ("agent_001", [float("nan"), 0.0])], ids)
    with pytest.raises(SharedWorldPacketError, match="2-element"):
        validate_action_packet([("agent_000", [0.0, 0.0, 0.0])], ids[:1])


def _make_runner(num_robots: int = 2, seed: int = 0, **config_kwargs) -> SharedWorldRunner:
    """Build a deterministic shared-world runner on the default map."""
    config = MultiRobotConfig(num_robots=num_robots, **config_kwargs)
    return SharedWorldRunner.create(
        world_id="test-world", env_config=config, num_robots=num_robots, seed=seed
    )


def _hold_packet(runner: SharedWorldRunner) -> list[tuple[str, np.ndarray]]:
    """Return an all-zero action packet for every agent."""
    return [(agent_id, np.zeros(2)) for agent_id in runner.agent_ids]


def test_runner_binds_exactly_one_simulator() -> None:
    """All robots share one simulator, one clock, and explicit membership."""
    runner = _make_runner(3)
    assert len(runner.simulator.robots) == 3
    assert runner.agent_ids == ("agent_000", "agent_001", "agent_002")
    assert runner.world_id == "test-world"
    assert runner.dt > 0


def test_one_step_advances_every_robot_once() -> None:
    """One packet advances every robot and exactly one logical timestep."""
    runner = _make_runner(2)
    before = runner.robot_positions()
    result = runner.step(_hold_packet(runner))
    assert result.step_index == 1
    assert runner.step_index == 1
    assert result.statuses[0].agent_id == "agent_000"
    assert result.episode_done is False
    assert runner.robot_positions().shape == before.shape


def test_reordered_packets_give_identical_worlds() -> None:
    """Stable-ID remapping makes packet order unobservable in the world."""
    first = _make_runner(2)
    ids = first.agent_ids
    first.step([(ids[0], np.array([2.0, 0.1])), (ids[1], np.array([0.0, -0.2]))])
    second = _make_runner(2)
    second.step([(ids[1], np.array([0.0, -0.2])), (ids[0], np.array([2.0, 0.1]))])
    np.testing.assert_array_equal(first.robot_positions(), second.robot_positions())
    np.testing.assert_array_equal(first.ped_positions(), second.ped_positions())


def test_two_robots_interact_through_shared_pedestrians() -> None:
    """Robot motion propagates to the shared pedestrian state.

    The repulsion range is widened purely as an observability knob: the
    default 2 m threshold never triggers on this map/seed (closest approach
    2.25 m), which would make the mechanism unobservable, not absent. The
    force callback reads live robot positions either way.
    """
    config = MultiRobotConfig(num_robots=2, peds_have_robot_repulsion=True)
    config.sim_config.prf_config.activation_threshold = 30.0

    calm = SharedWorldRunner.create(world_id="calm", env_config=config, num_robots=2, seed=0)
    for _ in range(15):
        calm.step(_hold_packet(calm))

    disturbed = SharedWorldRunner.create(
        world_id="disturbed", env_config=config, num_robots=2, seed=0
    )
    ids = disturbed.agent_ids
    for _ in range(15):
        disturbed.step([(ids[0], np.array([5.0, 0.0])), (ids[1], np.zeros(2))])

    assert float(np.abs(calm.ped_positions() - disturbed.ped_positions()).max()) > 0.0
    # Control isolation: agent_1 holds still in both runs, so it never moves.
    np.testing.assert_array_equal(calm.robot_positions()[1], disturbed.robot_positions()[1])


def test_invalid_packet_leaves_world_unchanged() -> None:
    """Rejected packets never partially mutate robot or pedestrian state."""
    runner = _make_runner(2)
    robots_before = runner.robot_positions()
    peds_before = runner.ped_positions()
    ids = runner.agent_ids
    bad_packets = [
        [(ids[0], np.zeros(2))],
        [(ids[0], np.zeros(2)), ("agent_999", np.zeros(2))],
        [(ids[0], np.zeros(2)), (ids[0], np.zeros(2))],
        [(ids[0], np.zeros(2)), (ids[1], np.array([np.inf, 0.0]))],
    ]
    for packet in bad_packets:
        with pytest.raises(SharedWorldPacketError):
            runner.step(packet)
    np.testing.assert_array_equal(runner.robot_positions(), robots_before)
    np.testing.assert_array_equal(runner.ped_positions(), peds_before)
    assert runner.step_index == 0


def test_no_implicit_reset_across_steps() -> None:
    """Stepping never resets the simulator; only explicit reset() does.

    The simulator bumps an internal episode index on every reset, so the
    index is an exact reset counter without any monkeypatching.
    """
    runner = _make_runner(2)
    sim = runner.simulator
    index_before = sim._oracle_episode_index
    for _ in range(4):
        runner.step(_hold_packet(runner))
    assert sim._oracle_episode_index == index_before
    assert runner.step_index == 4
    runner.reset(seed=0)
    assert sim._oracle_episode_index == index_before + 1
    assert runner.step_index == 0


def test_finished_agent_holds_while_other_advances() -> None:
    """A done agent holds under zero action; the other robot keeps stepping."""
    runner = _make_runner(2)
    runner._done_flags = lambda: [True, False]
    ids = runner.agent_ids
    result = runner.step([(ids[0], np.array([5.0, 0.0])), (ids[1], np.zeros(2))])
    assert result.statuses[0].holding is True
    assert result.statuses[1].holding is False
    # The supplied move command for the done agent was replaced by hold:
    # identical to a run where agent_0 genuinely holds still.
    reference = _make_runner(2)
    reference.step(_hold_packet(reference))
    np.testing.assert_allclose(runner.robot_positions(), reference.robot_positions())


def test_horizon_timeout_ends_episode_and_requires_reset() -> None:
    """Timeout is represented per agent; a done episode refuses more steps."""
    config = MultiRobotConfig(num_robots=2)
    config.sim_config.sim_time_in_secs = 0.3
    runner = SharedWorldRunner.create(world_id="timeout", env_config=config, num_robots=2, seed=0)
    last = None
    for _ in range(10):
        last = runner.step(_hold_packet(runner))
        if last.episode_done:
            break
    assert last is not None and last.episode_done is True
    assert all(status.timed_out for status in last.statuses)
    assert last.episode_truncated is True
    with pytest.raises(SharedWorldPacketError, match="reset"):
        runner.step(_hold_packet(runner))
    runner.reset(seed=1)
    resumed = runner.step(_hold_packet(runner))
    assert resumed.episode_done is False


def test_over_capacity_creation_is_rejected() -> None:
    """The runner refuses worlds the map cannot host instead of splitting."""
    config = MultiRobotConfig(num_robots=50)
    with pytest.raises(ValueError, match="insufficient"):
        SharedWorldRunner.create(world_id="too-big", env_config=config, num_robots=50, seed=0)


def test_init_simulators_preserves_exact_multiples() -> None:
    """End-to-end cardinality: exact-multiple requests keep every robot."""
    config = MultiRobotConfig(num_robots=2)
    pool = config.map_pool.map_defs
    map_def = (
        pool[getattr(config, "map_id", None)]
        if getattr(config, "map_id", None) in pool
        else next(iter(pool.values()))
    )
    sims = init_simulators(config, map_def, 16, random_start_pos=False)
    assert sum(len(sim.robots) for sim in sims) == 16
