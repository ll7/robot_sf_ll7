"""Real development collections protect predictive actor joins before retraining."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.training.predictive_supervision import observation_episode_ids
from scripts.training import collect_predictive_hardcase_data as hardcase
from scripts.training import collect_predictive_planner_data as base


@pytest.fixture(params=[base, hardcase], ids=["base", "hardcase"])
def collector(request: pytest.FixtureRequest):
    """Exercise both collectors against the same simulator truth."""
    return request.param


@pytest.fixture
def collection_env():
    """Use the production structured sensor and a development-only environment."""
    config = RobotSimulationConfig()
    config.observation_mode = ObservationMode.SOCNAV_STRUCT
    config.use_planner = False
    config.sample_positions_globally = False
    env = make_robot_env(config=config, seed=1001, debug=False, recording_enabled=False)
    try:
        yield env
    finally:
        env.close()


def _collect_frame(collector, env, obs, episode_id):
    """Collect through the production observation/ID/frame seam; retain truth separately."""
    identities = observation_episode_ids(env, episode_id=episode_id)
    sensor = getattr(env.state.sensors, "wrapped_adapter", env.state.sensors)
    truth = {
        identity: np.asarray(env.simulator.ped_pos[index], dtype=np.float32).copy()
        for identity, index in zip(identities, sensor.current_source_indices, strict=True)
    }
    frame = collector._extract_frame(obs, max_agents=128, ped_ids=identities)
    np.testing.assert_allclose(
        frame.ped_positions_world,
        np.asarray(list(truth.values()), dtype=np.float32),
        atol=1e-5,
    )
    return frame, truth


@pytest.mark.parametrize("seed", [1001, 1002, 1003])
def test_dev_collection_targets_follow_simulator_actors(collector, collection_env, seed):
    """Check every supervised row in a small live rollout against simulator truth."""
    env = collection_env
    obs, _ = env.reset(seed=seed)
    frames, truths = [], []
    for _ in range(12):
        frame, truth = _collect_frame(collector, env, obs, f"dev-{seed}")
        frames.append(frame)
        truths.append(truth)
        obs, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        if terminated or truncated:
            break
    assert len(frames) > 4, "Development fixture must produce a supervised horizon"
    _, targets, masks, target_masks = collector._frames_to_samples(
        frames, max_agents=128, horizon_steps=4, ego_conditioning=False
    )
    assert np.count_nonzero(target_masks) > 0
    for t, frame in enumerate(frames[:-4]):
        np.testing.assert_array_equal(masks[t, : frame.ped_count], 1)
        # Independent geometric oracle: rotate simulator positions into the source ego frame.
        cos_h, sin_h = np.cos(frame.robot_heading), np.sin(frame.robot_heading)
        rotation = np.array([[cos_h, sin_h], [-sin_h, cos_h]])
        for row, identity in enumerate(frame.ped_ids):
            for k in range(4):
                future = truths[t + k + 1].get(identity)
                assert target_masks[t, row, k] == (future is not None)
                if future is not None:
                    expected = rotation @ (future - frame.robot_pos)
                    np.testing.assert_allclose(targets[t, row, k], expected, atol=1e-5)


def test_seeded_crossing_collection_keeps_actor_targets(collector, collection_env):
    """Replay #10262's crossing using real reset, sensor sorting, IDs, and collection."""
    env = collection_env
    env.reset(seed=1001)
    sensor = getattr(env.state.sensors, "wrapped_adapter", env.state.sensors)
    # Stage the original two-actor witness in a seeded simulator population. These
    # controlled positions are a regression fixture, not a navigation evaluation.
    positions = env.simulator.ped_pos
    assert len(positions) >= 2
    positions[:] = (20.0, 20.0)
    positions[:2] = [(4.0, 4.0), (4.8, 4.0)]
    source, _ = _collect_frame(collector, env, sensor.next_obs(), "crossing-dev-1001")
    positions[:2] = [(4.7, 4.0), (4.1, 4.0)]
    future, truth = _collect_frame(collector, env, sensor.next_obs(), "crossing-dev-1001")
    _, targets, _, target_masks = collector._frames_to_samples(
        [source, future], max_agents=128, horizon_steps=1, ego_conditioning=False
    )
    cos_h, sin_h = np.cos(source.robot_heading), np.sin(source.robot_heading)
    rotation = np.array([[cos_h, sin_h], [-sin_h, cos_h]])
    for slot in (0, 1):
        identity = f"crossing-dev-1001:simulator-slot-{slot}"
        row = source.ped_ids.index(identity)
        assert target_masks[0, row, 0] == 1
        np.testing.assert_allclose(
            targets[0, row, 0], rotation @ (truth[identity] - source.robot_pos), atol=1e-5
        )


def test_seeded_route_respawn_collection_masks_prior_actor(collector, collection_env):
    """Real route respawns must invalidate old identities in both target producers."""
    env = collection_env
    obs, _ = env.reset(seed=1001)
    source, _ = _collect_frame(collector, env, obs, "respawn-dev-1001")
    behavior = next(
        behavior
        for behavior in env.simulator.peds_behaviors
        if hasattr(behavior, "respawn_group_at_start") and behavior.groups.group_ids
    )
    group_id = next(iter(behavior.groups.group_ids))
    slots = {behavior.global_ped_offset + pid for pid in behavior.groups.groups[group_id]}
    behavior.respawn_group_at_start(group_id, guard_robot=False)
    sensor = getattr(env.state.sensors, "wrapped_adapter", env.state.sensors)
    future, _ = _collect_frame(collector, env, sensor.next_obs(), "respawn-dev-1001")
    _, targets, _, target_masks = collector._frames_to_samples(
        [source, future], max_agents=128, horizon_steps=1, ego_conditioning=False
    )
    for slot in slots:
        identity = f"respawn-dev-1001:simulator-slot-{slot}"
        row = source.ped_ids.index(identity)
        assert identity not in future.ped_ids
        assert target_masks[0, row, 0] == 0
        np.testing.assert_array_equal(targets[0, row, 0], [0, 0])
    assert np.count_nonzero(target_masks) > 0, "Unaffected actors must retain targets"
