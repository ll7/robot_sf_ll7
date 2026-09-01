"""Actor/oracle channel separation tests for opt-in force traces (#8065)."""

from __future__ import annotations

import numpy as np

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.sim.sim_config import SimulationSettings


def _contains_key(value: object, key: str) -> bool:
    """Return whether a nested observation payload contains ``key``."""
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, key) for item in value)
    return False


def test_oracle_force_trace_is_info_only_and_absent_from_actor_observation() -> None:
    """Opt-in privileged truth is emitted after observation construction only."""
    config = RobotSimulationConfig(
        sim_config=SimulationSettings(oracle_force_trace_enabled=True),
    )
    env = make_robot_env(config=config)
    try:
        _obs, _reset_info = env.reset(seed=7)
        obs, _reward, _terminated, _truncated, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32),
        )
    finally:
        env.close()

    assert "oracle_transition_trace" in info
    assert info["oracle_transition_trace"]["schema_version"] == "oracle_transition_trace.v1"
    assert not _contains_key(obs, "oracle_transition_trace")
