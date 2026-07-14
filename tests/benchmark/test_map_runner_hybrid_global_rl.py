"""Tests for map-runner hybrid global/RL-local policy registration."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.map_runner import _build_policy
from robot_sf.benchmark.map_runner_policies.hybrid_global_rl import HYBRID_GLOBAL_RL_KEYS


def test_hybrid_global_rl_builder_registers_aliases(monkeypatch) -> None:
    class _DummyAdapter:
        def __init__(self, *, config):
            self.config = config

        def plan(self, observation):
            return 0.0, 0.0

        def diagnostics(self):
            return {"status": "ok", "waypoint_status": "ok"}

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner_policies.hybrid_global_rl.HybridGlobalRLLocalAdapter",
        _DummyAdapter,
    )

    for algo in HYBRID_GLOBAL_RL_KEYS:
        policy, meta = _build_policy(
            algo,
            {
                "local_policy_config": {
                    "fallback_to_goal": False,
                    "obs_mode": "dict",
                    "action_space": "unicycle",
                }
            },
        )

        assert callable(policy)
        assert meta["algorithm"] == algo
        assert meta["hybrid_global_rl"]["status"] == "enabled"
        assert meta["planner_kinematics"]["limitations"] == "diagnostic_only_not_benchmark_evidence"


def test_hybrid_global_rl_builder_missing_sac_checkpoint_fails_closed() -> None:
    with pytest.raises(ValueError, match="local-only model artifact|model_id"):
        _build_policy(
            "hybrid_global_rl",
            {
                "allow_goal_fallback": False,
                "local_policy_config": {
                    "fallback_to_goal": False,
                    "obs_mode": "dict",
                    "action_space": "unicycle",
                },
            },
        )
