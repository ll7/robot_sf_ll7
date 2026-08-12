"""Contract tests for the bounded BRNE map-runner policy adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.map_runner_policies import brne as brne_policy


class _FakeBRNEPlanner:
    """Small planner double that exercises the map-runner bridge without staged GPL code."""

    last_instance: _FakeBRNEPlanner | None = None

    def __init__(self, config: Any, *, seed: int | None = None) -> None:
        self.config = config
        self.seed = seed
        self.reset_calls: list[int | None] = []
        self.closed = False
        self.last_observation: dict[str, Any] | None = None
        _FakeBRNEPlanner.last_instance = self

    def get_metadata(self) -> dict[str, Any]:
        """Return native-looking planner metadata for the adapter contract."""
        return {
            "algorithm": "brne",
            "status": "ok",
            "runtime_status": "not_started",
            "effective_num_samples": None,
        }

    def reset(self, *, seed: int | None = None) -> None:
        """Record the seed-aware episode reset."""
        self.reset_calls.append(seed)

    def close(self) -> None:
        """Record adapter teardown."""
        self.closed = True

    def step(self, obs: dict[str, Any]) -> dict[str, float]:
        """Return a finite native unicycle action and retain converted observations."""
        self.last_observation = obs
        return {"v": 0.25, "omega": -0.1}


def _map_observation() -> dict[str, Any]:
    """Return the smallest flattened map observation accepted by the bridge."""
    return {
        "robot_position": [0.0, 0.0],
        "robot_velocity_xy": [0.5, 0.0],
        "robot_heading": [0.0],
        "robot_speed": [0.5],
        "robot_radius": [0.3],
        "goal_current": [5.0, 0.0],
        "pedestrians_positions": [],
        "pedestrians_velocities": [],
        "pedestrians_count": [0],
        "sim_timestep": [0.1],
    }


def test_brne_policy_builds_native_diagnostic_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder exposes explicit provenance and runs the converted observation path."""
    monkeypatch.setattr(brne_policy, "BRNEPlanner", _FakeBRNEPlanner)

    policy, metadata = brne_policy.build(
        "brne",
        {"stage_path": str(Path("third_party/external_repos/brne"))},
        robot_kinematics="unicycle",
        robot_command_mode="unicycle",
    )

    assert metadata["status"] == "ok"
    assert metadata["brne_diagnostic"]["status"] == "native_core_via_adapter"
    assert metadata["brne_diagnostic"]["fallback_policy"] == (
        "disabled; fallback/degraded rows are unavailable"
    )
    assert metadata["upstream_reference"]["commit"] == "633a5cd"

    action = policy(_map_observation())
    assert action == pytest.approx((0.25, -0.1))
    planner = _FakeBRNEPlanner.last_instance
    assert planner is not None
    assert planner.last_observation is not None
    assert planner.last_observation["robot"]["goal"] == [5.0, 0.0]
    assert callable(policy._planner_stats)
    assert policy._planner_stats()["planner_metadata"]["status"] == "ok"

    policy._planner_reset(seed=113)
    assert planner.reset_calls == [113]
    policy._planner_close()
    assert planner.closed is True


def test_brne_policy_rejects_fallback_and_paper_flags() -> None:
    """The diagnostic builder must not silently widen its evidence boundary."""
    with pytest.raises(ValueError, match="fallback_on_error: false"):
        brne_policy.build("brne", {"fallback_on_error": True})
    with pytest.raises(ValueError, match="include_in_paper: true"):
        brne_policy.build("brne", {"include_in_paper": True})


def test_brne_policy_fails_closed_when_stage_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing staged core is an unavailable diagnostic row, not a fallback."""

    class _MissingPlanner(_FakeBRNEPlanner):
        def get_metadata(self) -> dict[str, Any]:
            return {"algorithm": "brne", "status": "missing_dependency"}

    monkeypatch.setattr(brne_policy, "BRNEPlanner", _MissingPlanner)
    with pytest.raises(RuntimeError, match="staged core is unavailable"):
        brne_policy.build("brne", {})
    assert _MissingPlanner.last_instance is not None
    assert _MissingPlanner.last_instance.closed is True
