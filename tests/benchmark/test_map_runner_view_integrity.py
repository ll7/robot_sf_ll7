"""Runtime fail-closed guard for degenerate planner observation views (#3634).

These tests pin the comprehensive runtime version of the #3568 static observation-format contract:
``map_runner`` must FAIL CLOSED when a planner's effective view is degenerate (the observation
carries pedestrians but the planner's own extractor sees none), unless the planner is declared
pedestrian-blind by design. They cover the three planner cases named in the issue Definition of
Done — ``stream_gap`` (post-#3567 fix), a shared-SOCNAV classical planner, and the trivial
reference planner — plus the silent-blind failure class and the conservative no-false-trip cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robot_sf.benchmark.algorithm_metadata import observation_spec_for_algorithm
from robot_sf.benchmark.map_runner import _build_policy, _run_map_episode
from robot_sf.benchmark.map_runner_view_integrity import (
    DEGENERATE_PLANNER_VIEW_REASON,
    DegeneratePlannerViewError,
    evaluate_effective_view_integrity,
    is_pedestrian_blind_by_design,
    observation_ped_count,
    probe_extracted_ped_count,
)
from robot_sf.planner.socnav import SocialForcePlannerAdapter, SocNavPlannerConfig
from robot_sf.planner.stream_gap import StreamGapPlannerAdapter, build_stream_gap_config
from tests.benchmark.test_map_runner_utils import _minimal_map_def


def _nested_observation(n_peds: int) -> dict[str, Any]:
    """Build the nested SOCNAV observation map_runner feeds, with ``n_peds`` pedestrians."""
    positions = [[8.0 + i, 5.0] for i in range(n_peds)]
    return {
        "robot": {"position": [5.0, 5.0], "heading": [0.0], "speed": [0.0], "radius": [0.4]},
        "goal": {"current": [12.0, 5.0], "next": [12.0, 5.0]},
        "pedestrians": {
            "positions": positions,
            "velocities": [[0.0, 0.0] for _ in range(n_peds)],
            "count": [float(n_peds)],
            "radius": [0.3],
        },
    }


def _flat_observation(n_peds: int) -> dict[str, Any]:
    """Build the flat map_runner-style observation with ``n_peds`` pedestrians."""
    positions = [[8.0 + i, 5.0] for i in range(n_peds)]
    return {
        "robot_position": [5.0, 5.0],
        "robot_heading": [0.0],
        "goal_current": [12.0, 5.0],
        "goal_next": [12.0, 5.0],
        "pedestrians_positions": positions,
        "pedestrians_velocities": [[0.0, 0.0] for _ in range(n_peds)],
        "pedestrians_count": [float(n_peds)],
    }


class _PolicyStub:
    """Minimal stand-in for a built policy callable carrying a planner adapter."""

    def __init__(self, adapter: Any) -> None:
        """Attach the adapter the way ``build_adapter_policy`` does."""
        self._planner_adapter = adapter

    def __call__(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Defer to the adapter's planner (unused by the guard, present for realism)."""
        return self._planner_adapter.plan(observation)


class _BlindSocnavAdapter:
    """Adapter whose extractor reads the wrong keys and returns zero peds (pre-#3567 bug class)."""

    def _socnav_fields(self, observation: dict[str, Any]) -> tuple[dict, dict, dict]:
        """Return an origin/empty view regardless of the real observation."""
        del observation
        return {"position": [0.0, 0.0]}, {"current": [0.0, 0.0]}, {"positions": None, "count": [0]}


class _OpaqueAdapter:
    """Adapter that exposes no recognised extractor, so its view is not probeable."""


def _goal_meta() -> dict[str, Any]:
    """Return algo_meta for the pedestrian-blind ``goal`` reference planner."""
    return {"observation_spec": observation_spec_for_algorithm("goal")}


def _socnav_meta(algo: str = "social_force") -> dict[str, Any]:
    """Return algo_meta for a pedestrian-consuming planner."""
    return {"observation_spec": observation_spec_for_algorithm(algo)}


# --- observation_ped_count: ground truth presented to the planner -------------------------------


def test_observation_ped_count_reads_nested_count() -> None:
    """The nested SOCNAV observation's declared count is the ground truth."""
    assert observation_ped_count(_nested_observation(3)) == 3


def test_observation_ped_count_reads_flat_count() -> None:
    """The flat map_runner observation's pedestrian count is read too."""
    assert observation_ped_count(_flat_observation(2)) == 2


def test_observation_ped_count_zero_when_absent() -> None:
    """An observation with no pedestrian payload reports zero (no phantom agents)."""
    assert observation_ped_count({"robot": {"position": [0.0, 0.0]}}) == 0


# --- pedestrian-blind-by-design registry --------------------------------------------------------


def test_goal_reference_declares_itself_pedestrian_blind() -> None:
    """The ``goal`` reference omits pedestrians from its inputs and is exempt by declaration."""
    assert is_pedestrian_blind_by_design(_goal_meta()) is True


def test_pedestrian_consuming_planner_is_not_blind_by_design() -> None:
    """A planner whose inputs include pedestrians is not exempt."""
    assert is_pedestrian_blind_by_design(_socnav_meta("orca")) is False


def test_missing_observation_spec_is_not_treated_as_blind() -> None:
    """Absent contract metadata must not silently exempt a planner (fail-closed default)."""
    assert is_pedestrian_blind_by_design({}) is False


# --- effective-view probe against real adapters -------------------------------------------------


def test_shared_socnav_planner_extracts_pedestrians_no_false_trip() -> None:
    """A shared-SOCNAV classical planner sees the pedestrians; the guard does not trip."""
    policy = _PolicyStub(SocialForcePlannerAdapter(config=SocNavPlannerConfig()))
    obs = _nested_observation(2)
    assert probe_extracted_ped_count(policy, obs) == 2
    result = evaluate_effective_view_integrity(
        policy_fn=policy, observation=obs, algo_meta=_socnav_meta("social_force")
    )
    assert result.degraded is False
    assert result.probed is True
    assert result.extracted_ped_count == 2


def test_stream_gap_post_fix_extracts_pedestrians_no_false_trip() -> None:
    """``stream_gap`` (post-#3567) extracts the pedestrians it was shown; the guard does not trip."""
    policy = _PolicyStub(StreamGapPlannerAdapter(config=build_stream_gap_config({})))
    obs = _nested_observation(2)
    assert probe_extracted_ped_count(policy, obs) == 2
    result = evaluate_effective_view_integrity(
        policy_fn=policy, observation=obs, algo_meta=_socnav_meta("stream_gap")
    )
    assert result.degraded is False


def test_stream_gap_reads_flat_observation_no_false_trip() -> None:
    """``stream_gap``'s flat-observation fallback (the #3567 fix) is not flagged as blind."""
    policy = _PolicyStub(StreamGapPlannerAdapter(config=build_stream_gap_config({})))
    assert probe_extracted_ped_count(policy, _flat_observation(2)) == 2


# --- the silent-blind failure class: fail closed ------------------------------------------------


def test_degenerate_view_trips_with_canonical_reason() -> None:
    """A planner that extracts zero peds from a peds-carrying observation fails closed."""
    policy = _PolicyStub(_BlindSocnavAdapter())
    obs = _nested_observation(2)
    assert probe_extracted_ped_count(policy, obs) == 0
    result = evaluate_effective_view_integrity(
        policy_fn=policy, observation=obs, algo_meta=_socnav_meta("stream_gap")
    )
    assert result.degraded is True
    assert result.degraded_reason == DEGENERATE_PLANNER_VIEW_REASON
    assert result.observation_ped_count == 2
    assert result.extracted_ped_count == 0


def test_degenerate_view_error_carries_diagnostic() -> None:
    """The fail-closed error carries the structured diagnostic and the canonical reason."""
    policy = _PolicyStub(_BlindSocnavAdapter())
    result = evaluate_effective_view_integrity(
        policy_fn=policy, observation=_nested_observation(2), algo_meta=_socnav_meta("stream_gap")
    )
    error = DegeneratePlannerViewError(result)
    assert error.diagnostic.degraded_reason == DEGENERATE_PLANNER_VIEW_REASON
    assert DEGENERATE_PLANNER_VIEW_REASON in str(error)


# --- conservative no-false-trip guards ----------------------------------------------------------


def test_blind_by_design_planner_is_exempt_even_with_peds_present() -> None:
    """The ``goal`` reference is not flagged even though it ignores the pedestrians it was shown."""
    policy = _PolicyStub(_BlindSocnavAdapter())
    result = evaluate_effective_view_integrity(
        policy_fn=policy, observation=_nested_observation(2), algo_meta=_goal_meta()
    )
    assert result.degraded is False
    assert result.pedestrian_blind_by_design is True
    assert result.probed is False  # exempt planners are not probed


def test_genuinely_zero_pedestrian_scenario_is_not_degenerate() -> None:
    """A scenario that legitimately presents zero pedestrians never trips the guard."""
    policy = _PolicyStub(_BlindSocnavAdapter())
    result = evaluate_effective_view_integrity(
        policy_fn=policy, observation=_nested_observation(0), algo_meta=_socnav_meta("stream_gap")
    )
    assert result.degraded is False
    assert result.observation_ped_count == 0


def test_unprobeable_planner_does_not_trip() -> None:
    """When the planner exposes no extractor the view is not probeable and the guard stays silent."""
    policy = _PolicyStub(_OpaqueAdapter())
    result = evaluate_effective_view_integrity(
        policy_fn=policy, observation=_nested_observation(2), algo_meta=_socnav_meta("orca")
    )
    assert result.degraded is False
    assert result.probed is False
    assert result.extracted_ped_count is None


def test_policy_without_adapter_is_not_probeable() -> None:
    """A bare policy callable with no attached adapter is treated as not probeable."""

    def _bare_policy(_obs: dict[str, Any]) -> tuple[float, float]:
        return 0.0, 0.0

    assert probe_extracted_ped_count(_bare_policy, _nested_observation(2)) is None


@pytest.mark.parametrize("count", [1, 4, 7])
def test_no_false_trip_scales_with_pedestrian_count(count: int) -> None:
    """A healthy shared-SOCNAV planner never trips regardless of how many peds are present."""
    policy = _PolicyStub(SocialForcePlannerAdapter(config=SocNavPlannerConfig()))
    result = evaluate_effective_view_integrity(
        policy_fn=policy,
        observation=_nested_observation(count),
        algo_meta=_socnav_meta("social_force"),
    )
    assert result.degraded is False
    assert result.extracted_ped_count == count


# --- end-to-end: the guard fires inside the real map_runner episode loop -------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENARIO_PATH = _REPO_ROOT / "configs/scenarios/sanity_v1.yaml"


class _PedPresentSim:
    """Simulator stub that always reports two pedestrians present in the world."""

    def __init__(self) -> None:
        """Seed the deterministic robot/goal/pedestrian buffers."""
        self.robot_pos = [np.array([5.0, 5.0], dtype=float)]
        self.ped_pos = np.array([[8.0, 5.0], [9.0, 5.0]], dtype=float)
        self.goal_pos = [np.array([12.0, 5.0], dtype=float)]
        self.map_def = _minimal_map_def()
        self.last_ped_forces = np.zeros((2, 2), dtype=float)


class _PedPresentEnv:
    """Environment stub emitting a nested SOCNAV observation that carries two pedestrians."""

    def __init__(self) -> None:
        """Attach the pedestrian-present simulator stub."""
        self.simulator = _PedPresentSim()

    @staticmethod
    def _observation() -> dict[str, Any]:
        """Return a nested SOCNAV observation with two visible pedestrians."""
        return {
            "robot": {
                "position": np.array([5.0, 5.0], dtype=np.float32),
                "heading": np.array([0.0], dtype=np.float32),
                "speed": np.array([0.0], dtype=np.float32),
                "radius": np.array([0.4], dtype=np.float32),
            },
            "goal": {
                "current": np.array([12.0, 5.0], dtype=np.float32),
                "next": np.array([12.0, 5.0], dtype=np.float32),
            },
            "pedestrians": {
                "positions": np.array([[8.0, 5.0], [9.0, 5.0]], dtype=np.float32),
                "velocities": np.zeros((2, 2), dtype=np.float32),
                "count": np.array([2.0], dtype=np.float32),
                "radius": np.array([0.3], dtype=np.float32),
            },
        }

    def reset(self, seed: int | None = None):
        """Return the pedestrian-carrying observation."""
        del seed
        return self._observation(), {}

    def step(self, action):
        """Advance one step and terminate (the guard fires on the first step anyway)."""
        del action
        return self._observation(), 0.0, True, False, {"success": False}

    def close(self) -> None:
        """Accept map-runner cleanup."""
        return None


def _patch_episode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch map_runner env construction and metrics so the episode loop runs in isolation."""
    dummy_config = type(
        "Cfg",
        (),
        {"sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})()},
    )()
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.make_robot_env",
        lambda config, seed, debug: _PedPresentEnv(),
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_shortest_path_length", lambda *args: 1.0
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_all_metrics",
        lambda *args, **kwargs: {"success": 0.0, "collisions": 0.0},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.post_process_metrics", lambda metrics, **kwargs: metrics
    )


def _run_stream_gap_episode() -> dict[str, Any]:
    """Run a single ``stream_gap`` episode through the real map_runner loop."""
    return _run_map_episode(
        {"name": "view-integrity-smoke", "simulation_config": {"max_episode_steps": 1}},
        seed=1,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="stream_gap",
        algo_config={},
        scenario_path=_SCENARIO_PATH,
    )


def test_map_episode_fails_closed_on_degenerate_planner_view(monkeypatch) -> None:
    """A blind planner with pedestrians present must fail the episode closed, not emit a result."""
    _patch_episode_env(monkeypatch)

    real_build_policy = _build_policy

    def _blind_build_policy(algo, algo_config, **kwargs):
        """Build the real policy/metadata, then swap in a blind extractor adapter."""
        policy_fn, algo_meta = real_build_policy(algo, algo_config, **kwargs)
        policy_fn._planner_adapter = _BlindSocnavAdapter()
        return policy_fn, algo_meta

    monkeypatch.setattr("robot_sf.benchmark.map_runner._build_policy", _blind_build_policy)

    with pytest.raises(DegeneratePlannerViewError, match=DEGENERATE_PLANNER_VIEW_REASON):
        _run_stream_gap_episode()


def test_map_episode_records_clean_effective_view_for_healthy_planner(monkeypatch) -> None:
    """A healthy ``stream_gap`` planner produces a record with a clean (non-degraded) view."""
    _patch_episode_env(monkeypatch)

    record = _run_stream_gap_episode()

    effective_view = record["integrity"]["effective_view"]
    assert effective_view is not None
    assert effective_view["degraded"] is False
    assert effective_view["degraded_reason"] is None
    assert effective_view["observation_ped_count"] == 2
    assert effective_view["extracted_ped_count"] == 2
