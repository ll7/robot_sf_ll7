"""Regression tests for issue #4954.

``run_map_episode`` finalizes episode metadata by writing ~10 keys into the
``algo_meta`` dict produced by the policy builder (``adapter_impact`` status,
``tracking_precision``, ``safety_wrapper``, ``planner_runtime``, ...).

``enrich_algorithm_metadata`` only *shallow*-copies that dict, so nested mutable
structures -- notably the ``adapter_impact`` counters -- stayed shared with the
builder's original object. A builder that reuses/caches the same ``algo_meta``
across episodes (an extension point, since ``policy_builder`` is caller-provided)
would therefore see finalization writes leak back across episodes.

These tests pin the fix: the finalization phase must operate on an isolated copy
so a builder-provided (and builder-retained) ``algo_meta`` is never mutated in
place, while the returned record still carries the correctly finalized values.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from robot_sf.benchmark import map_runner_episode


class _EpisodeSim:
    def __init__(self) -> None:
        self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
        self.ped_pos = np.array([[2.0, 0.0]], dtype=float)
        self.goal_pos = [np.array([1.0, 0.0], dtype=float)]
        self.map_def = SimpleNamespace(obstacles=[], bounds=(0.0, 0.0, 1.0, 1.0))
        self.robot_vel = [np.array([0.0, 0.0], dtype=float)]

    def get_pedestrian_forces(self) -> np.ndarray:
        return np.zeros((1, 2), dtype=float)


class _EpisodeEnv:
    def __init__(self) -> None:
        self.simulator = _EpisodeSim()
        self.action_space = None

    def reset(self, seed=None):
        return (
            {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 0.0]},
                "pedestrians": {"positions": self.simulator.ped_pos},
            },
            {},
        )

    def step(self, _action):
        return self.reset()[0], 0.0, True, False, {"meta": {"is_route_complete": True}}

    def close(self) -> None:
        return None


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        sim_config=SimpleNamespace(
            time_per_step_in_secs=0.1,
            robot_radius=0.1,
            ped_radius=0.1,
        )
    )


def _patch_episode_runtime(monkeypatch) -> None:
    """Replace the heavy simulator/metrics calls with cheap deterministic stubs."""
    monkeypatch.setattr(
        map_runner_episode,
        "_build_env_config",
        lambda _scenario, scenario_path: _config(),
    )
    monkeypatch.setattr(
        map_runner_episode,
        "make_robot_env",
        lambda config, seed, debug: _EpisodeEnv(),
    )
    monkeypatch.setattr(map_runner_episode, "sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(map_runner_episode, "compute_shortest_path_length", lambda *args: 1.0)
    monkeypatch.setattr(
        map_runner_episode,
        "compute_all_metrics",
        lambda *args, **kwargs: {"success": 1.0, "collisions": 0.0},
    )
    monkeypatch.setattr(
        map_runner_episode,
        "post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )


def _run_episode(monkeypatch, builder_meta: dict[str, Any]) -> dict[str, Any]:
    """Run a minimal episode whose builder returns (and retains) ``builder_meta``."""
    _patch_episode_runtime(monkeypatch)

    def policy_builder(*_args, **_kwargs):
        def policy(_obs):
            return (1.0, 0.0)

        # Return the SAME object each call so a retained reference simulates a
        # builder that caches/reuses algo_meta across episodes.
        return policy, builder_meta

    return map_runner_episode.run_map_episode(
        {"name": "algo-meta-isolation-4954", "simulation_config": {"max_episode_steps": 1}},
        seed=1,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=Path(__file__),
        adapter_impact_eval=True,
        policy_builder=policy_builder,
    )


def test_builder_provided_algo_meta_not_mutated_by_finalization(monkeypatch) -> None:
    """A builder-retained algo_meta must not be mutated in place (#4954)."""
    builder_meta = {
        "algorithm": "goal",
        "adapter_impact": {
            "requested": True,
            "native_steps": 0,
            "adapted_steps": 0,
            "status": "pending",
        },
    }
    snapshot_before = copy.deepcopy(builder_meta)

    record = _run_episode(monkeypatch, builder_meta)

    # The builder's retained dict must be byte-for-byte unchanged.
    assert builder_meta == snapshot_before
    # Specifically the nested adapter_impact finalization writes must not leak:
    # status must stay "pending" and adapter_fraction must not be injected.
    assert builder_meta["adapter_impact"]["status"] == "pending"
    assert "adapter_fraction" not in builder_meta["adapter_impact"]
    assert "execution_mode" not in builder_meta["adapter_impact"]
    # The record, by contrast, must carry the finalized adapter_impact values.
    finalized = record["algorithm_metadata"]["adapter_impact"]
    assert "adapter_fraction" in finalized
    assert finalized["status"] in {"complete", "not_applicable"}


def test_finalized_algo_meta_is_independent_object(monkeypatch) -> None:
    """The record's algo_meta and nested adapter_impact must be fresh objects (#4954)."""
    builder_meta: dict[str, Any] = {
        "algorithm": "goal",
        "adapter_impact": {
            "requested": True,
            "native_steps": 0,
            "adapted_steps": 0,
            "status": "pending",
        },
    }

    record = _run_episode(monkeypatch, builder_meta)

    record_meta = record["algorithm_metadata"]
    # Top-level dict must be a distinct object.
    assert record_meta is not builder_meta
    # Nested adapter_impact must also be a distinct object (deep isolation).
    assert record_meta["adapter_impact"] is not builder_meta["adapter_impact"]
    # Mutating the record's nested dict after the fact must not touch the builder.
    record_meta["adapter_impact"]["status"] = "tampered"
    assert builder_meta["adapter_impact"]["status"] == "pending"


def test_finalization_keys_present_in_record(monkeypatch) -> None:
    """Regression guard: isolation must not drop the finalized metadata keys."""
    builder_meta = {"algorithm": "goal"}

    record = _run_episode(monkeypatch, builder_meta)

    meta = record["algorithm_metadata"]
    # A representative subset of the ~10 finalization-phase writes.
    for key in (
        "tracking_precision",
        "ammv_feasibility",
        "algorithm",
        "observation_level",
    ):
        assert key in meta, f"finalization key missing from record: {key}"


def test_builder_provided_algo_meta_top_level_not_mutated(monkeypatch) -> None:
    """Even a bare builder meta (no nested structures) must not gain finalization keys."""
    builder_meta = {"algorithm": "goal"}
    snapshot_before = copy.deepcopy(builder_meta)

    _run_episode(monkeypatch, builder_meta)

    # No finalization-phase top-level keys should have leaked back.
    assert builder_meta == snapshot_before
