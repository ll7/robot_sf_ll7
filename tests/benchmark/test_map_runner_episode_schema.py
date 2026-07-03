"""Write-time episode-row mechanism/exposure instrumentation (issue #4242 AC #2).

These tests pin the native emission of the ``failure_mechanism_taxonomy.v1`` and
``interaction_exposure.v1`` schema blocks by the map-runner episode writer:

- the failure-mechanism block is always fail-closed ``unknown`` at write time
  (a single episode is not a paired-trace mechanism analysis, and geometry or
  scenario names must never be substituted for a real label);
- the interaction-exposure block is computed from the episode's own trajectory,
  or emitted as an explicit ``not_derivable`` block when trace support is absent
  (never fabricated zeros);
- the blocks are actually attached to the record returned by ``run_map_episode``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robot_sf.benchmark.failure_mechanism_taxonomy import (
    MECHANISM_SCHEMA_VERSION,
    validate_failure_mechanism_record,
)
from robot_sf.benchmark.interaction_exposure import INTERACTION_EXPOSURE_SCHEMA_VERSION
from robot_sf.benchmark.map_runner_episode import (
    _INTERACTION_EXPOSURE_RADIUS_M,
    _episode_evidence_fields,
    _finite_pedestrian_frames,
)
from tests.benchmark.test_map_runner_utils import _minimal_map_def

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENARIO_PATH = _REPO_ROOT / "configs/scenarios/sanity_v1.yaml"

# --- unit tests on the write-time helper -----------------------------------


def test_mechanism_block_is_failclosed_unknown_and_valid() -> None:
    """The writer emits a schema-valid unknown mechanism record with a caveat."""
    fields = _episode_evidence_fields(
        robot_pos_arr=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        ped_pos_arr=np.array([[[0.5, 0.0]], [[0.5, 0.0]]], dtype=float),
        dt=0.1,
        success=False,
    )
    mechanism = fields["failure_mechanism"]
    assert mechanism["mechanism_schema_version"] == MECHANISM_SCHEMA_VERSION
    assert mechanism["mechanism_label"] == "unknown"
    assert mechanism["mechanism_confidence"] == "unknown"
    assert mechanism["mechanism_caveat"]  # non-empty caveat required for unknown
    # A geometry/scenario name is never used as a label: the block validates.
    assert validate_failure_mechanism_record(mechanism)["mechanism_label"] == "unknown"


def test_exposure_computed_when_pedestrian_within_radius() -> None:
    """Pedestrians inside the radius yield a computed exposure share and clearance."""
    # Robot moves along x; a pedestrian sits at the origin so step 0 is exposed
    # (distance 0.5 < 2.0 m) and step 2 clears (distance 2.0 not < radius bound).
    robot = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype=float)
    peds = np.array([[[0.5, 0.0]], [[0.5, 0.0]], [[0.5, 0.0]]], dtype=float)
    fields = _episode_evidence_fields(robot_pos_arr=robot, ped_pos_arr=peds, dt=0.1, success=True)
    exposure = fields["interaction_exposure"]
    assert exposure["interaction_exposure_schema_version"] == INTERACTION_EXPOSURE_SCHEMA_VERSION
    assert exposure["interaction_exposure_status"] == "computed"
    assert exposure["interaction_exposure_denominator_steps"] == 3
    assert exposure["interaction_exposure_radius_m"] == _INTERACTION_EXPOSURE_RADIUS_M
    assert 0.0 <= exposure["interaction_exposure_share"] <= 1.0
    assert exposure["first_clearance_reason"] == "clearance_observed"


def test_low_exposure_success_requires_success_flag() -> None:
    """low_exposure_success is only true for a successful, low-exposure episode."""
    # Pedestrian far away every step -> exposure share 0, well below threshold.
    robot = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    peds = np.array([[[50.0, 50.0]], [[50.0, 50.0]]], dtype=float)
    success = _episode_evidence_fields(robot_pos_arr=robot, ped_pos_arr=peds, dt=0.1, success=True)[
        "interaction_exposure"
    ]
    failure = _episode_evidence_fields(
        robot_pos_arr=robot, ped_pos_arr=peds, dt=0.1, success=False
    )["interaction_exposure"]
    assert success["low_exposure_success"] is True
    assert failure["low_exposure_success"] is False


def test_no_pedestrians_is_not_derivable_not_zero_imputed() -> None:
    """A trace with no pedestrians fails closed rather than emitting fabricated zeros."""
    fields = _episode_evidence_fields(
        robot_pos_arr=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        ped_pos_arr=np.zeros((2, 0, 2), dtype=float),
        dt=0.1,
        success=True,
    )
    exposure = fields["interaction_exposure"]
    assert exposure["interaction_exposure_status"] == "not_derivable_no_pedestrians"
    assert exposure["first_clearance_reason"] == "no_pedestrians"


def test_empty_trajectory_is_not_derivable_missing_trace() -> None:
    """An empty robot trajectory yields a missing-trace exposure block."""
    fields = _episode_evidence_fields(
        robot_pos_arr=np.zeros((0, 2), dtype=float),
        ped_pos_arr=np.zeros((0, 0, 2), dtype=float),
        dt=0.1,
        success=False,
    )
    assert fields["interaction_exposure"]["interaction_exposure_status"] == (
        "not_derivable_missing_trace"
    )
    # Mechanism stays fail-closed unknown even for a degenerate episode.
    assert fields["failure_mechanism"]["mechanism_label"] == "unknown"


def test_finite_pedestrian_frames_drops_nan_padding() -> None:
    """NaN-padded pedestrian slots are dropped and frames align to the robot length."""
    padded = np.array(
        [
            [[1.0, 2.0], [np.nan, np.nan]],
            [[3.0, 4.0], [5.0, 6.0]],
        ],
        dtype=float,
    )
    frames = _finite_pedestrian_frames(padded, step_count=2)
    assert frames == [[(1.0, 2.0)], [(3.0, 4.0), (5.0, 6.0)]]


def test_nan_padded_pedestrians_do_not_break_exposure() -> None:
    """Padded pedestrian tensors compute an exposure block instead of raising."""
    robot = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    peds = np.array(
        [
            [[0.2, 0.0], [np.nan, np.nan]],
            [[np.nan, np.nan], [np.nan, np.nan]],
        ],
        dtype=float,
    )
    exposure = _episode_evidence_fields(
        robot_pos_arr=robot, ped_pos_arr=peds, dt=0.1, success=False
    )["interaction_exposure"]
    assert exposure["interaction_exposure_status"] == "computed"


# --- integration: the writer attaches both blocks to the real record --------


class _PedSim:
    """Simulator stub exposing map/robot buffers the episode loop reads."""

    def __init__(self) -> None:
        self.robot_pos = [np.array([5.0, 5.0], dtype=float)]
        self.ped_pos = np.array([[5.5, 5.0], [9.0, 5.0]], dtype=float)
        self.goal_pos = [np.array([12.0, 5.0], dtype=float)]
        self.map_def = _minimal_map_def()
        self.last_ped_forces = np.zeros((2, 2), dtype=float)


class _PedEnv:
    """Environment stub emitting a pedestrian-carrying observation for one step."""

    def __init__(self) -> None:
        self.simulator = _PedSim()

    @staticmethod
    def _observation() -> dict[str, Any]:
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
                "positions": np.array([[5.5, 5.0], [9.0, 5.0]], dtype=np.float32),
                "velocities": np.zeros((2, 2), dtype=np.float32),
                "count": np.array([2.0], dtype=np.float32),
                "radius": np.array([0.3], dtype=np.float32),
            },
        }

    def reset(self, seed: int | None = None):
        del seed
        return self._observation(), {}

    def step(self, action):
        del action
        return self._observation(), 0.0, True, False, {"success": False}

    def close(self) -> None:
        return None


def test_run_map_episode_record_carries_native_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real map-runner episode record natively carries both schema blocks."""
    from robot_sf.benchmark.map_runner import _run_map_episode

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
        lambda config, seed, debug: _PedEnv(),
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

    record = _run_map_episode(
        {"name": "episode-schema-smoke", "simulation_config": {"max_episode_steps": 1}},
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

    assert record["failure_mechanism"]["mechanism_schema_version"] == MECHANISM_SCHEMA_VERSION
    assert record["failure_mechanism"]["mechanism_label"] == "unknown"
    exposure = record["interaction_exposure"]
    assert exposure["interaction_exposure_schema_version"] == INTERACTION_EXPOSURE_SCHEMA_VERSION
    # Two pedestrians are present, so exposure is derivable (computed), not blank.
    assert exposure["interaction_exposure_status"] == "computed"
