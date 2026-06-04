"""Semantic-equivalence guards for simulator hot-path optimization work."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from robot_sf.benchmark.runner import (
    _simple_robot_policy,
    _simulate_episode_with_policy,
    run_batch,
    run_episode,
)

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"
DT = 0.1
HORIZON = 6
SEED = 321
FRAME_LIMIT = 5
SEMANTIC_ATOL = 1e-10
SCENARIO = {
    "id": "semantic-uni-low-open",
    "density": "low",
    "flow": "uni",
    "obstacle": "open",
    "groups": 0.0,
    "speed_var": "low",
    "goal_topology": "point",
    "robot_context": "embedded",
    "repeats": 1,
}
NON_FORCE_METRICS = (
    "success",
    "collisions",
    "near_misses",
    "min_distance",
    "mean_distance",
    "avg_speed",
    "path_efficiency",
)


def _direct_policy(
    robot_pos: np.ndarray,
    _robot_vel: np.ndarray,
    robot_goal: np.ndarray,
    _ped_positions: np.ndarray,
    _dt: float,
) -> np.ndarray:
    """Return the same deterministic simple-policy command used by the benchmark runner."""

    return _simple_robot_policy(robot_pos, robot_goal, speed=1.0)


def _capture_simulator_semantics(*, record_forces: bool) -> dict[str, np.ndarray]:
    """Run the bounded synthetic simulator path and capture trajectory invariants."""

    (
        robot_pos,
        robot_vel,
        _robot_acc,
        ped_pos,
        ped_forces,
        _obstacles,
        _goal,
        _reached_goal_step,
    ) = _simulate_episode_with_policy(
        dict(SCENARIO),
        SEED,
        _direct_policy,
        HORIZON,
        DT,
        robot_start=None,
        robot_goal=None,
        record_forces=record_forces,
    )
    ped_pos_arr = np.stack(ped_pos)
    return {
        "robot_pos": np.vstack(robot_pos),
        "robot_vel": np.vstack(robot_vel),
        "ped_pos": ped_pos_arr,
        "ped_vel": np.diff(ped_pos_arr, axis=0) / DT,
        "ped_forces": np.stack(ped_forces),
    }


def _episode_record(*, record_forces: bool) -> dict[str, Any]:
    """Run a bounded episode and return the semantic record payload."""

    return run_episode(
        dict(SCENARIO),
        seed=SEED,
        horizon=HORIZON,
        dt=DT,
        record_forces=record_forces,
    )


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime-only fields before checking deterministic record equality."""

    normalized = dict(record)
    normalized.pop("timestamps", None)
    normalized.pop("wall_time_sec", None)
    normalized.pop("timing", None)
    return normalized


def _read_records(path) -> list[dict[str, Any]]:
    """Load JSONL records written by the bounded fixture."""

    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_simulator_hot_path_frames_are_semantically_repeatable() -> None:
    """Repeated runs should preserve positions, velocities, forces, and summary outcome."""

    first = _capture_simulator_semantics(record_forces=True)
    second = _capture_simulator_semantics(record_forces=True)

    for key in ("robot_pos", "robot_vel", "ped_pos", "ped_vel", "ped_forces"):
        np.testing.assert_allclose(
            first[key][:FRAME_LIMIT],
            second[key][:FRAME_LIMIT],
            rtol=0.0,
            atol=SEMANTIC_ATOL,
            err_msg=f"{key} drifted across repeated deterministic simulator runs",
        )

    first_record = _episode_record(record_forces=True)
    second_record = _episode_record(record_forces=True)
    assert first_record["termination_reason"] == second_record["termination_reason"]
    assert first_record["status"] == second_record["status"]
    assert first_record["outcome"] == second_record["outcome"]
    assert _normalize_record(first_record) == _normalize_record(second_record)


def test_record_forces_flag_preserves_motion_but_changes_force_semantics() -> None:
    """Force recording should not change motion, but should change force sample semantics."""

    with_forces = _capture_simulator_semantics(record_forces=True)
    without_forces = _capture_simulator_semantics(record_forces=False)

    for key in ("robot_pos", "robot_vel", "ped_pos", "ped_vel"):
        np.testing.assert_allclose(
            with_forces[key][:FRAME_LIMIT],
            without_forces[key][:FRAME_LIMIT],
            rtol=0.0,
            atol=SEMANTIC_ATOL,
            err_msg=f"{key} changed when only record_forces was toggled",
        )

    assert with_forces["ped_forces"].shape == without_forces["ped_forces"].shape
    assert np.count_nonzero(np.linalg.norm(with_forces["ped_forces"], axis=2) > 0.0) > 0
    np.testing.assert_allclose(without_forces["ped_forces"], 0.0, rtol=0.0, atol=0.0)

    record_with_forces = _episode_record(record_forces=True)
    record_without_forces = _episode_record(record_forces=False)
    assert record_with_forces["termination_reason"] == record_without_forces["termination_reason"]
    assert record_with_forces["status"] == record_without_forces["status"]
    assert record_with_forces["outcome"] == record_without_forces["outcome"]

    for key in NON_FORCE_METRICS:
        assert record_with_forces["metrics"][key] == pytest.approx(
            record_without_forces["metrics"][key]
        )
    assert record_with_forces["metrics"]["force_sample_stats"]["status"] == "ok"
    assert record_without_forces["metrics"]["force_sample_stats"]["status"] == "all-zero"
    assert "force_quantiles" in record_with_forces["metrics"]
    assert "force_quantiles" not in record_without_forces["metrics"]


def test_semantic_fixture_jsonl_ordering_is_stable(tmp_path) -> None:
    """Repeated batch writes should keep JSONL ordering and semantic payloads stable."""

    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    scenario = dict(SCENARIO, repeats=2)
    for path in (first_path, second_path):
        run_batch(
            [scenario],
            out_path=path,
            schema_path=SCHEMA_PATH,
            base_seed=SEED,
            horizon=HORIZON,
            dt=DT,
            record_forces=True,
            append=False,
            workers=1,
            resume=False,
        )

    first_records = _read_records(first_path)
    second_records = _read_records(second_path)
    assert [record["seed"] for record in first_records] == [SEED, SEED + 1]
    assert [record["episode_id"] for record in first_records] == [
        record["episode_id"] for record in second_records
    ]
    assert [record["seed"] for record in first_records] == [
        record["seed"] for record in second_records
    ]
    for first, second in zip(first_records, second_records, strict=True):
        assert list(first.keys()) == list(second.keys())
        assert list(first["metrics"].keys()) == list(second["metrics"].keys())
        assert _normalize_record(first) == _normalize_record(second)
