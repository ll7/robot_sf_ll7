"""Tests for the Robot SF public API facade (Issue #8245)."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

import robot_sf
from robot_sf.benchmark.types import EpisodeRecord, MetricsBundle, ScenarioSpec

if TYPE_CHECKING:
    from pathlib import Path


def test_import_weight_fresh_process():
    """Verify that 'import robot_sf' does not pull in pygame, torch, or stable_baselines3."""
    code = (
        "import sys, robot_sf; "
        "heavy = [m for m in ('pygame', 'torch', 'stable_baselines3') if m in sys.modules]; "
        "assert not heavy, f'Heavy modules imported: {heavy}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Import weight failure:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )


def test_public_api_exports_and_dir():
    """Verify __all__, __dir__, and lazy attribute resolution."""
    expected_exports = {
        "EpisodeRecord",
        "ManifestWriter",
        "PlannerProtocol",
        "RunRegistry",
        "RunTrackerConfig",
        "ScenarioSpec",
        "api",
        "generate_run_id",
        "load_scenario",
        "make_env",
        "run_episode",
        "telemetry",
    }
    assert set(robot_sf.__all__) == expected_exports

    dir_names = set(dir(robot_sf))
    for name in expected_exports:
        assert name in dir_names
        attr = getattr(robot_sf, name)
        assert attr is not None

    assert robot_sf.EpisodeRecord is EpisodeRecord
    assert robot_sf.ScenarioSpec is ScenarioSpec
    dummy_bundle = MetricsBundle(values={"test": 1.0})
    assert dummy_bundle.get("test") == 1.0

    with pytest.raises(AttributeError, match="has no attribute 'nonexistent_symbol'"):
        _ = robot_sf.nonexistent_symbol


def test_load_scenario_resolves_path_and_stem():
    """Verify scenario resolution by file path and by stem."""
    sc_from_path = robot_sf.load_scenario("configs/scenarios/single/quickstart_demo.yaml")
    assert isinstance(sc_from_path, dict)
    assert sc_from_path.get("name") == "quickstart_demo_crossing_basic"
    assert "__scenario_path__" in sc_from_path

    sc_from_stem = robot_sf.load_scenario("quickstart_demo")
    assert isinstance(sc_from_stem, dict)
    assert sc_from_stem.get("name") == "quickstart_demo_crossing_basic"

    with pytest.raises(FileNotFoundError):
        robot_sf.load_scenario("definitely_nonexistent_scenario_12345")


def test_make_env_and_run_episode_roundtrip(tmp_path: Path):
    """Verify make_env + run_episode + episode.save round-trip."""
    env = robot_sf.make_env(
        scenario="configs/scenarios/single/quickstart_demo.yaml",
        seed=111,
    )
    try:
        assert hasattr(env, "scenario_id")
        assert env.applied_seed == 111

        record = robot_sf.run_episode(env, max_steps=5)
        assert isinstance(record, EpisodeRecord)
        assert record.seed == 111
        assert 1 <= record.horizon <= 5
        assert "steps" in record.metrics.values
        assert record.metrics.values["steps"] == float(record.horizon)

        # Save and reload round-trip
        save_file = tmp_path / "saved_episode.json"
        saved_path = record.save(save_file)
        assert saved_path.is_file()

        loaded = EpisodeRecord.load(save_file)
        assert loaded.episode_id == record.episode_id
        assert loaded.scenario_id == record.scenario_id
        assert loaded.seed == record.seed
        assert loaded.horizon == record.horizon
        assert loaded.metrics.values == record.metrics.values
    finally:
        env.close()


def test_make_env_keyword_only_default():
    """Verify make_env works with defaults when scenario is omitted."""
    env = robot_sf.make_env(seed=42)
    try:
        assert env.applied_seed == 42
        record = robot_sf.run_episode(env, max_steps=2)
        assert isinstance(record, EpisodeRecord)
        assert record.seed == 42
        assert record.horizon == 2
    finally:
        env.close()


def test_make_env_with_mapping_and_planner():
    """Verify make_env supports scenario mappings and run_episode supports planners."""
    import numpy as np

    sc = robot_sf.load_scenario("quickstart_demo")
    env = robot_sf.make_env(scenario=sc, seed=123)
    try:

        class DummyPlanner:
            name = "dummy_planner"

            def step(self, obs):
                return np.zeros(2, dtype=np.float32)

            def reset(self, seed=None):
                pass

        rec = robot_sf.run_episode(env, planner=DummyPlanner(), max_steps=2)
        assert rec.algo == "dummy_planner"
        assert rec.seed == 123
    finally:
        env.close()


@pytest.mark.parametrize(
    "planner_action",
    [{"v": 0.0, "omega": 0.0}, {"vx": 0.0, "vy": 0.0}],
)
def test_run_episode_converts_protocol_action(planner_action):
    """Verify baseline protocol mappings are projected into the env action space."""
    env = robot_sf.make_env(seed=123)
    try:

        class DictPlanner:
            def step(self, obs):
                return planner_action

        record = robot_sf.run_episode(env, planner=DictPlanner(), max_steps=1)
        assert record.horizon == 1
    finally:
        env.close()


def test_load_scenario_resolves_entry_name():
    """Verify lookup by the scenario entry name, not only the manifest filename."""
    scenario = robot_sf.load_scenario("quickstart_demo_crossing_basic")
    assert scenario["name"] == "quickstart_demo_crossing_basic"


def test_make_env_invalid_scenario_type():
    """Verify make_env rejects unsupported scenario types."""
    with pytest.raises(TypeError, match="scenario must be a str, Path, or Mapping"):
        robot_sf.make_env(scenario=12345)
