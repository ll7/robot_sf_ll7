"""Validate that RobotEnv stores pickle recordings in the canonical artifact tree."""

import pickle
from pathlib import Path

import pytest
from loguru import logger

from robot_sf.common.artifact_paths import resolve_artifact_path
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.robot_env import VisualizableSimState
from robot_sf.nav.map_config import MapDefinition


@pytest.mark.slow
def test_recording():
    """Verify pickle recordings are written and contain expected state payloads.

    This protects the legacy recording contract relied upon by downstream tooling.
    """
    recordings_dir = resolve_artifact_path(Path("recordings"))
    recordings_dir.mkdir(parents=True, exist_ok=True)
    existing_files = set(recordings_dir.glob("*.pkl"))

    env = make_robot_env(recording_enabled=True, debug=False)
    env.reset()

    # Run the simulation for a few timesteps
    steps = 3
    for _ in range(steps):
        action = env.action_space.sample()  # replace with your action sampling logic
        env.step(action)

    # Save the recording
    env.reset()

    current_files = set(recordings_dir.glob("*.pkl"))
    new_files = current_files - existing_files
    assert new_files, "Recording file was not created"
    # Resolve deterministic ordering in case multiple artifacts exist
    recording_path = sorted(new_files)[-1]

    # Load the recording
    with recording_path.open("rb") as fh:
        recorded_states, map_def = pickle.load(fh)

    # Check that the recording has the correct length
    assert len(recorded_states) == steps

    # Check that the recorded states are instances of VisualizableSimState
    assert all(isinstance(state, VisualizableSimState) for state in recorded_states)

    # Check that the map definition is an instance of MapDefinition
    assert isinstance(map_def, MapDefinition)

    recording_path.unlink()


@pytest.mark.slow
def test_empty_save_first_reset_is_debug_not_warning():
    """First-reset empty save logs at debug; post-save empty save warns (#8422).

    Every ``reset()`` flushes the previous episode via ``save_recording()``, so the
    very first reset always finds an empty buffer. That expected case must not warn;
    only an empty save *after* a successful save is suspicious.
    """
    recordings_dir = resolve_artifact_path(Path("recordings"))
    recordings_dir.mkdir(parents=True, exist_ok=True)
    existing_files = set(recordings_dir.glob("*.pkl"))

    records: list = []
    handler_id = logger.add(lambda msg: records.append(msg.record), level="DEBUG")
    try:
        env = make_robot_env(recording_enabled=True, debug=False)
        env.reset()

        first_reset_skips = [r for r in records if "skipping save" in r["message"]]
        assert first_reset_skips, "expected an empty-save skip on first reset"
        assert all(r["level"].name != "WARNING" for r in first_reset_skips), (
            "first-reset empty save must not warn"
        )

        for _ in range(3):
            env.step(env.action_space.sample())
        env.reset()  # flushes the stepped episode; one successful save now
        assert env.unwrapped._legacy_recording_saves == 1

        records.clear()
        env.unwrapped.save_recording()
        post_save_skips = [r for r in records if "skipping save" in r["message"]]
        assert post_save_skips, "expected an empty-save skip after a save"
        assert any(r["level"].name == "WARNING" for r in post_save_skips), (
            "post-save empty save must warn"
        )
    finally:
        logger.remove(handler_id)
        for path in set(recordings_dir.glob("*.pkl")) - existing_files:
            path.unlink()


def test_pedestrian_empty_save_first_reset_is_debug(tmp_path, monkeypatch):
    """PedestrianEnv mirrors the first-reset debug / post-save warning split (#8422)."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    from robot_sf.gym_env.pedestrian_env import PedestrianEnv

    records: list = []
    handler_id = logger.add(lambda msg: records.append(msg.record), level="DEBUG")
    try:
        env = PedestrianEnv(robot_model=None, recording_enabled=True)
        try:
            env.reset()

            first_reset_skips = [r for r in records if "skipping save" in r["message"]]
            assert first_reset_skips, "expected an empty-save skip on first reset"
            assert all(r["level"].name != "WARNING" for r in first_reset_skips), (
                "first-reset empty save must not warn"
            )

            env.step(env.action_space.sample())
            env.reset()  # flushes the stepped episode; one successful save now
            assert env._legacy_recording_saves == 1

            records.clear()
            env.save_recording()
            post_save_skips = [r for r in records if "skipping save" in r["message"]]
            assert post_save_skips, "expected an empty-save skip after a save"
            assert any(r["level"].name == "WARNING" for r in post_save_skips), (
                "post-save empty save must warn"
            )
        finally:
            env.exit()
    finally:
        logger.remove(handler_id)


def test_single_agent_delegation_forwards_save_counter(tmp_path):
    """Delegation shim forwards/restores the save counter without AttributeError."""
    from robot_sf.gym_env.abstract_envs import SingleAgentEnv

    class _Stub:
        recorded_states: list = []
        map_def = None
        _legacy_recording_saves = 0

    stub = _Stub()
    SingleAgentEnv.save_recording(stub, str(tmp_path / "delegated.pkl"))
    assert stub._legacy_recording_saves == 0
    assert not (tmp_path / "delegated.pkl").exists()
