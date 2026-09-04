"""Fast deterministic coverage for the legacy recording-save empty-buffer split (#8422).

Slow end-to-end recording behavior lives in ``tests/test_state_recording.py``;
this module pins the branch contract (debug before the first save, warning after
a save, counter increment) without constructing environments.
"""

from loguru import logger

from robot_sf.gym_env.abstract_envs import SingleAgentEnv
from robot_sf.gym_env.base_env import BaseEnv
from robot_sf.gym_env.pedestrian_env import PedestrianEnv


def _bare_env(cls: type, tmp_path) -> object:
    """Build an uninitialized env shell with only the recording fields set."""
    env = cls.__new__(cls)
    env.recorded_states = []
    env.map_def = None
    env._recording_dir = tmp_path
    env._legacy_recording_saves = 0
    return env


def _skip_messages(records: list) -> list:
    return [r for r in records if "skipping save" in r["message"]]


def test_base_empty_save_before_first_save_is_debug(tmp_path):
    """Empty buffer with zero prior saves logs at debug, writes nothing."""
    env = _bare_env(BaseEnv, tmp_path)
    records: list = []
    handler_id = logger.add(lambda msg: records.append(msg.record), level="DEBUG")
    try:
        BaseEnv.save_recording(env, str(tmp_path / "first.pkl"))
    finally:
        logger.remove(handler_id)
    skips = _skip_messages(records)
    assert skips, "expected an empty-save skip"
    assert all(r["level"].name != "WARNING" for r in skips)
    assert env._legacy_recording_saves == 0
    assert not (tmp_path / "first.pkl").exists()


def test_base_empty_save_after_save_warns(tmp_path):
    """Empty buffer after a successful save warns; success increments the counter."""
    env = _bare_env(BaseEnv, tmp_path)
    env.recorded_states = ["state"]
    BaseEnv.save_recording(env, str(tmp_path / "episode.pkl"))
    assert env._legacy_recording_saves == 1
    assert (tmp_path / "episode.pkl").exists()

    records: list = []
    handler_id = logger.add(lambda msg: records.append(msg.record), level="DEBUG")
    try:
        BaseEnv.save_recording(env, str(tmp_path / "again.pkl"))
    finally:
        logger.remove(handler_id)
    skips = _skip_messages(records)
    assert skips, "expected an empty-save skip after a save"
    assert any(r["level"].name == "WARNING" for r in skips)
    assert not (tmp_path / "again.pkl").exists()


def test_pedestrian_empty_save_split(tmp_path):
    """PedestrianEnv override mirrors the debug-before / warn-after split."""
    env = _bare_env(PedestrianEnv, tmp_path)
    records: list = []
    handler_id = logger.add(lambda msg: records.append(msg.record), level="DEBUG")
    try:
        PedestrianEnv.save_recording(env, str(tmp_path / "first.pkl"))
    finally:
        logger.remove(handler_id)
    skips = _skip_messages(records)
    assert skips
    assert all(r["level"].name != "WARNING" for r in skips)

    env.recorded_states = ["state"]
    PedestrianEnv.save_recording(env, str(tmp_path / "episode.pkl"))
    assert env._legacy_recording_saves == 1

    records.clear()
    handler_id = logger.add(lambda msg: records.append(msg.record), level="DEBUG")
    try:
        PedestrianEnv.save_recording(env, str(tmp_path / "again.pkl"))
    finally:
        logger.remove(handler_id)
    skips = _skip_messages(records)
    assert skips
    assert any(r["level"].name == "WARNING" for r in skips)


def test_single_agent_delegation_forwards_save_counter(tmp_path):
    """Delegation shim forwards/restores the save counter without AttributeError."""

    class _Stub:
        recorded_states: list = []
        map_def = None
        _legacy_recording_saves = 0

    stub = _Stub()
    SingleAgentEnv.save_recording(stub, str(tmp_path / "delegated.pkl"))
    assert stub._legacy_recording_saves == 0
    assert not (tmp_path / "delegated.pkl").exists()
