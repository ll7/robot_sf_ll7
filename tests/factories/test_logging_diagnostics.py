"""T014: Logging & diagnostics tests for environment factories.

Covers:
- INFO creation log emitted for robot, image, and pedestrian env factories.
- WARNING emitted on precedence override (reuse existing incompatibility scenario).
- Legacy mapping warnings already covered in T008 but we assert coexistence with creation log.
"""

from __future__ import annotations

from contextlib import contextmanager

from loguru import logger

from robot_sf.gym_env.environment_factory import (
    RecordingOptions,
    make_image_robot_env,
    make_pedestrian_env,
    make_robot_env,
)


@contextmanager
def capture_logs():
    """TODO docstring. Document this function."""
    messages: list[str] = []

    def _sink(msg):  # type: ignore[override]
        """TODO docstring. Document this function.

        Args:
            msg: TODO docstring.
        """
        messages.append(f"{msg.record['level'].name}:{msg.record['message']}")

    sink_id = logger.add(_sink)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def test_creation_logs_robot():
    """TODO docstring. Document this function."""
    with capture_logs() as logs:
        make_robot_env()
    assert any(entry.startswith("INFO:Creating robot env") for entry in logs)


def test_creation_logs_image():
    """TODO docstring. Document this function."""
    with capture_logs() as logs:
        make_image_robot_env()
    assert any(entry.startswith("INFO:Creating image robot env") for entry in logs)


def test_creation_logs_pedestrian_with_dummy_model():
    """TODO docstring. Document this function."""

    class DummyPolicy:  # minimal stub sufficient for constructor usage paths
        """TODO docstring. Document this class."""

        def predict(self, *_args, **_kwargs):  # pragma: no cover - simple stub
            """TODO docstring. Document this function.

            Args:
                _args: TODO docstring.
                _kwargs: TODO docstring.
            """
            return 0, {}

    with capture_logs() as logs:
        make_pedestrian_env(robot_model=DummyPolicy())
    assert any("Creating pedestrian env" in entry for entry in logs)
    assert any("robot_model=True" in entry for entry in logs)


def test_precedence_warning_and_creation_log(tmp_path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    rec = RecordingOptions(record=False)
    with capture_logs() as logs:
        make_robot_env(record_video=True, recording_options=rec, video_path=str(tmp_path / "v.mp4"))
    # Creation info + precedence warning
    assert any(entry.startswith("INFO:Creating robot env") for entry in logs)
    assert any("precedence" in entry for entry in logs if entry.startswith("WARNING:"))
