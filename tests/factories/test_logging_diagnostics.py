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
    """Collect level-prefixed loguru messages emitted inside the with-block."""
    messages: list[str] = []

    def _sink(msg):  # type: ignore[override]
        """Append the message as "<LEVEL>:<message>" for later assertions.

        Args:
            msg: Loguru message record passed to the sink.
        """
        messages.append(f"{msg.record['level'].name}:{msg.record['message']}")

    sink_id = logger.add(_sink)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def test_creation_logs_robot():
    """make_robot_env emits a DEBUG "Creating robot env" creation log."""
    with capture_logs() as logs:
        make_robot_env()
    assert any(entry.startswith("DEBUG:Creating robot env") for entry in logs)


def test_creation_logs_image():
    """make_image_robot_env emits an INFO "Creating image robot env" log."""
    with capture_logs() as logs:
        make_image_robot_env()
    assert any(entry.startswith("INFO:Creating image robot env") for entry in logs)


def test_creation_logs_pedestrian_with_dummy_model():
    """make_pedestrian_env logs creation and that a robot model was supplied."""

    class DummyPolicy:  # minimal stub sufficient for constructor usage paths
        """Minimal policy stub exposing the predict method the env expects."""

        def predict(self, *_args, **_kwargs):  # pragma: no cover - simple stub
            """Return a trivial (action, info) tuple, ignoring all inputs.

            Args:
                _args: Unused positional arguments from the policy interface.
                _kwargs: Unused keyword arguments from the policy interface.
            """
            return 0, {}

    with capture_logs() as logs:
        make_pedestrian_env(robot_model=DummyPolicy())
    assert any("Creating pedestrian env" in entry for entry in logs)
    assert any("robot_model=True" in entry for entry in logs)


def test_precedence_warning_and_creation_log(tmp_path):
    """Creation log and precedence warning both appear for conflicting options.

    Args:
        tmp_path: Temporary directory supplying the video output path.
    """
    rec = RecordingOptions(record=False)
    with capture_logs() as logs:
        make_robot_env(record_video=True, recording_options=rec, video_path=str(tmp_path / "v.mp4"))
    # Creation debug log + precedence warning
    assert any(entry.startswith("DEBUG:Creating robot env") for entry in logs)
    assert any("precedence" in entry for entry in logs if entry.startswith("WARNING:"))
