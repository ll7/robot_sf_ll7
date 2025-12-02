"""T011: Tests for incompatible combination normalization.

Ensures that record_video=True with RecordingOptions.record=False emits warning and flips.
"""

from __future__ import annotations

from contextlib import contextmanager

from loguru import logger

from robot_sf.gym_env.environment_factory import RecordingOptions, make_robot_env


@contextmanager
def capture_warnings():
    """Capture warnings.

    Returns:
        Any: Auto-generated placeholder description.
    """
    messages: list[str] = []

    def _sink(msg):  # type: ignore[override]
        """Sink.

        Args:
            msg: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        if msg.record["level"].name in {"WARNING", "INFO"}:
            messages.append(msg.record["message"])

    sink_id = logger.add(_sink)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def test_boolean_and_options_conflict_flipped(tmp_path):
    """Test boolean and options conflict flipped.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    rec = RecordingOptions(record=False)
    with capture_warnings() as logs:
        env = make_robot_env(
            record_video=True,
            recording_options=rec,
            video_path=str(tmp_path / "vid.mp4"),
        )
    # Ensure SimulationView created (=> recording on)
    assert getattr(env, "sim_ui", None) is not None
    # Warning about precedence should appear
    assert any("precedence" in m for m in logs)
