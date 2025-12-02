"""T012: Pedestrian factory normalization tests."""

from __future__ import annotations

from robot_sf.gym_env.environment_factory import RecordingOptions, make_pedestrian_env


def test_pedestrian_factory_convenience_record_video(tmp_path):
    """Test pedestrian factory convenience record video.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    env = make_pedestrian_env(record_video=True, video_path=str(tmp_path / "ped.mp4"))
    assert getattr(env, "sim_ui", None) is not None


def test_pedestrian_factory_explicit_options_override():
    """Test pedestrian factory explicit options override.

    Returns:
        Any: Auto-generated placeholder description.
    """
    rec = RecordingOptions(record=False)
    env = make_pedestrian_env(record_video=True, recording_options=rec)
    # Explicit RecordingOptions.record=False prevents recording view creation
    assert getattr(env, "sim_ui", None) is None
