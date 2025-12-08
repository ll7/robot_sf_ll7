"""Replay alignment and export smoke tests."""

from __future__ import annotations

import numpy as np

from robot_sf.telemetry.history import TelemetryReplay
from robot_sf.telemetry.visualization import export_combined_image


def test_replay_alignment_within_tolerance():
    """Replay scrub stays within tolerance of requested frame."""
    samples = [
        {"frame_idx": 0, "metrics": {"fps": 1.0}},
        {"frame_idx": 5, "metrics": {"fps": 2.0}},
        {"frame_idx": 10, "metrics": {"fps": 3.0}},
    ]
    replay = TelemetryReplay(samples=samples)
    snap = replay.scrub_to_frame(6, tolerance=1)
    assert snap["frame_idx"] == 5


def test_export_combined_image(tmp_path):
    """Combined export writes a PNG with both view and pane content."""
    main = np.zeros((100, 150, 4), dtype=np.uint8)
    main[..., 0] = 255  # red
    pane = np.zeros((50, 80, 4), dtype=np.uint8)
    pane[..., 1] = 255  # green
    out = tmp_path / "combined.png"
    export_combined_image(main, pane, out_path=str(out), layout="vertical_split")
    assert out.exists()
