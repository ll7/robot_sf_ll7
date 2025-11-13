"""Archived: Interactive playback demo (prefer advanced/15_view_recording.py).

Usage:
    uv run python examples/_archived/interactive_playback_demo.py

Prerequisites:
    - Pre-recorded pickle file under examples/recordings

Expected Output:
    - Pygame window with keyboard-controlled playback of recorded trajectories.

Limitations:
    - Manual workflow; superseded by examples/advanced/15_view_recording.py utilities.

References:
    - examples/advanced/15_view_recording.py
"""

from robot_sf.render.interactive_playback import load_and_play_interactively

if __name__ == "__main__":
    load_and_play_interactively("examples/recordings/2024-12-06_15-39-44.pkl")
