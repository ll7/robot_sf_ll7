"""
This script serves as an entry point to load and play back recorded interactive states.
"""

from robot_sf.render.interactive_playback import load_and_play_interactively

if __name__ == "__main__":
    load_and_play_interactively("examples/recordings/2024-12-06_15-39-44.pkl")
