"""Export a JSONL or pickle recording to a static browser viewer.

Usage:
    uv run python examples/advanced/33_threejs_recording_viewer.py
    uv run python examples/advanced/33_threejs_recording_viewer.py path/to/episode.jsonl

Expected Output:
    - Static viewer files under `output/threejs_viewer/`.

Limitations:
    - This is an optional qualitative playback view. Pygame remains the reference renderer.
    - The browser loads Three.js from a CDN, so offline use requires vendoring that asset first.
"""

import json
import sys
from pathlib import Path

from robot_sf.common.artifact_paths import resolve_artifact_path
from robot_sf.render.threejs_viewer import main as viewer_main


def _write_demo_recording() -> Path:
    """Write a tiny self-contained JSONL recording for smoke/demo runs.

    Returns:
        Path: Path to the generated JSONL recording.
    """
    demo_dir = Path(resolve_artifact_path("output/threejs_viewer_demo"))
    demo_dir.mkdir(parents=True, exist_ok=True)
    recording = demo_dir / "demo_episode.jsonl"
    records = [
        {
            "episode_id": 0,
            "step_idx": step,
            "event": "step",
            "timestamp": float(step),
            "state": {
                "timestep": step,
                "robot_pose": [[1.0 + step, 2.0], 0.1 * step],
                "pedestrian_positions": [[3.0, 4.0]],
                "ray_vecs": [[[1.0 + step, 2.0], [2.0 + step, 3.0]]],
            },
        }
        for step in range(3)
    ]
    recording.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return recording


def main() -> int:
    """Run the viewer exporter against a provided or generated recording.

    Returns:
        int: Process exit status code.
    """
    args = sys.argv[1:]
    if not args:
        args = [str(_write_demo_recording())]
    return viewer_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
