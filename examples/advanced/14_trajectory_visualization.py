"""Visualize trajectories during interactive playback sessions.

Fixture smoke path: run with ``--fixture --headless --out-dir <dir>`` to
generate a deterministic recording and render frames plus ``summary.json``
without a display or a pre-existing recording.

Usage:
    uv run python examples/advanced/14_trajectory_visualization.py <recording.pkl>
    uv run python examples/advanced/14_trajectory_visualization.py --fixture --headless \\
        --out-dir output/example-trajectory-visualization

Prerequisites:
    - Explicit recording path for the interactive path; none for the
      bare or ``--fixture`` smoke path (a fixture is generated).

Expected Output:
    - Interactive: playback window with trajectory overlays enabled by default.
    - Headless: PNG frame sequence plus ``summary.json`` under ``--out-dir``.

Limitations:
    - Interactive playback requires a GUI display.
    - Fixture visuals prove plumbing only, not physical or behavioral validity.

References:
    - docs/SIM_VIEW.md
"""

import argparse
import sys

from loguru import logger

from examples.advanced.trajectory_viz_fixture import (
    REASON_OK,
    build_fixture_recording,
    resolve_out_dir,
    resolve_recording_path,
    run_headless,
)
from robot_sf.render.interactive_playback import InteractivePlayback, load_states


def demonstrate_trajectory_visualization(recording_file: str):
    """
    Load and demonstrate trajectory visualization with an interactive playback.

    Args:
        recording_file: Path to the pickle file containing recorded states
    """
    try:
        logger.info(f"Loading recording from: {recording_file}")
        states, map_def = load_states(recording_file)

        logger.info(f"Loaded {len(states)} states for trajectory demo")
        logger.info("Starting interactive playback with trajectory visualization...")
        logger.info("Press 'V' to toggle trajectory display")
        logger.info("Press 'B' or 'C' to adjust trail length")
        logger.info("Press 'X' to clear trajectories")
        logger.info("Press 'H' for full help")

        # Create interactive playback with trajectory visualization enabled
        playback = InteractivePlayback(states, map_def)

        # Enable trajectories by default for the demo
        playback.show_trajectories = True
        # Apply trail length using public API so existing deques are reconfigured
        playback.set_trail_length(50)  # Start with moderate trail length

        logger.info("Trajectory visualization enabled by default")
        logger.info(f"Initial trail length: {playback.max_trajectory_length}")

        # Run the interactive playback
        playback.run()

    except FileNotFoundError:
        logger.error(f"Recording file not found: {recording_file}")
        logger.info("Please provide a valid recording file path")
    except Exception as e:
        logger.error(f"Error during trajectory demo: {e}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for interactive and headless fixture modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording", nargs="?", default=None, help="Recording .pkl for interactive playback"
    )
    parser.add_argument(
        "--fixture", action="store_true", help="Generate a deterministic fixture recording"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Render frames without opening a window"
    )
    parser.add_argument(
        "--out-dir",
        default="output/example-trajectory-visualization",
        help="Caller-owned output directory",
    )
    parser.add_argument(
        "--max-frames", type=int, default=6, help="Bound on rendered frames in headless mode"
    )
    parser.add_argument(
        "--fixture-steps", type=int, default=12, help="State count for a generated fixture"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point routing interactive, fixture, and headless modes."""
    args = build_parser().parse_args(argv)

    recording = args.recording
    output_dir = None
    if recording is None or args.fixture or args.headless:
        output_dir, reason = resolve_out_dir(args.out_dir)
        if output_dir is None:
            logger.error(f"Output directory refused: {reason} (out-dir: {args.out_dir})")
            return 2

    if recording is None:
        # Bare invocation is always the CI-safe fixture smoke path. It never
        # auto-opens interactive playback: a stale recording on disk must not
        # hang headless runs waiting for user input.
        assert output_dir is not None
        fixture_path = output_dir / "fixture_recording.pkl"
        build_fixture_recording(fixture_path, steps=args.fixture_steps)
        logger.info(f"Generated fixture recording at: {fixture_path}")
        recording = str(fixture_path)
        args.headless = True
    elif args.fixture:
        assert output_dir is not None
        fixture_path = output_dir / "fixture_recording.pkl"
        build_fixture_recording(fixture_path, steps=args.fixture_steps)
        logger.info(f"Generated fixture recording at: {fixture_path}")
        recording = str(fixture_path)

    if args.headless:
        assert output_dir is not None
        summary, reason = run_headless(recording, output_dir, max_frames=args.max_frames)
        if reason != REASON_OK:
            logger.error(f"Headless run refused: {reason} (recording: {recording})")
            return 2
        assert summary is not None
        logger.info(
            f"Headless run wrote {summary['rendered_frames']} frames; summary: {summary['summary_path']}"
        )
        return 0

    resolved, reason = resolve_recording_path(recording)
    if resolved is None:
        logger.error(f"Recording refused: {reason} (recording: {recording})")
        return 2
    demonstrate_trajectory_visualization(str(resolved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
