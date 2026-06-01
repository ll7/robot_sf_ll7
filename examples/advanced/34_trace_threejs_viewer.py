"""Export a simulation_trace_export.v1 trace to a static browser viewer.

Usage:
    uv run python examples/advanced/34_trace_threejs_viewer.py
    uv run python examples/advanced/34_trace_threejs_viewer.py path/to/trace.json
    uv run python examples/advanced/34_trace_threejs_viewer.py path/to/trace.json --annotations path/to/annotations.json

Expected Output:
    - Static viewer files under `output/trace_viewer/`.

Limitations:
    - This is a diagnostic-only qualitative viewer. It is not benchmark evidence.
    - Map bounds are auto-computed from trace positions; no SVG map geometry is available.
    - The browser loads Three.js from a CDN, so offline use requires vendoring that asset first.
"""

import sys
from pathlib import Path

from robot_sf.render.trace_viewer import main as viewer_main


def main() -> int:
    """Run the trace viewer exporter against a provided or fixture trace.

    Returns:
        int: Process exit status code.
    """
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        default_args = _default_fixture_args()
        if default_args is None:
            return 1
        args = [*default_args, *args]
    return viewer_main(args)


def _default_fixture_args() -> list[str] | None:
    """Return default trace/annotation arguments for demo runs.

    Returns:
        CLI argument list, or ``None`` when the tracked fixture is unavailable.
    """
    fixture_dir = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "analysis_workbench"
        / "simulation_trace_export_v1"
    )
    fixture = fixture_dir / "planner_sanity_open_episode_0000.json"
    fallback_fixture = fixture_dir / "minimal_trace.json"
    annotation_fixture = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "analysis_workbench"
        / "trace_annotation_set_v1"
        / "issue_1962_planner_sanity_open_annotations.json"
    )
    if fixture.exists() and annotation_fixture.exists():
        return [
            str(fixture),
            "--annotations",
            str(annotation_fixture),
        ]
    if fallback_fixture.exists():
        return [str(fallback_fixture)]
    sys.stderr.write(f"Fixture not found: {fixture}\n")
    return None


if __name__ == "__main__":
    raise SystemExit(main())
