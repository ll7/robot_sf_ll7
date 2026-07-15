#!/usr/bin/env python3
"""One-command Robot SF visual demo: install -> run -> see something.

This is the artifact-producing entry point behind ``robot-sf demo``. It runs a
single tiny deterministic episode (``quickstart_demo_crossing_basic`` + the
built-in ``random`` planner), records it as JSONL, exports a static Three.js
viewer, renders a map thumbnail, and writes a plain-English summary.

The experience is intentionally stable and reviewable: a fixed seed drives both
the environment and the planner, so re-running the demo reproduces the same
artifacts byte-for-byte on a clean CPU checkout.

Artifacts land under ``output/demo/latest/``:

    episode.jsonl        - per-step recorded simulation states
    summary.json         - plain-English run summary + outcome metadata
    metrics.json         - machine-readable outcome metrics
    viewer/index.html    - static browser viewer built from the recording
    thumbnail.png        - top-down map/route thumbnail

Usage:
    uv run robot-sf demo
    uv run python scripts/demo/quickstart_demo.py --output-root output/demo/latest --seed 270
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from loguru import logger

# Keep the one-command demo readable.
logger.remove()
logger.add(sys.stderr, level="ERROR")

from robot_sf.gym_env.environment_factory import make_robot_env  # noqa: E402
from robot_sf.gym_env.robot_env import RobotEnv  # noqa: E402
from robot_sf.render.jsonl_playback import JSONLPlaybackLoader  # noqa: E402
from robot_sf.render.threejs_viewer import export_threejs_viewer  # noqa: E402
from robot_sf.training.scenario_loader import (  # noqa: E402
    build_robot_config_from_scenario,
    load_scenarios,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO = ROOT / "configs/scenarios/single/quickstart_demo.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "output/demo/latest"
DEFAULT_SEED = 270
DEMO_SCENARIO_NAME = "quickstart_demo_crossing_basic"
DEMO_PLANNER = "random"
CLAIM_BOUNDARY = "Local demo output is reproducibility/UX evidence only; not a benchmark result."


@dataclass
class DemoResult:
    """Outcomes and artifact paths produced by the one-command demo."""

    scenario_name: str
    planner: str
    seed: int
    steps: int
    collisions: int
    min_ped_distance_m: float | None
    route_complete: bool
    episode_jsonl: Path
    summary_json: Path
    metrics_json: Path
    viewer_html: Path
    thumbnail_png: Path

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable plain-English summary payload."""
        return {
            "schema_version": "robot_sf_quickstart_demo.v1",
            "scenario": self.scenario_name,
            "planner": self.planner,
            "seed": self.seed,
            "steps": self.steps,
            "collisions": self.collisions,
            "min_ped_distance_m": self.min_ped_distance_m,
            "route_complete": self.route_complete,
            "claim_boundary": CLAIM_BOUNDARY,
            "artifacts": {
                "episode_jsonl": str(self.episode_jsonl),
                "summary_json": str(self.summary_json),
                "metrics_json": str(self.metrics_json),
                "viewer_html": str(self.viewer_html),
                "thumbnail_png": str(self.thumbnail_png),
            },
        }


def _build_scenario_config(scenario_path: Path) -> dict[str, Any]:
    """Load the demo scenario definition from the YAML matrix."""
    scenarios = load_scenarios(scenario_path)
    matches = [sc for sc in scenarios if sc.get("name") == DEMO_SCENARIO_NAME]
    if not matches:
        available = [sc.get("name") for sc in scenarios]
        raise ValueError(
            f"Demo scenario '{DEMO_SCENARIO_NAME}' not found in {scenario_path}; "
            f"available: {available}",
        )
    return matches[0]


def _select_action(planner: Any) -> np.ndarray:
    """Convert a planner decision into the env-action array.

    The built-in random planner ignores observation contents (it samples
    actions), so we pass a minimal planner-facing mapping rather than the
    full environment observation dict. The robot lidar env expects a
    ``(vx, vy)`` action vector.
    """
    decision = planner.step({"dt": 0.1, "robot": {}, "agents": []})
    return np.array([float(decision["vx"]), float(decision["vy"])], dtype=np.float32)


def _step_min_ped_distance(env: RobotEnv) -> float | None:
    """Return the closest pedestrian distance to the robot at the current step.

    ``RobotEnv.step`` does not surface ``min_ped_distance`` in its ``info``
    payload by default (it only appears in the optional telemetry stream), so
    the demo computes the headline summary metric directly from the simulator
    state. Returns ``None`` when there are no pedestrians.
    """
    simulator = getattr(env, "simulator", None)
    if simulator is None:
        return None
    ped_positions = np.asarray(getattr(simulator, "ped_pos", np.zeros((0, 2))), dtype=float)
    robot_poses = getattr(simulator, "robot_poses", None)
    if ped_positions.size == 0 or not robot_poses:
        return None
    try:
        robot_pos = np.asarray(robot_poses[0][0], dtype=float)
    except (IndexError, TypeError, ValueError):  # Defensive: malformed pose
        return None
    deltas = ped_positions - robot_pos
    if deltas.shape[1] < 2:
        return None
    return float(np.min(np.linalg.norm(deltas[:, :2], axis=1)))


def _run_deterministic_episode(
    *,
    scenario_config: dict[str, Any],
    scenario_path: Path,
    seed: int,
    recording_dir: Path,
) -> DemoResult:
    """Run one deterministic recorded episode and capture outcome metrics."""
    from robot_sf.baselines.random_policy import RandomPlanner

    config = build_robot_config_from_scenario(
        scenario_config,
        scenario_path=scenario_path,
    )
    planner = RandomPlanner({"mode": "velocity", "v_max": 1.5}, seed=seed)

    env: RobotEnv = make_robot_env(  # type: ignore[assignment]
        config=config,
        seed=seed,
        debug=False,
        recording_enabled=True,
        use_jsonl_recording=True,
        recording_dir=str(recording_dir),
        suite_name="demo",
        scenario_name=DEMO_SCENARIO_NAME,
        algorithm_name=DEMO_PLANNER,
        recording_seed=seed,
    )
    env.applied_seed = seed

    try:
        env.reset(seed=seed)
        planner.reset(seed=seed)

        steps = 0
        collisions = 0
        min_ped_distance: float | None = None
        route_complete = False

        truncated = False
        while not truncated:
            action = _select_action(planner)
            _obs, _reward, terminated, truncated, info = env.step(action)
            steps += 1
            if bool(info.get("collision")):
                collisions += 1
            step_min = _step_min_ped_distance(env)
            if step_min is not None:
                min_ped_distance = (
                    step_min if min_ped_distance is None else min(min_ped_distance, step_min)
                )
            if bool(info.get("success")):
                route_complete = True
            if terminated:
                break

        env.end_episode_recording()
    finally:
        env.close_recorder()
        env.exit()

    # The JSONLRecorder names files as ``{suite}_{scenario}_{algorithm}_{seed}_ep{id}.jsonl``;
    # prefer that, falling back to the most recent recording in the directory.
    episode_jsonl = recording_dir / f"demo_{DEMO_SCENARIO_NAME}_{DEMO_PLANNER}_{seed}_ep0000.jsonl"
    if not episode_jsonl.exists():
        candidates = sorted(recording_dir.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"No recorded episode JSONL found under {recording_dir}",
            )
        episode_jsonl = candidates[-1]

    return DemoResult(
        scenario_name=DEMO_SCENARIO_NAME,
        planner=DEMO_PLANNER,
        seed=seed,
        steps=steps,
        collisions=collisions,
        min_ped_distance_m=min_ped_distance,
        route_complete=route_complete,
        episode_jsonl=episode_jsonl,
        summary_json=recording_dir.parent / "summary.json",
        metrics_json=recording_dir.parent / "metrics.json",
        viewer_html=recording_dir.parent / "viewer" / "index.html",
        thumbnail_png=recording_dir.parent / "thumbnail.png",
    )


def _export_viewer(episode_jsonl: Path, output_root: Path) -> Path:
    """Export the recorded episode into a static browser viewer."""
    viewer_dir = output_root / "viewer"
    result = export_threejs_viewer(episode_jsonl, viewer_dir)
    return result.html_path


def _render_thumbnail(episode_jsonl: Path, output_root: Path) -> Path:
    """Render a top-down map thumbnail from the recorded episode metadata."""
    from robot_sf.maps.map_visualizer import visualize_map_definition

    _episode, map_def = JSONLPlaybackLoader().load_single_episode(episode_jsonl)
    thumbnail_path = output_root / "thumbnail.png"
    visualize_map_definition(
        map_def,
        output_path=thumbnail_path,
        title=f"{DEMO_SCENARIO_NAME} ({DEMO_PLANNER})",
    )
    return thumbnail_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable, sorted JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check_runtime_requirements(*, strict: bool = False) -> bool:
    """Run the shared runtime-requirements doctor check (advisory by default).

    Reuses ``scripts/dev/check_runtime_requirements.sh`` so the demo reports the
    same core-tool status as the rest of the project. In advisory mode the check
    only fails on missing *required* core tools; optional capabilities (docker,
    SLURM, GPU) are reported but never fail the demo. Returns ``True`` when the
    required core tools are present.
    """
    script = ROOT / "scripts" / "dev" / "check_runtime_requirements.sh"
    if not script.exists():
        print("Skipping runtime check (scripts/dev/check_runtime_requirements.sh not found).")
        return True
    cmd = ["bash", str(script)]
    if strict:
        cmd.append("--strict")
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        print("Runtime check: skipped (bash not available).")
        return True
    ok = completed.returncode == 0
    print(f"Runtime check: {'ok' if ok else 'missing required tool(s)'}")
    # Surface the full doctor report only when something is wrong, so the
    # default newcomer flow stays readable (a clean check prints one line).
    if not ok:
        if completed.stdout:
            sys.stdout.write(completed.stdout)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
    return ok


def run_demo(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    scenario_path: Path = DEFAULT_SCENARIO,
    seed: int = DEFAULT_SEED,
    check_deps: bool = True,
) -> DemoResult:
    """Run the one-command demo and write all artifacts under ``output_root``."""
    output_root = Path(output_root)
    scenario_path = Path(scenario_path)
    recording_dir = output_root / "recordings"

    if check_deps:
        _check_runtime_requirements()

    scenario_config = _build_scenario_config(scenario_path)
    result = _run_deterministic_episode(
        scenario_config=scenario_config,
        scenario_path=scenario_path,
        seed=seed,
        recording_dir=recording_dir,
    )

    _copy_episode_result(result, recording_dir, output_root)
    viewer_html = _export_viewer(result.episode_jsonl, output_root)
    thumbnail_png = _render_thumbnail(result.episode_jsonl, output_root)

    result.viewer_html = viewer_html
    result.thumbnail_png = thumbnail_png

    summary_path = output_root / "summary.json"
    metrics_path = output_root / "metrics.json"
    result.summary_json = summary_path
    result.metrics_json = metrics_path

    _write_json(summary_path, result.to_summary_dict())
    _write_json(
        metrics_path,
        {
            "schema_version": "robot_sf_quickstart_demo.metrics.v1",
            "scenario": result.scenario_name,
            "planner": result.planner,
            "seed": result.seed,
            "steps": result.steps,
            "collisions": result.collisions,
            "min_ped_distance_m": result.min_ped_distance_m,
            "route_complete": result.route_complete,
            "episode_jsonl": str(result.episode_jsonl),
            "viewer_html": str(viewer_html),
            "thumbnail_png": str(thumbnail_png),
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    return result


def _copy_episode_result(result: DemoResult, recording_dir: Path, output_root: Path) -> None:
    """Promote the recorded episode JSONL to ``output_root/episode.jsonl`` for stability."""
    if result.episode_jsonl.exists():
        target = output_root / "episode.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result.episode_jsonl, target)
        result.episode_jsonl = target


def _print_summary(result: DemoResult) -> None:
    """Print a plain-English summary of what the demo did."""
    min_dist = (
        f"{result.min_ped_distance_m:.2f} m" if result.min_ped_distance_m is not None else "n/a"
    )
    print("Robot SF demo complete.")
    print(
        f"Scenario: {result.scenario_name}   Planner: {result.planner}   Steps: {result.steps}",
    )
    print(f"Collisions: {result.collisions}   Min pedestrian distance: {min_dist}")
    print(f"Viewer: {result.viewer_html}   Metrics: {result.metrics_json}")
    print(CLAIM_BOUNDARY)


def build_parser() -> argparse.ArgumentParser:
    """Build the demo command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-check-deps",
        action="store_true",
        help="Skip the shared runtime-requirements doctor check.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed simulator logs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the one-command demo CLI."""
    args = build_parser().parse_args(argv)
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    result = run_demo(
        output_root=args.output_root,
        scenario_path=args.scenario,
        seed=args.seed,
        check_deps=not args.no_check_deps,
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
