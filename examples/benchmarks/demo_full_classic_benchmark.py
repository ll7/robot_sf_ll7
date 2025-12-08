"""Run the full classic interaction benchmark via programmatic helper.

Purpose:
    Provide a concise, reproducible example for invoking the full
    classic benchmark orchestrator (adaptive sampling + post-run
    visual artifact generation: plots + videos + manifests). The visuals
    subsystem now supports a renderer toggle and SimulationView-first
    fallback ladder documented below.

What it does:
    1. Resolves repository root and selects a scenario matrix.
    2. Creates an output directory under output/results/ with an informative stem.
    3. Constructs a lightweight BenchmarkCLIConfig (same dataclass the CLI uses).
    4. Calls run_full_benchmark(cfg) directly (programmatic use preferred over pure CLI).
    5. Prints locations of key output artifacts (episodes, aggregates, reports, visuals).

Smoke / fast run defaults:
    - initial_episodes=2, max_episodes=4, batch_size=2 (one adaptive iteration)
    - workers=1 for determinism
    - max_videos=1 to keep runtime low
    - video_renderer defaults to 'auto' (try sim-view then fallback synthetic)

Adjust the parameters at the bottom if you need a longer or heavier run.

Related docs:
    - docs/benchmark_full_classic.md (benchmark overview)
    - docs/benchmark_visuals.md (renderer toggle, lifecycle, dependency matrix)
    - specs/127-enhance-benchmark-visual/* (feature spec + tasks + plan)

Usage:
    uv run python examples/benchmarks/demo_full_classic_benchmark.py

Optional arguments / env vars:
    - Set cfg.video_renderer = 'synthetic' or 'sim-view' to force mode.
    - Env ROBOT_SF_VALIDATE_VISUALS=1 to enable JSON Schema validation.
    - Env ROBOT_SF_PERF_ENFORCE=1 to enforce performance budgets.
    - Disable videos: set cfg.disable_videos = True (skip with note 'disabled').

Renderer fallback ladder (auto mode):
    1. SimulationView path (requires: pygame + replay capture + moviepy/ffmpeg)
    2. Synthetic gradient placeholder video(s)
    3. Skip with diagnostic note if both disabled or insufficient replay

Skip note taxonomy (subset):
    simulation-view-missing | moviepy-missing | insufficient-replay |
    disabled | smoke-mode | render-error:<Type>

NOTE:
    This example intentionally avoids argparse to stay minimal. For full
    flag coverage see scripts/classic_benchmark_full.py.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

from robot_sf.render.helper_catalog import ensure_output_dir
from scripts.classic_benchmark_full import BenchmarkCLIConfig, run_full_benchmark  # type: ignore


def _project_root() -> Path:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    return Path(__file__).resolve().parents[2]


def main() -> int:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    root = _project_root()

    # Scenario matrix (choose one appropriate to your local setup)
    # You can substitute another matrix from configs/scenarios/ or configs/baselines/
    matrix = root / "configs/scenarios/classic_interactions.yaml"
    if not matrix.exists():  # fallback example matrix
        matrix = root / "configs/baselines/example_matrix.yaml"

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / f"output/results/demo_full_classic_{ts}"
    out_dir = ensure_output_dir(out_dir)

    fast_mode = os.getenv("ROBOT_SF_FAST_DEMO", "0") == "1"
    max_steps_env = os.getenv("ROBOT_SF_EXAMPLES_MAX_STEPS")
    horizon_override = int(max_steps_env) if max_steps_env and max_steps_env.isdigit() else None
    capture_replay = not fast_mode
    fast_stub = fast_mode
    smoke_horizon_cap = (horizon_override or 40) if fast_mode else 40

    cfg = BenchmarkCLIConfig(
        scenario_matrix_path=str(matrix),
        output_root=str(out_dir),
        workers=1,
        master_seed=123,
        smoke=fast_mode,
        algo="ppo",  # Label stored in scenario_params.algo for grouping
        capture_replay=capture_replay,
        fast_stub=fast_stub,
        initial_episodes=1 if fast_mode else 2,
        max_episodes=1
        if fast_mode
        else 4,  # Stop after max episodes (small demo) or earlier if precision hit
        batch_size=1 if fast_mode else 2,
        horizon_override=horizon_override,
        smoke_horizon_cap=smoke_horizon_cap,
        target_collision_half_width=0.05,
        target_success_half_width=0.05,
        target_snqi_half_width=0.05,
        disable_videos=fast_mode,
        max_videos=0 if fast_mode else 1,  # Keep runtime small; increase for more examples
    )

    print("[demo_full_classic] Running benchmark...", flush=True)
    run_full_benchmark(cfg)

    # Common artifact paths (existence depends on run and feature flags)
    episodes_jsonl = out_dir / "episodes" / "episodes.jsonl"
    aggregates_dir = out_dir / "aggregates"
    reports_dir = out_dir / "reports"
    plots_dir = out_dir / "plots"
    videos_dir = out_dir / "videos"

    print("[demo_full_classic] Completed.")
    print(f"  Matrix: {matrix}")
    print(f"  Episodes: {episodes_jsonl}")
    print(f"  Aggregates dir: {aggregates_dir}")
    print(f"  Reports dir: {reports_dir}")
    print(f"  Plots dir: {plots_dir}")
    print(f"  Videos dir: {videos_dir}")

    if not episodes_jsonl.exists():
        print("  WARNING: episodes file missing (run may have failed early)")

    perf_meta = reports_dir / "performance_visuals.json"
    if perf_meta.exists():
        print(f"  Visuals performance meta: {perf_meta}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
