"""CLI entry script for the Full Classic Interaction Benchmark.

Tasks:
    - T002 initial scaffold (completed earlier)
    - T039 expand parser with full benchmark flags
    - T040 add adaptive precision threshold flags

This script remains lightweight, delegating all logic to library functions.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime


class FullBenchmarkUnavailableError(RuntimeError):
    """Raised when the full classic benchmark backend cannot be imported."""


def _full_benchmark_unavailable_message(exc: BaseException) -> str:
    """Build an actionable fail-closed message for unavailable benchmark backend imports."""
    return (
        "Full Classic Interaction Benchmark is unavailable because "
        "`robot_sf.benchmark.full_classic.orchestrator.run_full_benchmark` could not be imported. "
        "Use the supported `robot_sf.benchmark.full_classic` package path in an initialized "
        "checkout, or run `uv sync --all-extras` before invoking this legacy CLI wrapper. "
        f"Import error: {exc}"
    )


try:
    from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark  # type: ignore
except (ImportError, ModuleNotFoundError) as _run_full_benchmark_import_error:
    _run_full_benchmark_unavailable_message = _full_benchmark_unavailable_message(
        _run_full_benchmark_import_error
    )

    def run_full_benchmark(cfg):
        """Fail closed when the full-classic benchmark backend is unavailable.

        Args:
            cfg: Parsed benchmark configuration, unused because execution cannot continue.
        """
        raise FullBenchmarkUnavailableError(_run_full_benchmark_unavailable_message)


@dataclass
class BenchmarkCLIConfig:
    """Container passed to run_full_benchmark.

    Captures both core execution parameters and adaptive precision thresholds.
    Only fields actually consumed by benchmark modules are included.
    """

    scenario_matrix_path: str
    output_root: str
    workers: int = 1
    master_seed: int = 123
    smoke: bool = False
    algo: str = "unknown"
    capture_replay: bool = True
    fast_stub: bool = False
    smoke_horizon_cap: int = 40
    # Episode planning
    initial_episodes: int = 1
    horizon_override: int | None = None
    max_episodes: int = 0  # 0 -> unlimited until precision criteria
    batch_size: int = 1
    # Adaptive precision thresholds (T040)
    target_collision_half_width: float = 0.05
    target_success_half_width: float = 0.05
    target_snqi_half_width: float = 0.05
    bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int | None = None
    # Resampling mode (issue #5139): "flat" (i.i.d. episode bootstrap, default)
    # or "hierarchical" (two-stage cluster bootstrap). In hierarchical mode the
    # cluster field is selected by bootstrap_cluster ("scenario" -> scenario_id,
    # "seed" -> seed).
    bootstrap_mode: str = "flat"
    bootstrap_cluster: str = "scenario"
    # Video / plots toggles
    disable_videos: bool = False
    max_videos: int = 1
    video_renderer: str = "auto"
    video_fps: int = 10
    freeze_manifest_path: str | None = None
    metrics_subset: list[str] | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for full classic benchmark runs.

    Returns:
        Configured ``argparse.ArgumentParser`` with all benchmark runtime,
        precision, visualization, and freeze-manifest flags.
    """
    # Generate timestamp for default output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output = f"tmp/results/full_classic_run_{timestamp}"

    parser = argparse.ArgumentParser(description="Full Classic Interaction Benchmark")
    # Core paths (now optional with suggested defaults)
    parser.add_argument(
        "--scenarios",
        default="configs/scenarios/classic_interactions.yaml",
        help="Scenario matrix YAML path",
    )
    parser.add_argument("--output", default=default_output, help="Benchmark output root directory")
    # Execution controls
    parser.add_argument("--workers", type=int, default=2, help="Parallel worker processes")
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Master seed for deterministic planning",
    )
    parser.add_argument("--algo", default="ppo", help="Algorithm label for manifest/records")
    parser.add_argument(
        "--initial-episodes",
        type=int,
        default=2,
        help="Initial per-scenario episode count",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=4,
        help="Maximum per-scenario episodes (0=unbounded until precision)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Episodes to add per adaptive sampling iteration",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="Optional horizon override for all episodes (0 means default)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode (fast smoke-only diagnostics, minimal output)",
    )
    parser.add_argument(
        "--capture-replay",
        dest="capture_replay",
        action="store_true",
        help="Capture per-step replay data for videos/plots (default: on)",
    )
    parser.add_argument(
        "--no-capture-replay",
        dest="capture_replay",
        action="store_false",
        help="Disable replay capture (faster, no videos)",
    )
    parser.set_defaults(capture_replay=True)
    # Precision thresholds (half-width targets)
    parser.add_argument(
        "--target-collision-half-width",
        type=float,
        default=0.05,
        help="Target CI half-width for collision_rate",
    )
    parser.add_argument(
        "--target-success-half-width",
        type=float,
        default=0.05,
        help="Target CI half-width for success_rate",
    )
    parser.add_argument(
        "--target-snqi-half-width",
        type=float,
        default=0.05,
        help="Target CI half-width for snqi metric",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap sample count for aggregate confidence intervals",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level for aggregate confidence intervals",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="Bootstrap RNG seed (default: master seed)",
    )
    parser.add_argument(
        "--bootstrap-mode",
        default="flat",
        choices=("flat", "hierarchical"),
        help=(
            "Resampling mode for aggregate confidence intervals (issue #5139): "
            "'flat' i.i.d. episode bootstrap (default) or 'hierarchical' "
            "two-stage cluster bootstrap (scenario-then-episode)."
        ),
    )
    parser.add_argument(
        "--bootstrap-cluster",
        default="scenario",
        choices=("scenario", "seed"),
        help=(
            "Cluster field for hierarchical bootstrap mode (issue #5139): "
            "'scenario' groups episodes by scenario_id (default), 'seed' by seed "
            "for a seed-level cluster bootstrap. Only used when bootstrap-mode is "
            "hierarchical."
        ),
    )
    parser.add_argument(
        "--metrics-subset",
        default="",
        help="Comma-separated metric subset for freeze-contract tracking",
    )
    parser.add_argument(
        "--freeze-manifest",
        default=None,
        help="Optional freeze manifest file (YAML/JSON) for reproducibility checks",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"),
        help="Log level for benchmark run (default: INFO)",
    )
    # Visualization toggles
    parser.add_argument(
        "--disable-videos",
        action="store_true",
        help="Disable video generation even outside smoke mode",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=1,
        help="Maximum number of representative videos to render",
    )
    parser.add_argument(
        "--video-renderer",
        default="auto",
        choices=("auto", "synthetic", "sim-view"),
        help="Video renderer backend preference",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Target FPS for rendered videos",
    )
    return parser


def _args_to_config(ns: argparse.Namespace) -> BenchmarkCLIConfig:
    """Convert parsed CLI arguments into a benchmark config object.

    Args:
        ns: Parsed command-line namespace from ``build_arg_parser``.

    Returns:
        ``BenchmarkCLIConfig`` populated from CLI arguments with normalized
        optional fields (for example, horizon and metric subset parsing).
    """
    horizon_override = ns.horizon if ns.horizon and ns.horizon > 0 else None
    metrics_subset = [
        metric.strip() for metric in str(ns.metrics_subset).split(",") if metric.strip()
    ]
    return BenchmarkCLIConfig(
        scenario_matrix_path=ns.scenarios,
        output_root=ns.output,
        workers=ns.workers,
        master_seed=ns.seed,
        smoke=ns.smoke,
        algo=ns.algo,
        capture_replay=ns.capture_replay,
        initial_episodes=ns.initial_episodes,
        horizon_override=horizon_override,
        max_episodes=ns.max_episodes,
        batch_size=ns.batch_size,
        target_collision_half_width=ns.target_collision_half_width,
        target_success_half_width=ns.target_success_half_width,
        target_snqi_half_width=ns.target_snqi_half_width,
        bootstrap_samples=ns.bootstrap_samples,
        bootstrap_confidence=ns.bootstrap_confidence,
        bootstrap_seed=ns.bootstrap_seed,
        bootstrap_mode=ns.bootstrap_mode,
        bootstrap_cluster=ns.bootstrap_cluster,
        disable_videos=ns.disable_videos,
        max_videos=ns.max_videos,
        video_renderer=ns.video_renderer,
        video_fps=ns.video_fps,
        freeze_manifest_path=ns.freeze_manifest,
        metrics_subset=metrics_subset or None,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the Full Classic Interaction Benchmark CLI.

    Args:
        argv: Optional CLI argument list. When ``None``, argparse reads ``sys.argv``.

    Returns:
        Process exit code. Returns ``2`` when the backend import is unavailable.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # Configure loguru level early
    try:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level=args.log_level.upper())
    except Exception:
        pass
    cfg = _args_to_config(args)
    print("[classic_benchmark_full] Configuration:", cfg)
    try:
        run_full_benchmark(cfg)
    except FullBenchmarkUnavailableError as exc:
        print(f"[classic_benchmark_full] ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
