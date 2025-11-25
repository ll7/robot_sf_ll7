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

try:
    from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - during early scaffolding

    def run_full_benchmark(cfg):  # fallback placeholder
        raise NotImplementedError("Full benchmark not yet implemented. Follow tasks T022+.")


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
    # Episode planning
    initial_episodes: int = 1
    horizon_override: int | None = None
    max_episodes: int = 0  # 0 -> unlimited until precision criteria
    batch_size: int = 1
    # Adaptive precision thresholds (T040)
    target_collision_half_width: float = 0.05
    target_success_half_width: float = 0.05
    target_snqi_half_width: float = 0.05
    # Video / plots toggles
    disable_videos: bool = False
    max_videos: int = 1
    video_renderer: str = "auto"
    video_fps: int = 10


def build_arg_parser() -> argparse.ArgumentParser:
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
        help="Smoke mode (fast placeholders, minimal output)",
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
    horizon_override = ns.horizon if ns.horizon and ns.horizon > 0 else None
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
        disable_videos=ns.disable_videos,
        max_videos=ns.max_videos,
        video_renderer=ns.video_renderer,
        video_fps=ns.video_fps,
    )


def main(argv: list[str] | None = None) -> int:
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
    run_full_benchmark(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
