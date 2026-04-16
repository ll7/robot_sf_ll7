"""GPU throughput microbenchmark for feature extractors.

Measures inference throughput (observations/second) and latency for each
extractor type at a fixed batch size.  No training, no environment — just
raw forward-pass speed on synthetic observations.

Usage::

    # Quick run (default presets, batch=256, 500 reps)
    uv run python scripts/tools/benchmark_feature_extractors.py

    # Custom batch and reps
    uv run python scripts/tools/benchmark_feature_extractors.py --batch 512 --reps 1000

    # Save results to a JSON file
    uv run python scripts/tools/benchmark_feature_extractors.py --out output/bench_fe.json

    # CPU-only (useful for CI where no GPU is available)
    uv run python scripts/tools/benchmark_feature_extractors.py --device cpu

Results are printed in a Markdown table and optionally written as JSON so
they can be included in context notes or CI artefacts.

Observation space used (matches the default robot environment)
--------------------------------------------------------------
- ``rays``:        shape (1, 272) — one timestep × 272 LiDAR rays
- ``drive_state``: shape (1, 7)  — one timestep × 7 drive state dims

These match the ``DEFAULT_GYM`` observation mode produced by
``robot_sf.gym_env.robot_env.RobotEnv`` with default ``EnvSettings``.
To benchmark a different shape, pass ``--ray-shape`` and ``--drive-shape``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch as th
from gymnasium import spaces
from loguru import logger

from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.feature_extractors.attention_extractor import AttentionFeatureExtractor
from robot_sf.feature_extractors.config import FeatureExtractorPresets
from robot_sf.feature_extractors.lightweight_cnn_extractor import LightweightCNNExtractor
from robot_sf.feature_extractors.lstm_extractor import LSTMFeatureExtractor
from robot_sf.feature_extractors.mlp_extractor import MLPFeatureExtractor
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS

if TYPE_CHECKING:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

_DEFAULT_RAY_SHAPE = (1, 272)    # (timesteps, num_rays)
_DEFAULT_DRIVE_SHAPE = (1, 7)    # (timesteps, drive_dim)
_DEFAULT_BATCH = 256
_DEFAULT_WARMUP = 50
_DEFAULT_REPS = 500
_FPS_WARN_THRESHOLD = 200_000    # obs/s below which a warning is printed


def _make_obs_space(ray_shape: tuple[int, ...], drive_shape: tuple[int, ...]) -> spaces.Dict:
    """Build a synthetic Dict observation space.

    Args:
        ray_shape: LiDAR observation shape (timesteps, num_rays).
        drive_shape: Drive-state observation shape (timesteps, drive_dim).

    Returns:
        ``spaces.Dict`` suitable for feature extractor constructors.
    """
    return spaces.Dict({
        OBS_RAYS: spaces.Box(low=0.0, high=1.0, shape=ray_shape, dtype=np.float32),
        OBS_DRIVE_STATE: spaces.Box(low=-1.0, high=1.0, shape=drive_shape, dtype=np.float32),
    })


def _make_batch(obs_space: spaces.Dict, batch_size: int, device: str) -> dict[str, th.Tensor]:
    """Sample a random observation batch on the target device.

    Args:
        obs_space: Observation space defining shapes and bounds.
        batch_size: Number of observations in the batch.
        device: PyTorch device string ('cuda' or 'cpu').

    Returns:
        Dict mapping observation keys to tensors of shape (batch_size, *shape).
    """
    return {
        key: th.tensor(
            obs_space[key].sample()[None, ...].repeat(batch_size, axis=0),
            dtype=th.float32,
            device=device,
        )
        for key in obs_space.spaces
    }


# ---------------------------------------------------------------------------
# Extractor definitions
# ---------------------------------------------------------------------------

@dataclass
class _ExtractorSpec:
    """Specification for one extractor variant to benchmark.

    Attributes:
        name: Short display name for the benchmark table.
        cls: Feature extractor class.
        kwargs: Constructor kwargs (forwarded to ``cls.__init__``).
    """

    name: str
    cls: type
    kwargs: dict = field(default_factory=dict)


def _build_specs() -> list[_ExtractorSpec]:
    """Return the list of extractor specs to benchmark.

    Returns:
        Ordered list of ``_ExtractorSpec`` instances covering all canonical presets.
    """
    dyn_defaults = FeatureExtractorPresets.dynamics_original()
    return [
        _ExtractorSpec("dynamics_original", DynamicsExtractor, dyn_defaults.params),
        _ExtractorSpec("mlp_small", MLPFeatureExtractor,
                       FeatureExtractorPresets.mlp_small().params),
        _ExtractorSpec("mlp_large", MLPFeatureExtractor,
                       FeatureExtractorPresets.mlp_large().params),
        _ExtractorSpec("lightweight_cnn", LightweightCNNExtractor,
                       FeatureExtractorPresets.lightweight_cnn().params),
        _ExtractorSpec("attention_small", AttentionFeatureExtractor,
                       FeatureExtractorPresets.attention_small().params),
        _ExtractorSpec("attention_large", AttentionFeatureExtractor,
                       FeatureExtractorPresets.attention_large().params),
        _ExtractorSpec("lstm_small", LSTMFeatureExtractor,
                       FeatureExtractorPresets.lstm_small().params),
        _ExtractorSpec("lstm_medium", LSTMFeatureExtractor,
                       FeatureExtractorPresets.lstm_medium().params),
    ]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Results for one extractor variant.

    Attributes:
        name: Extractor display name.
        params: Number of trainable parameters.
        features_dim: Output feature dimension.
        throughput_obs_per_s: Forward-pass throughput in observations per second.
        latency_ms: Mean batch latency in milliseconds.
        speedup: Throughput relative to the first (baseline) extractor.
        slow: Whether throughput falls below the warning threshold.
    """

    name: str
    params: int
    features_dim: int
    throughput_obs_per_s: float
    latency_ms: float
    speedup: float = 1.0
    slow: bool = False


def _count_params(model: "BaseFeaturesExtractor") -> int:
    """Count trainable parameters in a feature extractor.

    Args:
        model: Instantiated feature extractor.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _benchmark_one(
    spec: _ExtractorSpec,
    *,
    obs_space: spaces.Dict,
    batch: dict[str, th.Tensor],
    warmup: int,
    reps: int,
    device: str,
) -> BenchmarkResult:
    """Run the forward-pass benchmark for one extractor.

    Args:
        spec: Extractor specification.
        obs_space: Observation space used to construct the extractor.
        batch: Pre-allocated input batch on the target device.
        warmup: Number of warm-up iterations (not timed).
        reps: Number of timed repetitions.
        device: PyTorch device string.

    Returns:
        ``BenchmarkResult`` with throughput, latency, and parameter counts.
    """
    model = spec.cls(obs_space, **spec.kwargs).to(device)
    model.eval()
    n_params = _count_params(model)
    features_dim = model.features_dim

    with th.no_grad():
        for _ in range(warmup):
            model(batch)
        if device == "cuda":
            th.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(reps):
            model(batch)
        if device == "cuda":
            th.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    total_obs = reps * batch[OBS_RAYS].shape[0]
    throughput = total_obs / elapsed
    latency_ms = (elapsed / reps) * 1000.0

    return BenchmarkResult(
        name=spec.name,
        params=n_params,
        features_dim=features_dim,
        throughput_obs_per_s=throughput,
        latency_ms=latency_ms,
        slow=throughput < _FPS_WARN_THRESHOLD,
    )


def run_benchmark(
    *,
    ray_shape: tuple[int, ...],
    drive_shape: tuple[int, ...],
    batch_size: int,
    warmup: int,
    reps: int,
    device: str,
) -> list[BenchmarkResult]:
    """Run the full benchmark suite.

    Args:
        ray_shape: LiDAR observation shape.
        drive_shape: Drive-state observation shape.
        batch_size: Number of observations per forward pass.
        warmup: Warm-up iterations per extractor.
        reps: Timed repetitions per extractor.
        device: Target device ('cuda' or 'cpu').

    Returns:
        List of ``BenchmarkResult`` instances, one per extractor.
    """
    obs_space = _make_obs_space(ray_shape, drive_shape)
    batch = _make_batch(obs_space, batch_size, device)
    specs = _build_specs()
    results: list[BenchmarkResult] = []
    baseline_throughput: float | None = None

    for spec in specs:
        logger.info("Benchmarking {} ...", spec.name)
        try:
            r = _benchmark_one(
                spec,
                obs_space=obs_space,
                batch=batch,
                warmup=warmup,
                reps=reps,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("  {} failed: {}", spec.name, exc)
            continue

        if baseline_throughput is None:
            baseline_throughput = r.throughput_obs_per_s
        r.speedup = r.throughput_obs_per_s / baseline_throughput if baseline_throughput else 1.0
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_table(results: list[BenchmarkResult], *, device: str, batch_size: int) -> None:
    """Print benchmark results as a Markdown table.

    Args:
        results: Benchmark results to display.
        device: Device string shown in the table header.
        batch_size: Batch size shown in the table header.
    """
    print(f"\n## Feature extractor throughput — device={device}, batch={batch_size}\n")
    header = (
        f"{'Extractor':<22} {'Params':>10} {'Features':>9} "
        f"{'Throughput (obs/s)':>20} {'Latency (ms)':>14} {'Speedup':>9}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in results:
        flag = " [SLOW]" if r.slow else ""
        print(
            f"{r.name:<22} {r.params:>10,} {r.features_dim:>9} "
            f"{r.throughput_obs_per_s:>20,.0f} {r.latency_ms:>14.3f} "
            f"{r.speedup:>8.2f}×{flag}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_shape(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated shape string into a tuple of ints.

    Args:
        raw: Comma-separated integers, e.g. '1,272'.

    Returns:
        Shape tuple.
    """
    return tuple(int(x) for x in raw.split(","))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(description="Feature extractor GPU throughput benchmark.")
    p.add_argument("--batch", type=int, default=_DEFAULT_BATCH,
                   help=f"Batch size (default: {_DEFAULT_BATCH}).")
    p.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP,
                   help=f"Warm-up iterations (default: {_DEFAULT_WARMUP}).")
    p.add_argument("--reps", type=int, default=_DEFAULT_REPS,
                   help=f"Timed repetitions (default: {_DEFAULT_REPS}).")
    p.add_argument("--device", default="auto",
                   help="Device: 'cuda', 'cpu', or 'auto' (default: auto).")
    p.add_argument("--ray-shape", default=",".join(str(d) for d in _DEFAULT_RAY_SHAPE),
                   help=f"LiDAR shape as T,N (default: {_DEFAULT_RAY_SHAPE}).")
    p.add_argument("--drive-shape", default=",".join(str(d) for d in _DEFAULT_DRIVE_SHAPE),
                   help=f"Drive-state shape as T,D (default: {_DEFAULT_DRIVE_SHAPE}).")
    p.add_argument("--out", default=None, type=Path,
                   help="Optional output path for JSON results (e.g. output/bench_fe.json).")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print results."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"

    ray_shape = _parse_shape(args.ray_shape)
    drive_shape = _parse_shape(args.drive_shape)

    logger.info(
        "Benchmark: device={} batch={} reps={} ray_shape={} drive_shape={}",
        device, args.batch, args.reps, ray_shape, drive_shape,
    )

    results = run_benchmark(
        ray_shape=ray_shape,
        drive_shape=drive_shape,
        batch_size=args.batch,
        warmup=args.warmup,
        reps=args.reps,
        device=device,
    )

    _print_table(results, device=device, batch_size=args.batch)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "device": device,
            "batch_size": args.batch,
            "warmup": args.warmup,
            "reps": args.reps,
            "ray_shape": list(ray_shape),
            "drive_shape": list(drive_shape),
            "results": [asdict(r) for r in results],
        }
        args.out.write_text(json.dumps(payload, indent=2))
        logger.success("Results written to {}", args.out)

    slow = [r.name for r in results if r.slow]
    if slow:
        logger.warning("Slow extractors (< {:,} obs/s): {}", _FPS_WARN_THRESHOLD, ", ".join(slow))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
