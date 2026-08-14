"""Controlled numba on/off micro-benchmark for the LiDAR raycast hot path.

Measures the per-call wall time of :func:`robot_sf.sensor.range_sensor.raycast`
on a fixed synthetic workload, once with numba JIT compilation enabled and once
with ``NUMBA_DISABLE_JIT=1`` (the ``@njit`` decorators degrade to plain Python).
The ratio of the two medians is the steady-state speedup contributed by numba on
this hot path.

Scope and caveats
-----------------
* This is *diagnostic* timing of a single hot path on one machine and one
  synthetic workload. It is **not** a whole-simulation speedup figure and it is
  **not** a cross-platform guarantee.
* JIT compilation happens during a discarded warmup phase, so the reported
  numbers are steady state, not first-call cost.
* Both arms are run in separate subprocesses so that the JIT toggle, which numba
  reads once at import time, is unambiguous.

Usage
-----
    uv run python scripts/perf/raycast_numba_toggle_benchmark.py \
        --output-dir docs/context/evidence/raycast_numba_toggle_<date>

Single-arm mode (used internally by the driver, but callable directly)::

    uv run python scripts/perf/raycast_numba_toggle_benchmark.py --arm on --json-only
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# Workload contract. These values define the measurement; changing any of them
# produces a different benchmark, not a new sample of this one.
DEFAULT_SEED = 101
DEFAULT_NUM_RAYS = 272
DEFAULT_NUM_SEGMENTS = 60
DEFAULT_NUM_PEDS = 30
DEFAULT_CALLS = 2000
DEFAULT_WARMUP = 50
DEFAULT_REPEATS = 3
MAP_EXTENT = 20.0
MAX_SCAN_DIST = 10.0
PED_RADIUS = 0.4


def build_workload(
    seed: int,
    num_rays: int,
    num_segments: int,
    num_peds: int,
) -> dict[str, Any]:
    """Construct the deterministic synthetic scan geometry.

    Args:
        seed: Seed for the numpy generator producing obstacle and pedestrian layout.
        num_rays: Number of equally spaced rays over the full 360 degree circle.
        num_segments: Number of static obstacle line segments.
        num_peds: Number of circular pedestrian obstacles.

    Returns:
        dict[str, Any]: Keyword arguments accepted by
        :func:`robot_sf.sensor.range_sensor.raycast`.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    scanner_pos = (0.0, 0.0)
    ray_angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False).astype(np.float64)

    # Static obstacles: random short segments scattered over the map extent.
    starts = rng.uniform(-MAP_EXTENT, MAP_EXTENT, size=(num_segments, 2))
    deltas = rng.uniform(-4.0, 4.0, size=(num_segments, 2))
    obstacles = np.hstack([starts, starts + deltas]).astype(np.float64)

    # Pedestrians: uniformly placed inside the scan range so that most are hit.
    ped_pos = rng.uniform(-MAX_SCAN_DIST, MAX_SCAN_DIST, size=(num_peds, 2)).astype(np.float64)

    return {
        "scanner_pos": scanner_pos,
        "obstacles": obstacles,
        "max_scan_range": MAX_SCAN_DIST,
        "ped_pos": ped_pos,
        "ped_radius": PED_RADIUS,
        "ray_angles": ray_angles,
    }


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    """Time the raycast hot path in the current process.

    Args:
        args: Parsed command line arguments; ``args.arm`` labels the JIT state.

    Returns:
        dict[str, Any]: Timing samples and workload/environment identity.
    """
    import numba
    import numpy as np

    from robot_sf.sensor.range_sensor import raycast

    workload = build_workload(args.seed, args.num_rays, args.num_segments, args.num_peds)

    # Warmup: triggers JIT compilation on the "on" arm and warms caches on both.
    for _ in range(args.warmup):
        result = raycast(**workload)

    # Correctness cross-check: both arms must produce identical ranges.
    finite = result[np.isfinite(result)]
    output_signature = {
        "num_rays": int(result.shape[0]),
        "num_finite": int(finite.shape[0]),
        "sum_finite": float(np.round(finite.sum(), 9)),
        "min_finite": float(np.round(finite.min(), 9)) if finite.size else None,
    }

    repeat_samples: list[list[float]] = []
    for _ in range(args.repeats):
        samples_ns: list[int] = []
        for _ in range(args.calls):
            start = time.perf_counter_ns()
            raycast(**workload)
            samples_ns.append(time.perf_counter_ns() - start)
        repeat_samples.append([ns / 1000.0 for ns in samples_ns])  # microseconds

    return {
        "arm": args.arm,
        "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT", "0"),
        "numba_version": numba.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "workload": {
            "seed": args.seed,
            "num_rays": args.num_rays,
            "num_segments": args.num_segments,
            "num_peds": args.num_peds,
            "max_scan_dist_m": MAX_SCAN_DIST,
            "ped_radius_m": PED_RADIUS,
            "map_extent_m": MAP_EXTENT,
        },
        "protocol": {
            "warmup_calls": args.warmup,
            "calls_per_repeat": args.calls,
            "repeats": args.repeats,
            "timer": "time.perf_counter_ns",
        },
        "output_signature": output_signature,
        "samples_us_per_repeat": repeat_samples,
    }


def summarize(samples_us: list[float]) -> dict[str, float]:
    """Summarize a per-call timing sample in microseconds.

    Args:
        samples_us: Per-call durations in microseconds.

    Returns:
        dict[str, float]: Distribution summary (never a bare mean).
    """
    ordered = sorted(samples_us)
    quantiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "n": len(ordered),
        "min_us": ordered[0],
        "p25_us": quantiles[0],
        "median_us": quantiles[1],
        "p75_us": quantiles[2],
        "iqr_us": quantiles[2] - quantiles[0],
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))],
        "p99_us": ordered[int(0.99 * (len(ordered) - 1))],
        "max_us": ordered[-1],
        "mean_us": statistics.fmean(ordered),
        "stdev_us": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def spawn_arm(arm: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run one arm in a clean subprocess with the appropriate JIT setting.

    Args:
        arm: Either ``"on"`` (JIT enabled) or ``"off"`` (``NUMBA_DISABLE_JIT=1``).
        args: Parsed command line arguments carrying the workload contract.

    Returns:
        dict[str, Any]: The arm result payload decoded from the child's stdout.

    Raises:
        RuntimeError: If the child process fails or emits no JSON payload.
    """
    env = dict(os.environ)
    env["NUMBA_DISABLE_JIT"] = "1" if arm == "off" else "0"
    env["PYTHONHASHSEED"] = "0"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--arm",
        arm,
        "--json-only",
        "--seed",
        str(args.seed),
        "--num-rays",
        str(args.num_rays),
        "--num-segments",
        str(args.num_segments),
        "--num-peds",
        str(args.num_peds),
        "--calls",
        str(args.calls),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"arm {arm} failed (rc={proc.returncode}): {proc.stderr[-2000:]}")
    payload_line = next(
        (line for line in reversed(proc.stdout.splitlines()) if line.startswith("{")), None
    )
    if payload_line is None:
        raise RuntimeError(f"arm {arm} produced no JSON payload")
    return json.loads(payload_line)


def _git_commit() -> str:
    """Return the current git HEAD SHA, or ``"unknown"`` outside a checkout.

    Returns:
        str: 40 character commit SHA or ``"unknown"``.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("on", "off"), default=None)
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-rays", type=int, default=DEFAULT_NUM_RAYS)
    parser.add_argument("--num-segments", type=int, default=DEFAULT_NUM_SEGMENTS)
    parser.add_argument("--num-peds", type=int, default=DEFAULT_NUM_PEDS)
    parser.add_argument("--calls", type=int, default=DEFAULT_CALLS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark driver or a single arm.

    Args:
        argv: Optional argument vector.

    Returns:
        int: Process exit status.
    """
    args = parse_args(argv)

    if args.arm is not None:
        payload = run_arm(args)
        print(json.dumps(payload))
        return 0

    started = time.time()
    arms = {arm: spawn_arm(arm, args) for arm in ("on", "off")}
    finished = time.time()

    summaries: dict[str, Any] = {}
    for arm, payload in arms.items():
        pooled = [s for rep in payload["samples_us_per_repeat"] for s in rep]
        summaries[arm] = {
            "pooled": summarize(pooled),
            "per_repeat_median_us": [
                summarize(rep)["median_us"] for rep in payload["samples_us_per_repeat"]
            ],
        }

    on_median = summaries["on"]["pooled"]["median_us"]
    off_median = summaries["off"]["pooled"]["median_us"]
    on_reps = summaries["on"]["per_repeat_median_us"]
    off_reps = summaries["off"]["per_repeat_median_us"]

    outputs_match = arms["on"]["output_signature"] == arms["off"]["output_signature"]

    report = {
        "benchmark": "raycast_numba_toggle",
        "description": (
            "Per-call wall time of robot_sf.sensor.range_sensor.raycast with numba JIT "
            "enabled vs NUMBA_DISABLE_JIT=1, on a fixed synthetic LiDAR workload."
        ),
        "commit": _git_commit(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)),
        "hardware": {
            "machine_model": subprocess.run(
                ["sysctl", "-n", "hw.model"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or platform.machine(),
            "cpu": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or platform.processor(),
            "platform": platform.platform(),
        },
        "workload": arms["on"]["workload"],
        "protocol": arms["on"]["protocol"],
        "environment": {
            "python": arms["on"]["python_version"],
            "numba": arms["on"]["numba_version"],
            "numpy": arms["on"]["numpy_version"],
        },
        "outputs_identical_across_arms": outputs_match,
        "output_signature": arms["on"]["output_signature"],
        "results": summaries,
        "speedup": {
            "median_ratio": off_median / on_median,
            "min_repeat_ratio": min(off_reps) / max(on_reps),
            "max_repeat_ratio": max(off_reps) / min(on_reps),
            "on_median_us": on_median,
            "off_median_us": off_median,
        },
    }

    if args.output_dir is not None:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "raycast_numba_toggle_results.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raw = {arm: arms[arm]["samples_us_per_repeat"] for arm in arms}
        (out_dir / "raycast_numba_toggle_samples.json").write_text(
            json.dumps(raw, indent=None, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        print(f"wrote {out_dir}")

    print(json.dumps(report["speedup"], indent=2))
    if not outputs_match:
        print("WARNING: arms produced different scan outputs; timing is not comparable")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
