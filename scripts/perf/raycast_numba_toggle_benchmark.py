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
        emit_bundle(args.output_dir, report, arms)
        print(f"wrote bundle {args.output_dir}")

    print(json.dumps(report["speedup"], indent=2))
    if not outputs_match:
        print("WARNING: arms produced different scan outputs; timing is not comparable")
        return 1
    return 0


def _readme(report: dict[str, Any]) -> str:
    """Render the bundle README.

    Args:
        report: Assembled benchmark report.

    Returns:
        str: Markdown body (the review marker is prepended by the writer).
    """
    speedup = report["speedup"]
    on_pool = report["results"]["on"]["pooled"]
    off_pool = report["results"]["off"]["pooled"]
    proto = report["protocol"]
    work = report["workload"]
    return f"""<!-- AI-GENERATED - NEEDS-REVIEW -->
# Raycast numba on/off micro-benchmark

Per-call wall time of `robot_sf.sensor.range_sensor.raycast` with numba JIT
enabled versus `NUMBA_DISABLE_JIT=1`, on a fixed synthetic LiDAR workload.

## Result

| Arm | median | IQR | min | max | n |
| --- | --- | --- | --- | --- | --- |
| numba ON | {on_pool["median_us"]:.2f} us | {on_pool["iqr_us"]:.2f} us | \
{on_pool["min_us"]:.2f} us | {on_pool["max_us"]:.2f} us | {on_pool["n"]} |
| numba OFF | {off_pool["median_us"] / 1000:.3f} ms | {off_pool["iqr_us"] / 1000:.3f} ms | \
{off_pool["min_us"] / 1000:.3f} ms | {off_pool["max_us"] / 1000:.3f} ms | {off_pool["n"]} |

Median speedup: **{speedup["median_ratio"]:.1f}x**
(per-repeat median ratios span {speedup["min_repeat_ratio"]:.1f}x to \
{speedup["max_repeat_ratio"]:.1f}x).

Both arms produced an identical scan output signature
(`outputs_identical_across_arms = {report["outputs_identical_across_arms"]}`), so the
comparison is between two implementations of the same computation.

## Protocol

* Workload: {work["num_rays"]} rays x {work["num_segments"]} obstacle segments x
  {work["num_peds"]} pedestrians, seed {work["seed"]}, {work["max_scan_dist_m"]} m max scan
  distance, {work["map_extent_m"]} m map extent.
* {proto["warmup_calls"]} discarded warmup calls per arm, so JIT compilation is excluded and
  the reported numbers are steady state.
* {proto["repeats"]} independent repeat blocks of {proto["calls_per_repeat"]} timed calls per
  arm; the full per-call distribution is retained, not a headline mean.
* Each arm runs in its own subprocess because numba reads the JIT toggle once at import.
* Timer: `{proto["timer"]}`.

## Identity

* Commit: `{report["commit"]}`
* Hardware: {report["hardware"]["cpu"]} ({report["hardware"]["machine_model"]}),
  {report["hardware"]["platform"]}
* Environment: Python {report["environment"]["python"]}, numba {report["environment"]["numba"]},
  numpy {report["environment"]["numpy"]}
* Started / finished (UTC): {report["started_utc"]} / {report["finished_utc"]}

## Boundary

Diagnostic hot-path timing on one machine and one synthetic workload. It is **not** a
whole-simulation speedup figure and **not** a cross-platform guarantee. Whole-simulation gain
is workload-dependent and much smaller on near-empty scenes.

## Reproduce

```bash
uv run python scripts/perf/raycast_numba_toggle_benchmark.py \\
    --output-dir docs/context/evidence/{Path(report["_bundle_name"]).name}
```

## Files

* `run_metadata.json` - commit, config, environment and hardware identity.
* `raycast_numba_toggle_results.json` - full distribution summary for both arms.
* `raycast_numba_toggle_samples.csv` - every timed call ({on_pool["n"] + off_pool["n"]} rows).
* `SHA256SUMS` - integrity manifest over the files above.
"""


def emit_bundle(out_dir: Path, report: dict[str, Any], arms: dict[str, Any]) -> None:
    """Write the checksummed evidence bundle for one benchmark run.

    Args:
        out_dir: Destination directory; created if absent.
        report: Assembled benchmark report.
        arms: Raw per-arm payloads carrying the per-call samples.
    """
    from robot_sf.evidence.writers import (
        write_csv,
        write_json,
        write_sha256sums,
        write_text,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {**report, "_bundle_name": out_dir.name}

    run_metadata = {
        "schema_version": "raycast-numba-toggle-run.v1",
        "benchmark": report["benchmark"],
        "git_head": report["commit"],
        "generated_at_utc": report["finished_utc"],
        "started_at_utc": report["started_utc"],
        "invoked_command": (
            "uv run python scripts/perf/raycast_numba_toggle_benchmark.py "
            f"--output-dir docs/context/evidence/{out_dir.name}"
        ),
        "harness_module": "scripts/perf/raycast_numba_toggle_benchmark.py",
        "measured_symbol": "robot_sf.sensor.range_sensor.raycast",
        "config": report["workload"],
        "protocol": report["protocol"],
        "arms": {
            "on": {"numba_disable_jit": arms["on"]["numba_disable_jit"]},
            "off": {"numba_disable_jit": arms["off"]["numba_disable_jit"]},
        },
        "execution_context": {
            "cpu_model": report["hardware"]["cpu"],
            "machine_model": report["hardware"]["machine_model"],
            "platform": report["hardware"]["platform"],
            "python_version": report["environment"]["python"],
        },
        "packages": {
            "numba": report["environment"]["numba"],
            "numpy": report["environment"]["numpy"],
        },
        "outputs_identical_across_arms": report["outputs_identical_across_arms"],
        "output_signature": report["output_signature"],
        "claim_boundary": ("diagnostic_hot_path_timing_single_machine_no_whole_simulation_claim"),
        "evidence_tier": "diagnostic",
    }

    write_json(out_dir / "run_metadata.json", run_metadata)
    write_json(
        out_dir / "raycast_numba_toggle_results.json",
        {k: v for k, v in report.items() if not k.startswith("_")},
    )
    rows = [
        {"arm": arm, "repeat": rep_idx, "call_index": call_idx, "duration_us": duration}
        for arm, payload in arms.items()
        for rep_idx, rep in enumerate(payload["samples_us_per_repeat"])
        for call_idx, duration in enumerate(rep)
    ]
    write_csv(out_dir / "raycast_numba_toggle_samples.csv", rows)
    write_text(out_dir / "README.md", _readme(report))
    write_sha256sums(out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
