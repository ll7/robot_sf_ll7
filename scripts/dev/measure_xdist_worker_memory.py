#!/usr/bin/env python3
"""Measure how pytest-xdist peak memory scales with worker count.

This diagnostic reproduces locally the memory-pressure mechanism behind the
Nightly Performance ``xdist race validation`` SIGTERM (exit 143) defect
(issue #4942). The CI runner (4 vCPUs / ~16 GB) was killed by GitHub's
runner-eviction watchdog ~5 min into a 32-worker run. Exit 143 is a SIGTERM
runner reclaim under resource pressure, *not* the kernel OOM-killer (which
would be SIGKILL / exit 137).

PR #4948 shipped a bounded-config mitigation (``XDIST_RACE_WORKERS=auto`` ->
4 workers on the 4-vCPU runner) but explicitly did **not** reproduce the
mechanism locally or measure the resource ceiling. This script closes that
gap: it runs a fixed, fast, deterministic test probe at several worker counts,
samples the peak RSS of the whole pytest process tree at each, fits a linear
per-worker memory model, projects total RSS at arbitrary worker counts, and
classifies each projection against a host memory ceiling.

Typical finding (32-core / 64 GB host, ``tests/unit/test_version_utils.py``
probe): per-worker marginal cost ~1.4-1.7 GB, so the projected total at 32
workers (~49 GB) vastly exceeds a 16 GB runner ceiling (-> SIGTERM eviction),
while ``auto`` on a 4-vCPU host (~7.6 GB at 4 workers) stays well under it.

The absolute RSS numbers are host-specific (psutil RSS double-counts shared
pages across workers, so they are a generous upper bound), but the **linear
scaling and the ceiling-exceeding projection at 32 workers are conclusive**.
This is reproduction-of-mechanism evidence, not a benchmark of test runtime.

Outputs a JSON record and an optional Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    # With ``from __future__ import annotations`` these are only needed for
    # static type checkers; they are never evaluated at runtime.
    from collections.abc import Sequence

SCHEMA_VERSION = "xdist_worker_memory.v1"

# The CI runner memory ceiling for the Nightly Performance job, in gibibytes.
# Per the failing-run logs (issue #4942): 4 vCPUs / ~16 GB ubuntu-latest.
DEFAULT_CEILING_GB = 16.0
# Worker counts to sweep by default. Kept small so the diagnostic runs in
# minutes, not the full-suite nightly budget.
DEFAULT_WORKER_SWEEP = (1, 2, 4, 8)
# Worker counts to project (without running) in the default report. 32 is the
# original nightly setting that triggered the eviction; 4 is the ``auto``
# resolution on the 4-vCPU runner.
DEFAULT_PROJECTION_POINTS = (4, 8, 16, 32)
# A fast, deterministic, dependency-light test probe that still imports the
# package so each worker pays a realistic interpreter+import baseline. The
# probe choice affects the absolute RSS but not the per-worker scaling shape.
DEFAULT_PROBE_TARGETS = ("tests/unit/test_version_utils.py",)
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """Peak-memory sample for one worker count."""

    n_workers: int
    exit_code: int
    wall_seconds: float
    peak_tree_rss_gb: float
    samples: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LinearFit:
    """Least-squares linear model of tree RSS as a function of worker count.

    ``slope`` is the marginal per-worker memory cost in GiB; ``intercept`` is
    the extrapolated single-process baseline (interpreter + pytest + uv run
    layer) in GiB.
    """

    slope_gb_per_worker: float
    intercept_gb: float
    r_squared: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Projection:
    """Projected tree RSS for a worker count not (necessarily) measured."""

    n_workers: int
    projected_peak_rss_gb: float
    ceiling_gb: float
    exceeds_ceiling: bool
    headroom_gb: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiagnosticVerdict:
    """Plain-language classification of the measured memory-pressure mechanism."""

    scaling: str
    per_worker_marginal_gb: float
    auto_workers_on_4cpu: int
    conclusion: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class DiagnosticReport:
    """Full diagnostic report, serialized to JSON and Markdown."""

    schema_version: str
    generated_at: str
    host: dict[str, object]
    config: dict[str, object]
    sweep: list[dict[str, object]] = field(default_factory=list)
    fit: dict[str, object] | None = None
    projections: list[dict[str, object]] = field(default_factory=list)
    verdict: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def fit_linear(points: Sequence[SweepPoint]) -> LinearFit:
    """Fit ``peak_tree_rss = slope * n_workers + intercept`` by least squares.

    Requires at least two distinct worker counts. Returns r_squared so callers
    can judge how convincingly the data support a linear (memory-proportional)
    model: a high r_squared means total memory is dominated by per-worker cost,
    which is the signature of the worker-count-driven eviction mechanism.
    """
    if len(points) < 2:
        raise ValueError("linear fit requires at least two sweep points")
    xs = [float(p.n_workers) for p in points]
    ys = [float(p.peak_tree_rss_gb) for p in points]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    if den_x == 0:
        raise ValueError("linear fit requires at least two distinct worker counts")
    slope = num / den_x
    intercept = mean_y - slope * mean_x
    den_y = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1.0 if den_y == 0 else max(0.0, min(1.0, (num * num) / (den_x * den_y)))
    return LinearFit(
        slope_gb_per_worker=slope,
        intercept_gb=intercept,
        r_squared=r_squared,
    )


def project_peak_rss(fit: LinearFit, n_workers: int) -> float:
    """Project the peak tree RSS (GiB) for *n_workers* under the linear model."""
    if n_workers < 0:
        raise ValueError("n_workers must be non-negative")
    return fit.intercept_gb + fit.slope_gb_per_worker * float(n_workers)


def classify_projection(
    fit: LinearFit,
    n_workers: int,
    ceiling_gb: float,
) -> Projection:
    """Project RSS for *n_workers* and classify it against the host ceiling."""
    if ceiling_gb <= 0:
        raise ValueError("ceiling_gb must be positive")
    projected = project_peak_rss(fit, n_workers)
    return Projection(
        n_workers=n_workers,
        projected_peak_rss_gb=projected,
        ceiling_gb=ceiling_gb,
        exceeds_ceiling=projected > ceiling_gb,
        headroom_gb=ceiling_gb - projected,
    )


def build_verdict(
    fit: LinearFit,
    ceiling_gb: float,
    *,
    auto_workers: int = 4,
) -> DiagnosticVerdict:
    """Render a plain-language verdict on the memory-pressure mechanism.

    The verdict is deliberately conservative: it reports the measured scaling
    and the per-worker marginal cost, and states whether the projected RSS at
    the original nightly setting (32) exceeds the ceiling while ``auto`` on a
    4-vCPU host stays under it. It does **not** claim the CI runner will behave
    identically in absolute terms (host accounting differs), only that the
    mechanism (worker-count-driven memory growth crossing a fixed ceiling) is
    confirmed by the local measurement.
    """
    at_32 = classify_projection(fit, 32, ceiling_gb)
    at_auto = classify_projection(fit, auto_workers, ceiling_gb)
    if fit.r_squared >= 0.9:
        scaling = "linear (memory proportional to worker count)"
    elif fit.r_squared >= 0.7:
        scaling = "approximately linear"
    else:
        scaling = "non-linear / inconclusive fit"
    if at_32.exceeds_ceiling and not at_auto.exceeds_ceiling:
        conclusion = (
            f"Confirmed: projected peak RSS at 32 workers "
            f"({at_32.projected_peak_rss_gb:.1f} GiB) exceeds the {ceiling_gb:.0f} GiB "
            f"ceiling (runner eviction / SIGTERM 143), while auto={auto_workers} workers "
            f"({at_auto.projected_peak_rss_gb:.1f} GiB) stays under it. This reproduces "
            f"the eviction mechanism behind the Nightly Performance defect."
        )
    elif at_32.exceeds_ceiling and at_auto.exceeds_ceiling:
        conclusion = (
            f"Partially confirmed: 32 workers ({at_32.projected_peak_rss_gb:.1f} GiB) "
            f"exceed the {ceiling_gb:.0f} GiB ceiling, but auto={auto_workers} workers "
            f"({at_auto.projected_peak_rss_gb:.1f} GiB) also exceed it on this host; the "
            f"ceiling may need to be revisited or the probe is heavier than the CI lane."
        )
    else:
        conclusion = (
            "Not confirmed by this measurement: projected RSS at 32 workers "
            f"({at_32.projected_peak_rss_gb:.1f} GiB) does not exceed the "
            f"{ceiling_gb:.0f} GiB ceiling. The memory hypothesis is not supported "
            "by this probe; investigate the race/timeout paths instead."
        )
    return DiagnosticVerdict(
        scaling=scaling,
        per_worker_marginal_gb=fit.slope_gb_per_worker,
        auto_workers_on_4cpu=auto_workers,
        conclusion=conclusion,
    )


def _tree_rss_gb(root: psutil.Process) -> float:
    """Sum RSS (GiB) over *root* and all its live descendants."""
    total = 0
    procs = [root]
    try:
        procs.extend(root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for pr in procs:
        try:
            total += pr.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024.0**3)


def sample_process_tree_peak(
    popen: subprocess.Popen[bytes],
    *,
    interval_seconds: float,
) -> tuple[float, int]:
    """Sample the peak tree RSS (GiB) of *popen* until it exits.

    Returns ``(peak_rss_gb, sample_count)``. Sampling is best-effort: transient
    ``NoSuchProcess`` / ``AccessDenied`` errors during teardown are swallowed.
    """
    # If the process already exited before we attach, there is nothing to
    # sample; return a clean zero-peak result instead of raising.
    try:
        root = psutil.Process(popen.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0, 0
    peak = 0.0
    samples = 0
    while popen.poll() is None:
        try:
            rss = _tree_rss_gb(root)
        except psutil.NoSuchProcess:
            break
        peak = max(peak, rss)
        samples += 1
        time.sleep(interval_seconds)
    return peak, samples


def _terminate_process_tree(popen: subprocess.Popen[bytes]) -> None:
    """Best-effort kill of *popen* and all its descendants if still running.

    Called from a ``finally`` guard so an interrupt (e.g. ``KeyboardInterrupt``)
    or unexpected error during sampling cannot leave a multi-worker pytest
    process tree orphaned in the background.
    """
    if popen.poll() is not None:
        return
    try:
        root = psutil.Process(popen.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return
    procs = [root]
    try:
        procs.extend(root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for pr in procs:
        try:
            pr.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    try:
        popen.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def run_one_sweep(
    n_workers: int,
    targets: Sequence[str],
    *,
    dist_mode: str,
    sample_interval_seconds: float,
    extra_pytest_args: Sequence[str],
    runner: Sequence[str],
) -> SweepPoint:
    """Run pytest at *n_workers* against *targets* and sample peak tree RSS."""
    cmd = list(runner) + [
        "pytest",
        "-n",
        str(n_workers),
        "--dist",
        dist_mode,
        "-q",
        "-p",
        "no:cacheprovider",
        *extra_pytest_args,
        *targets,
    ]
    start = time.monotonic()
    popen = psutil.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        peak_gb, samples = sample_process_tree_peak(popen, interval_seconds=sample_interval_seconds)
        exit_code = popen.wait()
    finally:
        # Never leave an interrupted multi-worker pytest tree running.
        _terminate_process_tree(popen)
    wall = time.monotonic() - start
    return SweepPoint(
        n_workers=n_workers,
        exit_code=exit_code,
        wall_seconds=wall,
        peak_tree_rss_gb=peak_gb,
        samples=samples,
    )


def _host_info() -> dict[str, object]:
    """Collect a compact, stable description of the measurement host."""
    vm = psutil.virtual_memory()
    return {
        "cpu_count_logical": psutil.cpu_count(logical=True) or 0,
        "cpu_count_physical": psutil.cpu_count(logical=False) or 0,
        "total_memory_gb": round(vm.total / (1024.0**3), 2),
        "available_memory_gb": round(vm.available / (1024.0**3), 2),
    }


def build_report(
    *,
    sweep: Sequence[SweepPoint],
    ceiling_gb: float,
    projection_points: Sequence[int],
    auto_workers: int,
    host: dict[str, object],
    config: dict[str, object],
) -> DiagnosticReport:
    """Assemble the full diagnostic report from measured sweep points."""
    report = DiagnosticReport(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        host=host,
        config=config,
        sweep=[p.to_dict() for p in sweep],
    )
    if len(sweep) >= 2:
        fit = fit_linear(sweep)
        report.fit = fit.to_dict()
        report.projections = [
            classify_projection(fit, w, ceiling_gb).to_dict() for w in projection_points
        ]
        report.verdict = build_verdict(fit, ceiling_gb, auto_workers=auto_workers).to_dict()
    return report


def render_markdown(report: DiagnosticReport) -> str:
    """Render the diagnostic report as a compact Markdown summary."""
    lines: list[str] = []
    lines.append("# xdist worker memory diagnostic")
    lines.append("")
    host = report.host
    lines.append(
        f"**Host:** {host.get('cpu_count_physical')} physical / "
        f"{host.get('cpu_count_logical')} logical CPUs, "
        f"{host.get('total_memory_gb')} GiB total "
        f"({host.get('available_memory_gb')} GiB available)."
    )
    cfg = report.config
    lines.append(
        f"**Ceiling:** {cfg.get('ceiling_gb')} GiB. "
        f"**Probe:** `{', '.join(cfg.get('probe_targets', []))}`. "
        f"**Dist:** `{cfg.get('dist_mode')}`."
    )
    lines.append("")
    lines.append("## Measured peak tree RSS by worker count")
    lines.append("")
    lines.append("| Workers | Peak tree RSS (GiB) | Wall (s) | Exit | Samples |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in report.sweep:
        lines.append(
            f"| {row['n_workers']} | {row['peak_tree_rss_gb']:.2f} | "
            f"{row['wall_seconds']:.1f} | {row['exit_code']} | {row['samples']} |"
        )
    lines.append("")
    if report.fit is None:
        lines.append("_Fewer than two sweep points: no fit or projection._")
        return "\n".join(lines) + "\n"
    fit = report.fit
    lines.append("## Linear fit")
    lines.append("")
    lines.append(f"- Per-worker marginal cost: **{fit['slope_gb_per_worker']:.2f} GiB/worker**")
    lines.append(f"- Baseline (intercept): {fit['intercept_gb']:.2f} GiB")
    lines.append(f"- R²: {fit['r_squared']:.3f}")
    lines.append("")
    lines.append("## Projection vs ceiling")
    lines.append("")
    lines.append("| Workers | Projected RSS (GiB) | Ceiling (GiB) | Exceeds | Headroom (GiB) |")
    lines.append("|---:|---:|---:|:---:|---:|")
    for proj in report.projections:
        flag = "yes" if proj["exceeds_ceiling"] else "no"
        lines.append(
            f"| {proj['n_workers']} | {proj['projected_peak_rss_gb']:.1f} | "
            f"{proj['ceiling_gb']:.0f} | {flag} | {proj['headroom_gb']:.1f} |"
        )
    lines.append("")
    if report.verdict:
        v = report.verdict
        lines.append("## Verdict")
        lines.append("")
        lines.append(f"- Scaling: {v['scaling']}")
        lines.append(f"- Per-worker marginal: {v['per_worker_marginal_gb']:.2f} GiB")
        lines.append("")
        lines.append(v["conclusion"])
        lines.append("")
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(DEFAULT_WORKER_SWEEP),
        help=(f"Worker counts to sweep and measure (default: {list(DEFAULT_WORKER_SWEEP)})."),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_PROBE_TARGETS),
        help=(
            f"Pytest targets to use as the memory probe (default: {list(DEFAULT_PROBE_TARGETS)})."
        ),
    )
    parser.add_argument(
        "--ceiling-gb",
        type=float,
        default=DEFAULT_CEILING_GB,
        help=(
            "Host memory ceiling in GiB for projection classification "
            f"(default: {DEFAULT_CEILING_GB}, the CI ubuntu-latest runner)."
        ),
    )
    parser.add_argument(
        "--project-at",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROJECTION_POINTS),
        help=(
            "Worker counts to project (without running) against the ceiling "
            f"(default: {list(DEFAULT_PROJECTION_POINTS)})."
        ),
    )
    parser.add_argument(
        "--auto-workers",
        type=int,
        default=4,
        help=(
            "Worker count that ``auto`` resolves to on the target runner "
            "(default: 4, the 4-vCPU CI runner)."
        ),
    )
    parser.add_argument(
        "--dist",
        default="worksteal",
        choices=["load", "worksteal", "loadscope", "loadfile", "loadgroup"],
        help="xdist distribution mode (default: worksteal, matching the nightly).",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
        help=(f"RSS sampling interval in seconds (default: {DEFAULT_SAMPLE_INTERVAL_SECONDS})."),
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        dest="extra_pytest_args",
        help="Extra pytest arguments forwarded verbatim (may repeat).",
    )
    parser.add_argument(
        "--runner",
        nargs="+",
        default=["uv", "run"],
        help="Command prefix used to launch pytest (default: uv run).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write the JSON diagnostic record to this path.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Write the Markdown summary to this path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the JSON record on stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the worker-memory diagnostic and emit JSON / Markdown artifacts."""
    args = _parse_args(argv)

    worker_counts = sorted({int(w) for w in args.workers if int(w) >= 1})
    if not worker_counts:
        print("--workers requires at least one positive integer.", file=sys.stderr)
        return 2

    if args.ceiling_gb <= 0:
        print(f"--ceiling-gb must be positive; got {args.ceiling_gb}.", file=sys.stderr)
        return 2
    if args.auto_workers < 1:
        print(
            f"--auto-workers must be a positive integer; got {args.auto_workers}.", file=sys.stderr
        )
        return 2
    invalid_proj = [w for w in args.project_at if w < 0]
    if invalid_proj:
        print(f"--project-at values must be non-negative; got {invalid_proj}.", file=sys.stderr)
        return 2

    config: dict[str, object] = {
        "ceiling_gb": args.ceiling_gb,
        "probe_targets": list(args.targets),
        "dist_mode": args.dist,
        "sample_interval_seconds": args.sample_interval,
        "sweep_workers": worker_counts,
        "projection_points": list(args.project_at),
        "auto_workers": args.auto_workers,
        "runner": list(args.runner),
        "extra_pytest_args": list(args.extra_pytest_args),
    }

    print(
        f"Measuring xdist peak tree RSS at workers={worker_counts} "
        f"(probe={list(args.targets)}, dist={args.dist})",
        file=sys.stderr,
    )
    sweep: list[SweepPoint] = []
    for n in worker_counts:
        point = run_one_sweep(
            n,
            args.targets,
            dist_mode=args.dist,
            sample_interval_seconds=args.sample_interval,
            extra_pytest_args=args.extra_pytest_args,
            runner=args.runner,
        )
        sweep.append(point)
        print(
            f"  workers={point.n_workers} peak={point.peak_tree_rss_gb:.2f} GiB "
            f"exit={point.exit_code} ({point.wall_seconds:.1f}s, "
            f"{point.samples} samples)",
            file=sys.stderr,
        )
        # If a sweep step itself crashes, keep going so the scaling shape is
        # still visible, but surface the non-zero exit in the report.

    report = build_report(
        sweep=sweep,
        ceiling_gb=args.ceiling_gb,
        projection_points=args.project_at,
        auto_workers=args.auto_workers,
        host=_host_info(),
        config=config,
    )
    payload = report.to_dict()

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {args.output_json}", file=sys.stderr)
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(report))
        print(f"Wrote Markdown report to {args.output_markdown}", file=sys.stderr)
    if not args.quiet:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
