"""Tests for the xdist worker-memory diagnostic (scripts/dev).

Pure analysis logic (linear fit, projection, ceiling classification, verdict,
Markdown render) is tested at CPU level with no subprocess. The process-tree
RSS sampler is exercised with a short-lived controlled subprocess that
allocates a known block of memory.
"""

from __future__ import annotations

import sys

import pytest

from scripts.dev.measure_xdist_worker_memory import (
    SCHEMA_VERSION,
    SweepPoint,
    build_report,
    build_verdict,
    classify_projection,
    fit_linear,
    project_peak_rss,
    render_markdown,
    sample_process_tree_peak,
)

# Representative sweep drawn from the real local measurement that confirmed
# the eviction mechanism (issue #4942): peak tree RSS grows ~linearly with the
# xdist worker count, ~1.45 GiB per worker. Using fixed numbers keeps the
# analysis tests deterministic and independent of host load.
_SAMPLE_SWEEP = (
    SweepPoint(n_workers=1, exit_code=0, wall_seconds=18.0, peak_tree_rss_gb=2.77, samples=168),
    SweepPoint(n_workers=2, exit_code=0, wall_seconds=18.0, peak_tree_rss_gb=4.24, samples=170),
    SweepPoint(n_workers=4, exit_code=0, wall_seconds=19.0, peak_tree_rss_gb=7.14, samples=173),
    SweepPoint(n_workers=8, exit_code=0, wall_seconds=22.0, peak_tree_rss_gb=12.92, samples=189),
)


# --- fit_linear --------------------------------------------------------------


def test_fit_linear_recovers_per_worker_slope() -> None:
    """The fit slope should approximate the known per-worker marginal cost."""
    fit = fit_linear(_SAMPLE_SWEEP)
    assert fit.slope_gb_per_worker == pytest.approx(1.45, abs=0.01)
    assert fit.intercept_gb == pytest.approx(1.34, abs=0.01)
    assert fit.r_squared >= 0.999


def test_fit_linear_perfect_line_has_unit_r_squared() -> None:
    """A perfectly linear sweep must yield R^2 == 1."""
    perfect = [
        SweepPoint(
            n_workers=w, exit_code=0, wall_seconds=1.0, peak_tree_rss_gb=1.0 + 2.0 * w, samples=1
        )
        for w in (1, 2, 3, 4)
    ]
    fit = fit_linear(perfect)
    assert fit.slope_gb_per_worker == pytest.approx(2.0)
    assert fit.intercept_gb == pytest.approx(1.0)
    assert fit.r_squared == pytest.approx(1.0)


def test_fit_linear_requires_two_points() -> None:
    """A single point cannot define a line."""
    with pytest.raises(ValueError, match="at least two"):
        fit_linear(_SAMPLE_SWEEP[:1])


def test_fit_linear_requires_distinct_worker_counts() -> None:
    """Repeated worker counts (zero variance in x) must fail loudly."""
    dup = (
        SweepPoint(n_workers=4, exit_code=0, wall_seconds=1.0, peak_tree_rss_gb=7.0, samples=1),
        SweepPoint(n_workers=4, exit_code=0, wall_seconds=1.0, peak_tree_rss_gb=8.0, samples=1),
    )
    with pytest.raises(ValueError, match="distinct worker counts"):
        fit_linear(dup)


# --- projection --------------------------------------------------------------


def test_project_peak_rss_matches_linear_model() -> None:
    """Projection at a measured worker count should match the model output."""
    fit = fit_linear(_SAMPLE_SWEEP)
    assert project_peak_rss(fit, 8) == pytest.approx(12.92, abs=0.05)
    # 32 workers extrapolates well above the measured range.
    assert project_peak_rss(fit, 32) == pytest.approx(47.7, abs=0.5)


def test_project_peak_rss_rejects_negative_workers() -> None:
    """Negative worker counts are nonsensical."""
    fit = fit_linear(_SAMPLE_SWEEP)
    with pytest.raises(ValueError, match="non-negative"):
        project_peak_rss(fit, -1)


# --- ceiling classification --------------------------------------------------


def test_classify_projection_flags_ceiling_exceedance() -> None:
    """32 workers must exceed the 16 GiB CI runner ceiling."""
    fit = fit_linear(_SAMPLE_SWEEP)
    proj = classify_projection(fit, 32, ceiling_gb=16.0)
    assert proj.exceeds_ceiling is True
    assert proj.headroom_gb < 0
    assert proj.projected_peak_rss_gb > 16.0


def test_classify_projection_keeps_auto_under_ceiling() -> None:
    """auto=4 workers (the PR #4948 mitigation) must stay under the ceiling."""
    fit = fit_linear(_SAMPLE_SWEEP)
    proj = classify_projection(fit, 4, ceiling_gb=16.0)
    assert proj.exceeds_ceiling is False
    assert proj.headroom_gb > 0


def test_classify_projection_rejects_nonpositive_ceiling() -> None:
    """A non-positive ceiling is invalid."""
    fit = fit_linear(_SAMPLE_SWEEP)
    with pytest.raises(ValueError, match="positive"):
        classify_projection(fit, 4, ceiling_gb=0.0)


# --- verdict -----------------------------------------------------------------


def test_build_verdict_confirms_eviction_mechanism() -> None:
    """With the measured slope, the verdict should confirm the eviction mechanism."""
    fit = fit_linear(_SAMPLE_SWEEP)
    verdict = build_verdict(fit, ceiling_gb=16.0, auto_workers=4)
    assert "linear" in verdict.scaling
    assert verdict.per_worker_marginal_gb == pytest.approx(1.45, abs=0.01)
    assert "Confirmed" in verdict.conclusion
    assert "32" in verdict.conclusion
    assert "SIGTERM 143" in verdict.conclusion


def test_build_verdict_not_confirmed_when_ceiling_never_breached() -> None:
    """A very high ceiling means 32 workers do not breach it -> not confirmed."""
    fit = fit_linear(_SAMPLE_SWEEP)
    verdict = build_verdict(fit, ceiling_gb=1000.0, auto_workers=4)
    assert "Not confirmed" in verdict.conclusion


def test_build_verdict_partially_confirmed_when_auto_also_breaches() -> None:
    """If even auto breaches the ceiling, the verdict flags partial confirmation."""
    fit = fit_linear(_SAMPLE_SWEEP)
    verdict = build_verdict(fit, ceiling_gb=5.0, auto_workers=4)
    assert "Partially confirmed" in verdict.conclusion


def test_build_verdict_scaling_label_drops_with_poor_fit() -> None:
    """A poor linear fit should not claim linear scaling."""
    noisy = (
        SweepPoint(n_workers=1, exit_code=0, wall_seconds=1.0, peak_tree_rss_gb=10.0, samples=1),
        SweepPoint(n_workers=2, exit_code=0, wall_seconds=1.0, peak_tree_rss_gb=2.0, samples=1),
        SweepPoint(n_workers=4, exit_code=0, wall_seconds=1.0, peak_tree_rss_gb=9.0, samples=1),
        SweepPoint(n_workers=8, exit_code=0, wall_seconds=1.0, peak_tree_rss_gb=3.0, samples=1),
    )
    fit = fit_linear(noisy)
    verdict = build_verdict(fit, ceiling_gb=16.0, auto_workers=4)
    assert verdict.scaling == "non-linear / inconclusive fit"


# --- report assembly + markdown ---------------------------------------------


def test_cli_help_exits_zero_with_usage() -> None:
    """``--help`` must be a cheap exit-0 path printing usage (CI contract style)."""
    import subprocess
    import sys
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "dev" / "measure_xdist_worker_memory.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    # --help must never start a measurement sweep.
    assert "uv run pytest" not in result.stdout


def test_build_report_has_schema_and_projections() -> None:
    """The full report must carry schema version, fit, projections, and verdict."""
    report = build_report(
        sweep=_SAMPLE_SWEEP,
        ceiling_gb=16.0,
        projection_points=(4, 8, 16, 32),
        auto_workers=4,
        host={"cpu_count_logical": 32},
        config={"ceiling_gb": 16.0, "probe_targets": ["tests/unit/test_version_utils.py"]},
    )
    payload = report.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert "generated_at" in payload
    assert payload["fit"] is not None
    assert len(payload["projections"]) == 4
    assert payload["verdict"] is not None
    proj32 = payload["projections"][-1]
    assert proj32["n_workers"] == 32
    assert proj32["exceeds_ceiling"] is True


def test_build_report_without_fit_omits_verdict() -> None:
    """A single-point sweep cannot be fit, so no fit/verdict is produced."""
    report = build_report(
        sweep=_SAMPLE_SWEEP[:1],
        ceiling_gb=16.0,
        projection_points=(32,),
        auto_workers=4,
        host={},
        config={},
    )
    payload = report.to_dict()
    assert payload["fit"] is None
    assert payload["verdict"] is None
    assert payload["projections"] == []


def test_render_markdown_contains_key_sections() -> None:
    """Markdown output must be human-readable and cite the mechanism."""
    report = build_report(
        sweep=_SAMPLE_SWEEP,
        ceiling_gb=16.0,
        projection_points=(4, 32),
        auto_workers=4,
        host={
            "cpu_count_physical": 16,
            "cpu_count_logical": 32,
            "total_memory_gb": 62.7,
            "available_memory_gb": 55.0,
        },
        config={
            "ceiling_gb": 16.0,
            "probe_targets": ["tests/unit/test_version_utils.py"],
            "dist_mode": "worksteal",
        },
    )
    md = render_markdown(report)
    assert "# xdist worker memory diagnostic" in md
    assert "Measured peak tree RSS by worker count" in md
    assert "Linear fit" in md
    assert "Projection vs ceiling" in md
    assert "GiB/worker" in md
    assert "Confirmed" in md
    assert "SIGTERM 143" in md


# --- process-tree sampling (controlled subprocess) --------------------------


def test_sample_process_tree_peak_captures_allocation(tmp_path) -> None:
    """The sampler must observe peak RSS of a child that allocates a block.

    Uses a short-lived Python child that allocates ~200 MiB and holds it for a
    moment so the sampler (interval 5 ms) reliably sees it before exit. The
    assertion checks the sampler captured a non-trivial, bounded peak rather
    than an exact size, since RSS accounting is OS-dependent.
    """
    import subprocess

    child = tmp_path / "alloc.py"
    child.write_text(
        "import time, sys\n"
        "# Allocate ~200 MiB and touch every page so RSS reflects it.\n"
        "block = bytearray(200 * 1024 * 1024)\n"
        "for i in range(0, len(block), 4096):\n"
        "    block[i] = 1\n"
        "# Hold the allocation long enough for several samples.\n"
        "time.sleep(0.5)\n",
        encoding="utf-8",
    )
    popen = subprocess.Popen([sys.executable, str(child)])
    try:
        peak_gb, samples = sample_process_tree_peak(popen, interval_seconds=0.005)
    finally:
        popen.wait(timeout=30)
    # 200 MiB allocation plus interpreter overhead: well above the bare
    # interpreter footprint (~tens of MiB) and below a generous bound.
    assert peak_gb >= 0.1, f"peak {peak_gb:.3f} GiB too small; sampler missed allocation"
    assert peak_gb < 1.0
    assert samples >= 1


def test_sample_process_tree_peak_already_exited(tmp_path) -> None:
    """A child that exits before sampling should return a zero-ish peak cleanly."""
    import subprocess

    child = tmp_path / "quick.py"
    child.write_text("pass\n", encoding="utf-8")
    popen = subprocess.Popen([sys.executable, str(child)])
    popen.wait(timeout=30)
    # Sampling an already-exited process must not raise.
    peak_gb, samples = sample_process_tree_peak(popen, interval_seconds=0.01)
    assert peak_gb >= 0.0
    assert samples == 0
