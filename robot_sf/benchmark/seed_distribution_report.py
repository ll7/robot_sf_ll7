"""Unified seed-distribution report for statistical robustness.

This module defines the ``seed_distribution_report.v1`` schema and provides a
builder that normalizes seed-level benchmark artifacts from multiple campaign
reporting surfaces into a single auditable distributional report.

The report preserves seed counts, raw outcome counts, point estimates,
confidence intervals, and instability/insufficiency diagnostics. It does NOT
run new simulations; it consumes existing artifacts only.

Claim boundary:

  Existing multi-seed benchmark outputs can be normalized into a common,
  auditable statistical-robustness report.

This does NOT support claims that:

- the benchmark scenario catalog is complete;
- the simulator is calibrated to real-world behavior;
- narrower confidence intervals imply external validity;
- a single campaign is publication-ready without its original evidence review.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION = "seed_distribution_report.v1"
_REPORT_GENERATOR = "robot_sf.benchmark.seed_distribution_report"

#: Minimum seed count below which the ``insufficient_seed_count`` diagnostic is set.
DEFAULT_INSUFFICIENT_SEED_THRESHOLD = 3
#: CI half-width above which the ``wide_interval`` diagnostic is set.
DEFAULT_WIDE_INTERVAL_THRESHOLD = 0.15
#: Absolute rank-flip-rate above which the ``unstable_rank`` diagnostic is set.
DEFAULT_UNSTABLE_RANK_FLIP_THRESHOLD = 0.3


@dataclass(frozen=True)
class MetricSummary:
    """Distributional summary for a single metric on a single surface."""

    point_estimate: float
    std: float
    cv: float
    count: float
    ci_low: float
    ci_high: float
    ci_half_width: float
    method: str = "bootstrap_mean_over_seed_means"
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class RawCounts:
    """Optional raw numerator/denominator for discrete outcomes."""

    numerator: int | None = None
    denominator: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class SeedDistributionDiagnostics:
    """Machine-readable diagnostic flags for a surface record."""

    insufficient_seed_count: bool
    unstable_rank: bool | None = None
    wide_interval: bool | None = None
    advisory_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class IntervalEstimate:
    """Confidence interval metadata."""

    method: str
    lower: float
    upper: float
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class PerSeedValue:
    """Single per-seed metric observation."""

    seed: int
    value: float
    episode_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class SurfaceProvenance:
    """Provenance for a single surface record."""

    input_artifact: str
    input_schema_version: str | None = None
    adapter: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class SurfaceRecord:
    """One normalized surface (planner x scenario x metric) in the report."""

    surface_id: str
    metric: str
    seed_count: int
    episode_count: int
    point_estimate: float
    interval: IntervalEstimate
    metrics_summary: dict[str, MetricSummary]
    diagnostics: SeedDistributionDiagnostics
    provenance: SurfaceProvenance
    scenario_id: str | None = None
    planner_id: str | None = None
    unit: str | None = None
    raw_counts: RawCounts | None = None
    per_seed: list[PerSeedValue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        return {
            "surface_id": self.surface_id,
            "scenario_id": self.scenario_id,
            "planner_id": self.planner_id,
            "metric": self.metric,
            "unit": self.unit,
            "seed_count": self.seed_count,
            "episode_count": self.episode_count,
            "raw_counts": self.raw_counts.to_dict() if self.raw_counts else None,
            "point_estimate": self.point_estimate,
            "interval": self.interval.to_dict(),
            "metrics_summary": {k: v.to_dict() for k, v in self.metrics_summary.items()},
            "per_seed": [ps.to_dict() for ps in self.per_seed],
            "diagnostics": self.diagnostics.to_dict(),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class SourceProvenance:
    """Provenance for the overall report."""

    campaign_root: str
    report_paths: list[str]
    generated_by: str = _REPORT_GENERATOR
    commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class SeedDistributionReport:
    """Top-level ``seed_distribution_report.v1`` payload."""

    schema_version: str
    generated_at_utc: str
    source: SourceProvenance
    surfaces: list[SurfaceRecord]
    interpretation_boundary: str = (
        "Statistical repeatability improves internal evidence quality but is not "
        "a claim of scenario coverage, simulator fidelity, or real-world validity."
    )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "source": self.source.to_dict(),
            "surfaces": [s.to_dict() for s in self.surfaces],
            "interpretation_boundary": self.interpretation_boundary,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return JSON string representation."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


def validate_report_schema_version(payload: dict[str, Any]) -> None:
    """Reject payloads with unsupported schema versions.

    Raises:
        ValueError: when the schema_version is missing or unsupported.
    """
    version = payload.get("schema_version")
    if version != SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION:
        msg = f"Unsupported schema version: {version!r}; expected {SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION!r}"
        raise ValueError(msg)


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _classify_diagnostics(
    seed_count: int,
    ci_half_width: float | float,
    *,
    insufficient_seed_threshold: int = DEFAULT_INSUFFICIENT_SEED_THRESHOLD,
    wide_interval_threshold: float = DEFAULT_WIDE_INTERVAL_THRESHOLD,
    rank_flip_rate: float | None = None,
    unstable_rank_flip_threshold: float = DEFAULT_UNSTABLE_RANK_FLIP_THRESHOLD,
) -> SeedDistributionDiagnostics:
    insufficient = seed_count < insufficient_seed_threshold
    wide = bool(math.isfinite(ci_half_width) and ci_half_width > wide_interval_threshold)
    unstable: bool | None = None
    if rank_flip_rate is not None:
        unstable = rank_flip_rate > unstable_rank_flip_threshold
    return SeedDistributionDiagnostics(
        insufficient_seed_count=insufficient,
        unstable_rank=unstable,
        wide_interval=wide,
        advisory_only=insufficient,
    )


def _build_surface_from_seed_variability_row(
    row: dict[str, Any],
    *,
    primary_metric: str,
    insufficient_seed_threshold: int,
    wide_interval_threshold: float,
    input_artifact: str,
) -> SurfaceRecord:
    """Convert a single seed-variability row into a SurfaceRecord.

    Returns:
        Normalized surface record from the row.
    """
    scenario_id = str(row.get("scenario_id", "unknown"))
    planner_key = str(row.get("planner_key", row.get("algo", "unknown")))
    seed_count = int(row.get("seed_count", row.get("n", 0)))
    episode_count = int(row.get("episode_count", 0))
    summary = row.get("summary", {})
    primary_stats = summary.get(primary_metric, {})

    point_estimate = float(primary_stats.get("mean", float("nan")))
    ci_low = float(primary_stats.get("ci_low", float("nan")))
    ci_high = float(primary_stats.get("ci_high", float("nan")))
    ci_hw = float(primary_stats.get("ci_half_width", float("nan")))

    interval = IntervalEstimate(
        method=str(primary_stats.get("method", "bootstrap_mean_over_seed_means")),
        lower=ci_low,
        upper=ci_high,
        confidence_level=float(
            row.get("provenance", {}).get("confidence", {}).get("confidence", 0.95)
        ),
    )

    metrics_summary: dict[str, MetricSummary] = {}
    for metric_name, metric_stats in summary.items():
        metrics_summary[metric_name] = MetricSummary(
            point_estimate=float(metric_stats.get("mean", float("nan"))),
            std=float(metric_stats.get("std", float("nan"))),
            cv=float(metric_stats.get("cv", float("nan"))),
            count=float(metric_stats.get("count", 0.0)),
            ci_low=float(metric_stats.get("ci_low", float("nan"))),
            ci_high=float(metric_stats.get("ci_high", float("nan"))),
            ci_half_width=float(metric_stats.get("ci_half_width", float("nan"))),
        )

    per_seed: list[PerSeedValue] = []
    for entry in row.get("per_seed", []):
        seed = int(entry.get("seed", -1))
        seed_metrics = entry.get("metrics", {})
        val = float(seed_metrics.get(primary_metric, float("nan")))
        per_seed.append(
            PerSeedValue(
                seed=seed,
                value=val,
                episode_count=int(entry.get("episode_count", 1)),
            )
        )

    raw_counts: RawCounts | None = None
    if primary_metric in ("success", "collisions", "near_misses"):
        numerator = (
            round(point_estimate * episode_count) if math.isfinite(point_estimate) else None
        )
        raw_counts = RawCounts(numerator=numerator, denominator=episode_count)

    diagnostics = _classify_diagnostics(
        seed_count,
        ci_hw,
        insufficient_seed_threshold=insufficient_seed_threshold,
        wide_interval_threshold=wide_interval_threshold,
    )

    surface_id = f"{scenario_id}__{planner_key}__{primary_metric}"
    return SurfaceRecord(
        surface_id=surface_id,
        metric=primary_metric,
        seed_count=seed_count,
        episode_count=episode_count,
        point_estimate=point_estimate,
        interval=interval,
        metrics_summary=metrics_summary,
        diagnostics=diagnostics,
        provenance=SurfaceProvenance(
            input_artifact=input_artifact,
            input_schema_version=str(
                row.get("provenance", {}).get(
                    "schema_version", "benchmark-seed-variability-by-scenario.v1"
                )
            ),
            adapter="seed_variability",
        ),
        scenario_id=scenario_id,
        planner_id=planner_key,
        raw_counts=raw_counts,
        per_seed=per_seed,
    )


def _build_surface_from_rank_stability_cell(
    cell: dict[str, Any],
    rank_stability: list[dict[str, Any]] | None,
    *,
    primary_metric: str,
    insufficient_seed_threshold: int,
    wide_interval_threshold: float,
    input_artifact: str,
) -> SurfaceRecord:
    """Convert a rank-stability cell into a SurfaceRecord.

    Returns:
        Normalized surface record from the cell.
    """
    scenario_id = str(cell.get("scenario_id", "unknown"))
    planner_key = str(cell.get("planner_key", "unknown"))
    seed_count = int(cell.get("seed_count", 0))
    cell_metrics = cell.get("metrics", {})

    metric_stats = cell_metrics.get(primary_metric, {})
    point_estimate = float(metric_stats.get("mean", float("nan")))
    ci_low = float(metric_stats.get("ci_low", float("nan")))
    ci_high = float(metric_stats.get("ci_high", float("nan")))
    ci_hw = float(metric_stats.get("ci_half_width", float("nan")))

    interval = IntervalEstimate(
        method="bootstrap_mean_over_seed_means",
        lower=ci_low,
        upper=ci_high,
        confidence_level=0.95,
    )

    metrics_summary: dict[str, MetricSummary] = {}
    for mname, mstats in cell_metrics.items():
        metrics_summary[mname] = MetricSummary(
            point_estimate=float(mstats.get("mean", float("nan"))),
            std=float(mstats.get("std", float("nan"))),
            cv=float(mstats.get("cv", float("nan"))),
            count=float(mstats.get("count", 0.0)),
            ci_low=float(mstats.get("ci_low", float("nan"))),
            ci_high=float(mstats.get("ci_high", float("nan"))),
            ci_half_width=float(mstats.get("ci_half_width", float("nan"))),
        )

    rank_flip_rate: float | None = None
    if rank_stability:
        for rs in rank_stability:
            if rs.get("scenario_id") == scenario_id:
                rank_flip_rate = rs.get("rank_flip_rate")
                break

    diagnostics = _classify_diagnostics(
        seed_count,
        ci_hw,
        insufficient_seed_threshold=insufficient_seed_threshold,
        wide_interval_threshold=wide_interval_threshold,
        rank_flip_rate=rank_flip_rate,
    )

    surface_id = f"{scenario_id}__{planner_key}__{primary_metric}__rank_stability"
    return SurfaceRecord(
        surface_id=surface_id,
        metric=primary_metric,
        seed_count=seed_count,
        episode_count=0,
        point_estimate=point_estimate,
        interval=interval,
        metrics_summary=metrics_summary,
        diagnostics=diagnostics,
        provenance=SurfaceProvenance(
            input_artifact=input_artifact,
            input_schema_version="issue_3216_headline_ci_rank_stability.v1",
            adapter="rank_stability",
        ),
        scenario_id=scenario_id,
        planner_id=planner_key,
    )


def adapt_seed_variability_report(
    payload: dict[str, Any],
    *,
    primary_metric: str = "success",
    insufficient_seed_threshold: int = DEFAULT_INSUFFICIENT_SEED_THRESHOLD,
    wide_interval_threshold: float = DEFAULT_WIDE_INTERVAL_THRESHOLD,
    input_artifact: str = "seed_variability_by_scenario.json",
) -> list[SurfaceRecord]:
    """Adapt a ``benchmark-seed-variability-by-scenario.v1`` payload.

    Returns:
        Normalized surface records from the seed-variability payload.
    """
    rows = payload.get("rows", [])
    surfaces: list[SurfaceRecord] = []
    for row in rows:
        surfaces.append(
            _build_surface_from_seed_variability_row(
                row,
                primary_metric=primary_metric,
                insufficient_seed_threshold=insufficient_seed_threshold,
                wide_interval_threshold=wide_interval_threshold,
                input_artifact=input_artifact,
            )
        )
    return surfaces


def adapt_rank_stability_report(
    payload: dict[str, Any],
    *,
    primary_metric: str = "success",
    insufficient_seed_threshold: int = DEFAULT_INSUFFICIENT_SEED_THRESHOLD,
    wide_interval_threshold: float = DEFAULT_WIDE_INTERVAL_THRESHOLD,
    input_artifact: str = "headline_ci_rank_stability.json",
) -> list[SurfaceRecord]:
    """Adapt a ``issue_3216_headline_ci_rank_stability.v1`` payload.

    Returns:
        Normalized surface records from the rank-stability payload.
    """
    cells = payload.get("cells", [])
    rank_stability = payload.get("rank_stability")
    surfaces: list[SurfaceRecord] = []
    for cell in cells:
        if not cell.get("counted", True):
            continue
        surfaces.append(
            _build_surface_from_rank_stability_cell(
                cell,
                rank_stability,
                primary_metric=primary_metric,
                insufficient_seed_threshold=insufficient_seed_threshold,
                wide_interval_threshold=wide_interval_threshold,
                input_artifact=input_artifact,
            )
        )
    return surfaces


def build_seed_distribution_report(
    surfaces: list[SurfaceRecord],
    *,
    campaign_root: str = "",
    report_paths: list[str] | None = None,
    commit: str | None = None,
) -> SeedDistributionReport:
    """Build a ``seed_distribution_report.v1`` from pre-adapted surfaces.

    Returns:
        A fully-formed seed distribution report.
    """
    return SeedDistributionReport(
        schema_version=SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION,
        generated_at_utc=datetime.now(tz=UTC).isoformat(),
        source=SourceProvenance(
            campaign_root=campaign_root,
            report_paths=report_paths or [],
            commit=commit or _git_head(),
        ),
        surfaces=surfaces,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report_from_campaign_dir(
    campaign_root: Path,
    *,
    primary_metric: str = "success",
    insufficient_seed_threshold: int = DEFAULT_INSUFFICIENT_SEED_THRESHOLD,
    wide_interval_threshold: float = DEFAULT_WIDE_INTERVAL_THRESHOLD,
) -> SeedDistributionReport:
    """Build a seed distribution report from a campaign report directory.

    Detects supported seed-level artifacts and normalizes them. Exits with
    a hard error if no supported seed-level inputs are found.

    Returns:
        A seed distribution report with surfaces from all detected artifacts.

    Raises:
        FileNotFoundError: when no supported seed-level artifacts are found.
    """
    reports_dir = campaign_root / "reports"
    if not reports_dir.is_dir():
        reports_dir = campaign_root

    all_surfaces: list[SurfaceRecord] = []
    found_paths: list[str] = []

    seed_var_path = reports_dir / "seed_variability_by_scenario.json"
    if seed_var_path.exists():
        payload = _load_json(seed_var_path)
        all_surfaces.extend(
            adapt_seed_variability_report(
                payload,
                primary_metric=primary_metric,
                insufficient_seed_threshold=insufficient_seed_threshold,
                wide_interval_threshold=wide_interval_threshold,
                input_artifact=str(seed_var_path),
            )
        )
        found_paths.append(str(seed_var_path))

    rank_stability_path = reports_dir / "headline_ci_rank_stability.json"
    if rank_stability_path.exists():
        payload = _load_json(rank_stability_path)
        all_surfaces.extend(
            adapt_rank_stability_report(
                payload,
                primary_metric=primary_metric,
                insufficient_seed_threshold=insufficient_seed_threshold,
                wide_interval_threshold=wide_interval_threshold,
                input_artifact=str(rank_stability_path),
            )
        )
        found_paths.append(str(rank_stability_path))

    if not all_surfaces:
        msg = (
            f"No supported seed-level artifacts found in {campaign_root}. "
            f"Expected seed_variability_by_scenario.json or "
            f"headline_ci_rank_stability.json in {reports_dir}."
        )
        raise FileNotFoundError(msg)

    return build_seed_distribution_report(
        all_surfaces,
        campaign_root=str(campaign_root),
        report_paths=found_paths,
    )


def format_report_markdown(report: SeedDistributionReport) -> str:
    """Render a compact Markdown summary of the report.

    Returns:
        A human-readable Markdown summary.
    """
    lines: list[str] = []
    lines.append("# Seed Distribution Report")
    lines.append("")
    lines.append(f"Schema: `{report.schema_version}`")
    lines.append(f"Generated: {report.generated_at_utc}")
    lines.append(f"Source: `{report.source.campaign_root}`")
    lines.append(f"Surfaces: {len(report.surfaces)}")
    lines.append("")
    lines.append(f"> {report.interpretation_boundary}")
    lines.append("")

    n_stable = sum(1 for s in report.surfaces if not s.diagnostics.advisory_only)
    n_insufficient = sum(1 for s in report.surfaces if s.diagnostics.insufficient_seed_count)
    n_wide = sum(1 for s in report.surfaces if s.diagnostics.wide_interval)
    n_unstable = sum(1 for s in report.surfaces if s.diagnostics.unstable_rank)

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Stable surfaces: {n_stable}")
    lines.append(f"- Insufficient seed count: {n_insufficient}")
    lines.append(f"- Wide intervals: {n_wide}")
    lines.append(f"- Unstable rank: {n_unstable}")
    lines.append("")

    lines.append("## Surfaces")
    lines.append("")
    lines.append("| Surface | Metric | Seeds | Estimate | CI | Status |")
    lines.append("|---------|--------|-------|----------|-----|--------|")
    for surf in report.surfaces:
        ci_str = f"[{surf.interval.lower:.4f}, {surf.interval.upper:.4f}]"
        flags: list[str] = []
        if surf.diagnostics.insufficient_seed_count:
            flags.append("insufficient")
        if surf.diagnostics.wide_interval:
            flags.append("wide_ci")
        if surf.diagnostics.unstable_rank:
            flags.append("unstable_rank")
        status = ", ".join(flags) if flags else "ok"
        est_str = f"{surf.point_estimate:.4f}" if math.isfinite(surf.point_estimate) else "nan"
        lines.append(
            f"| {surf.surface_id} | {surf.metric} | {surf.seed_count} | "
            f"{est_str} | {ci_str} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)
