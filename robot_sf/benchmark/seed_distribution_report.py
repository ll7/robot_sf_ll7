"""Seed distribution report generator for multi-seed benchmark evidence.

This module defines the seed_distribution_report.v1 schema and provides adapters
to normalize existing seed-level artifacts into a common, auditable format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION = "seed_distribution_report.v1"
_SUPPORTED_SCHEMA_VERSIONS = frozenset({SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION})

_INSUFFICIENT_SEED_THRESHOLD = 10
_WIDE_INTERVAL_THRESHOLDS: dict[str, float] = {
    "success": 0.15,
    "collision": 0.10,
    "collisions": 0.10,
    "near_miss": 0.20,
    "near_misses": 0.20,
    "snqi": 0.25,
}
_DEFAULT_WIDE_INTERVAL_THRESHOLD = 0.20


def validate_schema_version(payload: dict[str, Any]) -> None:
    """Fail closed on unknown schema versions.

    Args:
        payload: Dictionary with a ``schema_version`` key.

    Raises:
        ValueError: If the schema version is missing or not supported.
    """
    version = payload.get("schema_version")
    if version is None:
        raise ValueError(
            "payload is missing required field 'schema_version'; "
            "refusing to process an unversioned seed distribution report"
        )
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported seed distribution report schema version: {version!r}; "
            f"supported versions: {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )


def build_seed_distribution_report(
    campaign_root: str | Path,
    *,
    generated_by: str = "build_seed_distribution_report",
) -> dict[str, Any]:
    """Build a seed distribution report from existing campaign artifacts.

    Searches for supported input artifacts (seed_variability.json,
    statistical_sufficiency.json, CI rank stability reports) and normalizes
    them into the seed_distribution_report.v1 schema.

    Args:
        campaign_root: Path to the campaign root directory.
        generated_by: Identifier for the generating tool.

    Returns:
        Dictionary conforming to seed_distribution_report.v1 schema.

    Raises:
        FileNotFoundError: If no supported input artifacts are found.
    """
    campaign_root = Path(campaign_root)
    if not campaign_root.exists():
        raise FileNotFoundError(f"Campaign root does not exist: {campaign_root}")

    surfaces: list[dict[str, Any]] = []
    report_paths: list[str] = []

    for adapter, filename in [
        (_adapt_seed_variability, "seed_variability.json"),
        (_adapt_statistical_sufficiency, "statistical_sufficiency.json"),
        (_adapt_ci_rank_stability, "issue_3216_headline_ci_rank_stability.json"),
    ]:
        path = campaign_root / filename
        if path.exists():
            surfaces.extend(adapter(path))
            report_paths.append(filename)

    if not surfaces:
        raise FileNotFoundError(
            f"No supported seed-level artifacts found in {campaign_root}. "
            "Expected one or more of: seed_variability.json, "
            "statistical_sufficiency.json, or "
            "issue_3216_headline_ci_rank_stability.json"
        )

    return {
        "schema_version": SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION,
        "source": {
            "campaign_root": str(campaign_root),
            "report_paths": report_paths,
            "generated_by": generated_by,
            "commit": _get_git_commit(campaign_root),
        },
        "surfaces": surfaces,
    }


def _build_surface(  # noqa: PLR0913
    *,
    surface_name: str,
    scenario_id: str | None,
    planner_id: str | None,
    metric: str,
    unit: str | None,
    seed_count: int | None,
    episode_count: int | None,
    raw_counts: dict[str, Any] | None,
    point_estimate: float | None,
    interval: dict[str, Any] | None,
    seed_distribution: dict[str, Any] | None,
    diagnostics: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build a single surface entry with canonical field order.

    Returns:
        Dictionary with all required seed_distribution_report.v1 surface fields.
    """
    return {
        "surface_id": surface_name,
        "scenario_id": scenario_id,
        "planner_id": planner_id,
        "metric": metric,
        "unit": unit,
        "seed_count": seed_count,
        "episode_count": episode_count,
        "raw_counts": raw_counts,
        "point_estimate": point_estimate,
        "interval": interval,
        "seed_distribution": seed_distribution,
        "diagnostics": diagnostics,
        "provenance": provenance,
    }


def _adapt_seed_variability(path: Path) -> list[dict[str, Any]]:
    """Adapt seed_variability.json to seed_distribution_report.v1 surfaces.

    Returns:
        List of normalized surface dictionaries.
    """
    with open(path) as f:
        data = json.load(f)

    surfaces: list[dict[str, Any]] = []
    provenance_base = {"input_artifact": "seed_variability.json"}

    for row in data.get("rows", []):
        scenario_id = row.get("scenario_id")
        planner_id = row.get("planner_id", row.get("planner"))
        seed_count = row.get("seed_count")
        episode_count = row.get("episode_count")

        for metric_name, metric_data in row.get("metrics", {}).items():
            per_seed_means = metric_data.get("per_seed_means", [])
            mean = metric_data.get("mean")
            std = metric_data.get("std")
            ci_low = metric_data.get("ci_low")
            ci_high = metric_data.get("ci_high")
            confidence_level = metric_data.get("confidence_level", 0.95)
            method = metric_data.get("method", "bootstrap")

            raw_counts = None
            if metric_data.get("raw_numerator") is not None:
                raw_counts = {
                    "numerator": metric_data["raw_numerator"],
                    "denominator": metric_data["raw_denominator"],
                }

            interval = None
            if ci_low is not None and ci_high is not None:
                interval = {
                    "method": method,
                    "lower": ci_low,
                    "upper": ci_high,
                    "confidence_level": confidence_level,
                }

            seed_distribution = None
            if per_seed_means:
                seed_distribution = {
                    "values": per_seed_means,
                    "mean": mean,
                    "std": std,
                    "min": min(per_seed_means),
                    "max": max(per_seed_means),
                }

            diag = _compute_diagnostics(
                seed_count=seed_count,
                unstable_rank=metric_data.get("rank_changed_across_seeds"),
                wide_interval=_is_interval_wide(ci_low, ci_high, metric_name),
            )

            surfaces.append(
                _build_surface(
                    surface_name="seed_variability",
                    scenario_id=scenario_id,
                    planner_id=planner_id,
                    metric=metric_name,
                    unit=metric_data.get("unit"),
                    seed_count=seed_count,
                    episode_count=episode_count,
                    raw_counts=raw_counts,
                    point_estimate=mean,
                    interval=interval,
                    seed_distribution=seed_distribution,
                    diagnostics=diag,
                    provenance={
                        **provenance_base,
                        "input_schema_version": row.get("schema_version"),
                    },
                )
            )

    return surfaces


def _adapt_statistical_sufficiency(path: Path) -> list[dict[str, Any]]:
    """Adapt statistical_sufficiency.json to seed_distribution_report.v1 surfaces.

    Returns:
        List of normalized surface dictionaries.
    """
    with open(path) as f:
        data = json.load(f)

    surfaces: list[dict[str, Any]] = []
    provenance_base = {"input_artifact": "statistical_sufficiency.json"}

    for row in data.get("rows", []):
        scenario_id = row.get("scenario_id")
        planner_id = row.get("planner_id", row.get("planner"))
        sufficiency_status = row.get("sufficiency_status", "unknown")
        seed_count = row.get("seed_count")
        episode_count = row.get("episode_count")

        for metric_name, metric_data in row.get("metrics", {}).items():
            ci_low = metric_data.get("ci_low")
            ci_high = metric_data.get("ci_high")
            confidence_level = metric_data.get("confidence_level", 0.95)
            mean = metric_data.get("mean")

            interval = None
            if ci_low is not None and ci_high is not None:
                interval = {
                    "method": "bootstrap",
                    "lower": ci_low,
                    "upper": ci_high,
                    "confidence_level": confidence_level,
                }

            diag = _compute_diagnostics(
                seed_count=seed_count,
                unstable_rank=None,
                wide_interval=_is_interval_wide(ci_low, ci_high, metric_name),
                advisory_only=sufficiency_status == "insufficient",
            )

            surfaces.append(
                _build_surface(
                    surface_name="statistical_sufficiency",
                    scenario_id=scenario_id,
                    planner_id=planner_id,
                    metric=metric_name,
                    unit=metric_data.get("unit"),
                    seed_count=seed_count,
                    episode_count=episode_count,
                    raw_counts=metric_data.get("raw_counts"),
                    point_estimate=mean,
                    interval=interval,
                    seed_distribution=None,
                    diagnostics=diag,
                    provenance={
                        **provenance_base,
                        "input_schema_version": row.get("schema_version"),
                    },
                )
            )

    return surfaces


def _adapt_ci_rank_stability(path: Path) -> list[dict[str, Any]]:
    """Adapt CI rank stability report to seed_distribution_report.v1 surfaces.

    Returns:
        List of normalized surface dictionaries.
    """
    with open(path) as f:
        data = json.load(f)

    surfaces: list[dict[str, Any]] = []
    provenance_base = {"input_artifact": "issue_3216_headline_ci_rank_stability.json"}

    for planner_data in data.get("planners", []):
        planner_id = planner_data.get("planner_id", planner_data.get("planner"))

        for scenario_data in planner_data.get("scenarios", []):
            scenario_id = scenario_data.get("scenario_id")

            for metric_name, metric_data in scenario_data.get("metrics", {}).items():
                ci_low = metric_data.get("ci_low")
                ci_high = metric_data.get("ci_high")
                confidence_level = metric_data.get("confidence_level", 0.95)
                mean = metric_data.get("mean")
                unstable_rank = metric_data.get("rank_unstable", False)
                seed_count = metric_data.get("seed_count")
                episode_count = metric_data.get("episode_count")

                interval = None
                if ci_low is not None and ci_high is not None:
                    interval = {
                        "method": "bootstrap",
                        "lower": ci_low,
                        "upper": ci_high,
                        "confidence_level": confidence_level,
                    }

                diag = _compute_diagnostics(
                    seed_count=seed_count,
                    unstable_rank=unstable_rank,
                    wide_interval=_is_interval_wide(ci_low, ci_high, metric_name),
                )

                surfaces.append(
                    _build_surface(
                        surface_name="ci_rank_stability",
                        scenario_id=scenario_id,
                        planner_id=planner_id,
                        metric=metric_name,
                        unit=metric_data.get("unit"),
                        seed_count=seed_count,
                        episode_count=episode_count,
                        raw_counts=metric_data.get("raw_counts"),
                        point_estimate=mean,
                        interval=interval,
                        seed_distribution=None,
                        diagnostics=diag,
                        provenance={
                            **provenance_base,
                            "input_schema_version": data.get("schema_version"),
                        },
                    )
                )

    return surfaces


def _compute_diagnostics(
    *,
    seed_count: int | None,
    unstable_rank: bool | None,
    wide_interval: bool,
    advisory_only: bool = False,
) -> dict[str, Any]:
    """Compute diagnostic flags for a surface.

    Args:
        seed_count: Number of seeds used in the measurement.
        unstable_rank: Whether rank ordering changed across seed resampling.
        wide_interval: Whether confidence interval is considered wide.
        advisory_only: Whether the measurement is advisory-only due to insufficient evidence.

    Returns:
        Diagnostic dictionary with boolean flags.
    """
    insufficient = seed_count is not None and seed_count < _INSUFFICIENT_SEED_THRESHOLD

    return {
        "insufficient_seed_count": insufficient,
        "unstable_rank": bool(unstable_rank) if unstable_rank is not None else False,
        "wide_interval": wide_interval,
        "advisory_only": advisory_only,
    }


def _is_interval_wide(
    ci_low: float | None,
    ci_high: float | None,
    metric_name: str,
) -> bool:
    """Determine if a confidence interval is wide for the given metric.

    Returns:
        True when the interval width exceeds the metric-specific threshold.
    """
    if ci_low is None or ci_high is None:
        return False

    width = ci_high - ci_low
    threshold = _WIDE_INTERVAL_THRESHOLDS.get(metric_name, _DEFAULT_WIDE_INTERVAL_THRESHOLD)
    return width > threshold


def _get_git_commit(campaign_root: Path) -> str | None:
    """Attempt to extract git commit hash from campaign metadata.

    Returns:
        Git commit hash string when available, or ``None``.
    """
    metadata_path = campaign_root / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            return metadata.get("git_commit") or metadata.get("commit")
        except (OSError, json.JSONDecodeError):
            pass
    return None


def render_markdown(report: dict[str, Any]) -> str:
    """Render a seed distribution report as a compact Markdown summary.

    Args:
        report: Seed distribution report dictionary.

    Returns:
        Markdown string with the report summary.
    """
    validate_schema_version(report)

    source = report.get("source", {})
    surfaces = report.get("surfaces", [])

    lines: list[str] = []
    lines.append("# Seed Distribution Report")
    lines.append("")
    lines.append(f"**Schema version:** {report.get('schema_version')}")
    lines.append(f"**Campaign root:** {source.get('campaign_root')}")
    lines.append(f"**Generated by:** {source.get('generated_by')}")
    if source.get("commit"):
        lines.append(f"**Commit:** {source.get('commit')}")
    lines.append(f"**Report paths:** {', '.join(source.get('report_paths', []))}")
    lines.append("")

    scenario_count = len({s.get("scenario_id") for s in surfaces if s.get("scenario_id")})
    planner_count = len({s.get("planner_id") for s in surfaces if s.get("planner_id")})
    lines.append(
        f"**Total surfaces:** {len(surfaces)} | "
        f"**Scenarios:** {scenario_count} | "
        f"**Planners:** {planner_count}"
    )
    lines.append("")

    diag_insufficient = sum(
        1 for s in surfaces if s.get("diagnostics", {}).get("insufficient_seed_count")
    )
    diag_unstable = sum(1 for s in surfaces if s.get("diagnostics", {}).get("unstable_rank"))
    diag_wide = sum(1 for s in surfaces if s.get("diagnostics", {}).get("wide_interval"))
    diag_advisory = sum(1 for s in surfaces if s.get("diagnostics", {}).get("advisory_only"))
    lines.append("## Diagnostics Summary")
    lines.append("")
    lines.append(f"- Insufficient seed count: {diag_insufficient}")
    lines.append(f"- Unstable rank: {diag_unstable}")
    lines.append(f"- Wide interval: {diag_wide}")
    lines.append(f"- Advisory only: {diag_advisory}")
    lines.append("")

    lines.append("## Surfaces")
    lines.append("")
    for surface in surfaces:
        sid = surface.get("surface_id", "unknown")
        scenario = surface.get("scenario_id") or "aggregate"
        planner = surface.get("planner_id") or "aggregate"
        metric = surface.get("metric")
        pe = surface.get("point_estimate")
        seeds = surface.get("seed_count")
        interval = surface.get("interval")

        pe_str = f"{pe:.4f}" if pe is not None else "N/A"
        ci_str = ""
        if interval:
            ci_str = f" (CI: [{interval['lower']:.4f}, {interval['upper']:.4f}])"

        diag = surface.get("diagnostics", {})
        diag_flags = [k for k, v in diag.items() if v and k != "advisory_only"]
        diag_str = ""
        if diag_flags:
            diag_str = f" [{', '.join(diag_flags)}]"

        lines.append(
            f"- **{sid}** | {scenario} | {planner} | {metric} "
            f"| seeds={seeds} | est={pe_str}{ci_str}{diag_str}"
        )

    lines.append("")
    lines.append(
        "Note: Statistical repeatability does not imply simulator fidelity, "
        "scenario coverage, or real-world validity."
    )
    return "\n".join(lines) + "\n"


def write_report(
    report: dict[str, Any],
    *,
    out_json: str | Path | None = None,
    out_md: str | Path | None = None,
) -> dict[str, Path]:
    """Write a seed distribution report to JSON and optionally Markdown files.

    Args:
        report: Seed distribution report dictionary.
        out_json: Path to write JSON output.
        out_md: Path to write Markdown output.

    Returns:
        Dictionary mapping output type to written file path.
    """
    validate_schema_version(report)

    written: dict[str, Path] = {}

    if out_json is not None:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        written["json"] = path

    if out_md is not None:
        path = Path(out_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(report), encoding="utf-8")
        written["md"] = path

    return written
