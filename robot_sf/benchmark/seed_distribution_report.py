"""Seed distribution report generator for multi-seed benchmark evidence.

This module defines the seed_distribution_report.v1 schema and provides adapters
to normalize existing seed-level artifacts into a common, auditable format.

The adapters consume the *real* on-disk artifact shapes emitted by the existing
benchmark pipeline:

- ``seed_variability_by_scenario.json`` (camera-ready seed-variability export)
- ``statistical_sufficiency.json`` (seed-sufficiency gate output)
- the headline CI rank-stability report (``result.json`` /
  ``issue_3216_headline_ci_rank_stability.json``)

Each adapter is a pure reader over the loaded JSON document and emits a list of
normalized ``surface`` dictionaries conforming to ``seed_distribution_report.v1``.
"""

from __future__ import annotations

import json
import math
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

    Searches for supported input artifacts in ``campaign_root`` and its
    ``reports/`` subdirectory and normalizes them into the
    ``seed_distribution_report.v1`` schema. No simulations or campaign runs are
    launched.

    Args:
        campaign_root: Path to the campaign root directory.
        generated_by: Identifier for the generating tool.

    Returns:
        Dictionary conforming to seed_distribution_report.v1 schema.

    Raises:
        FileNotFoundError: If the campaign root does not exist or no supported
            input artifacts are found.
    """
    campaign_root = Path(campaign_root)
    if not campaign_root.exists():
        raise FileNotFoundError(f"Campaign root does not exist: {campaign_root}")

    # (candidate filenames, adapter, accept-predicate). Candidate filenames are
    # checked in <root>/<name> then <root>/reports/<name>; the first existing
    # match wins. The accept-predicate guards generic names (e.g. result.json)
    # so an unrelated file with the same name is not mis-parsed.
    specs: list[
        tuple[
            tuple[str, ...],
            Any,
            Any,
        ]
    ] = [
        (
            ("seed_variability_by_scenario.json", "seed_variability.json"),
            _adapt_seed_variability,
            lambda data: True,
        ),
        (("statistical_sufficiency.json",), _adapt_statistical_sufficiency, lambda data: True),
        (
            (
                "issue_3216_headline_ci_rank_stability.json",
                "result.json",
            ),
            _adapt_ci_rank_stability,
            _is_ci_rank_stability_report,
        ),
    ]

    surfaces: list[dict[str, Any]] = []
    report_paths: list[str] = []

    for filenames, adapt, accept in specs:
        path = _find_artifact(campaign_root, filenames)
        if path is None:
            continue
        data = _load_json(path)
        if data is None or not accept(data):
            continue
        produced = adapt(data)
        if produced:
            surfaces.extend(produced)
            report_paths.append(path.name)

    if not surfaces:
        raise FileNotFoundError(
            f"No supported seed-level artifacts found in {campaign_root}. "
            "Expected one or more of: seed_variability_by_scenario.json, "
            "statistical_sufficiency.json, or a headline CI rank-stability "
            "report (result.json / issue_3216_headline_ci_rank_stability.json) "
            "in the root or its reports/ subdirectory."
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


def _find_artifact(root: Path, filenames: tuple[str, ...]) -> Path | None:
    """Return the first existing artifact path for the given candidate names.

    Each candidate is searched at ``root/<name>`` then ``root/reports/<name>`` to
    match both layout conventions used by the benchmark pipeline.
    """
    for name in filenames:
        for candidate in (root / name, root / "reports" / name):
            if candidate.is_file():
                return candidate
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object, returning None on read/parse failure.

    Returns:
        Loaded mapping, or None on read/parse failure / non-object content.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


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


def _finite_float(value: Any) -> float | None:
    """Coerce a value to float, returning None when missing or non-finite.

    Returns:
        The value as a finite float, or None when missing/non-finite/uncoercible.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _confidence_block(data: dict[str, Any]) -> dict[str, Any]:
    """Return the method/confidence block from a seed-level artifact."""
    confidence = data.get("confidence")
    if isinstance(confidence, dict):
        return confidence
    return {}


def _adapt_seed_variability(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt ``seed_variability_by_scenario.json`` to v1 surfaces.

    The canonical input shape is::

        {"schema_version": ..., "confidence": {method, confidence, ...},
         "rows": [{
            scenario_id, planner_key, algo, seed_count, episode_count,
            "per_seed": [{"seed", "episode_count", "metrics": {metric: value}}],
            "summary": {metric: {mean, std, ci_low, ci_high, ci_half_width, count}},
            ...}]}

    Returns:
        List of normalized surface dictionaries.
    """
    confidence = _confidence_block(data)
    method = confidence.get("method", "bootstrap")
    confidence_level = confidence.get("confidence", 0.95)
    provenance_base = {
        "input_artifact": "seed_variability_by_scenario.json",
        "input_schema_version": data.get("schema_version"),
    }

    surfaces: list[dict[str, Any]] = []

    for row in data.get("rows", []):
        scenario_id = row.get("scenario_id")
        planner_id = (
            row.get("planner_key") or row.get("planner_id") or row.get("planner") or row.get("algo")
        )
        seed_count = row.get("seed_count")
        episode_count = row.get("episode_count")
        per_seed_entries = row.get("per_seed") or []
        summary = row.get("summary") if isinstance(row.get("summary"), dict) else row.get("metrics")

        for metric_name, metric_data in (summary or {}).items():
            if not isinstance(metric_data, dict):
                continue
            mean = _finite_float(metric_data.get("mean"))
            std = _finite_float(metric_data.get("std"))
            ci_low = _finite_float(metric_data.get("ci_low"))
            ci_high = _finite_float(metric_data.get("ci_high"))

            per_seed_means: list[float] = []
            for entry in per_seed_entries:
                if isinstance(entry, dict):
                    value = (
                        entry.get("metrics", {}).get(metric_name)
                        if isinstance(entry.get("metrics"), dict)
                        else None
                    )
                    coerced = _finite_float(value)
                    if coerced is not None:
                        per_seed_means.append(coerced)

            interval = None
            if ci_low is not None and ci_high is not None:
                interval = {
                    "method": method,
                    "lower": ci_low,
                    "upper": ci_high,
                    "confidence_level": confidence_level,
                }

            seed_distribution: dict[str, Any] | None = None
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
                # Per-surface rank drift is not emitted by the current
                # seed-variability export; honor it when a future/source row
                # provides ``rank_changed_across_seeds`` (nullable otherwise).
                unstable_rank=metric_data.get("rank_changed_across_seeds"),
                wide_interval=_is_interval_wide(ci_low, ci_high, metric_name),
            )

            surfaces.append(
                _build_surface(
                    surface_name="seed_variability",
                    scenario_id=scenario_id,
                    planner_id=planner_id,
                    metric=metric_name,
                    unit=None,
                    seed_count=seed_count,
                    episode_count=episode_count,
                    raw_counts=None,
                    point_estimate=mean,
                    interval=interval,
                    seed_distribution=seed_distribution,
                    diagnostics=diag,
                    provenance=dict(provenance_base),
                )
            )

    return surfaces


def _adapt_statistical_sufficiency(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt ``statistical_sufficiency.json`` to v1 surfaces.

    The sufficiency report carries CI *half-widths* (precision) rather than point
    estimates, so ``point_estimate`` is None and the interval records the
    ``half_width``. Point-estimate information is not synthesized.

    Returns:
        List of normalized surface dictionaries.
    """
    confidence = _confidence_block(data)
    method = confidence.get("method", "bootstrap")
    confidence_level = confidence.get("confidence", 0.95)
    provenance_base = {
        "input_artifact": "statistical_sufficiency.json",
        "input_schema_version": data.get("schema_version"),
    }

    surfaces: list[dict[str, Any]] = []

    for row in data.get("rows", []):
        scenario_id = row.get("scenario_id")
        planner_id = (
            row.get("planner_key") or row.get("planner_id") or row.get("planner") or row.get("algo")
        )
        seed_count = row.get("seed_count")
        episode_count = row.get("episode_count")
        sufficiency_status = row.get("sufficiency_status", "unknown")

        for metric_name, metric_data in (row.get("metrics") or {}).items():
            if not isinstance(metric_data, dict):
                continue
            half_width = _finite_float(metric_data.get("ci_half_width"))

            interval = None
            if half_width is not None:
                interval = {
                    "method": method,
                    "lower": None,
                    "upper": None,
                    "half_width": half_width,
                    "confidence_level": confidence_level,
                }

            diag = _compute_diagnostics(
                seed_count=seed_count,
                unstable_rank=None,
                wide_interval=_width_is_wide(
                    half_width * 2 if half_width is not None else None, metric_name
                ),
                advisory_only=sufficiency_status == "insufficient",
            )

            surfaces.append(
                _build_surface(
                    surface_name="statistical_sufficiency",
                    scenario_id=scenario_id,
                    planner_id=planner_id,
                    metric=metric_name,
                    unit=None,
                    seed_count=seed_count,
                    episode_count=episode_count,
                    raw_counts=None,
                    point_estimate=None,
                    interval=interval,
                    seed_distribution=None,
                    diagnostics=diag,
                    provenance=dict(provenance_base),
                )
            )

    return surfaces


def _is_ci_rank_stability_report(data: dict[str, Any]) -> bool:
    """Return True when a loaded document is a headline CI rank-stability report."""
    schema_version = str(data.get("schema_version", ""))
    return schema_version.startswith("issue_3216")


def _adapt_ci_rank_stability(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt the headline CI rank-stability report to v1 surfaces.

    The report stores a flat ``cells`` list (one per scenario/planner); each
    cell's ``metrics`` use the canonical per-metric stat dict
    (``mean``/``std``/``ci_low``/``ci_high``/``ci_half_width``). Cells excluded
    from the headline ranking (``counted == false``) are retained and flagged
    ``advisory_only``. Per-cell rank instability is not a field in the source
    report, so ``unstable_rank`` is left null.

    Returns:
        List of normalized surface dictionaries.
    """
    confidence = _confidence_block(data) or (data.get("config") or {})
    method = confidence.get("method", "bootstrap")
    confidence_level = confidence.get("confidence", 0.95)
    provenance_base = {
        "input_artifact": "issue_3216_headline_ci_rank_stability",
        "input_schema_version": data.get("schema_version"),
    }

    surfaces: list[dict[str, Any]] = []

    for cell in data.get("cells", []):
        if not isinstance(cell, dict):
            continue
        scenario_id = cell.get("scenario_id")
        planner_id = cell.get("planner_key") or cell.get("planner_id") or cell.get("planner")
        seed_count = cell.get("seed_count")
        counted = cell.get("counted")
        advisory_only = counted is False

        for metric_name, metric_data in (cell.get("metrics") or {}).items():
            if not isinstance(metric_data, dict):
                continue
            mean = _finite_float(metric_data.get("mean"))
            ci_low = _finite_float(metric_data.get("ci_low"))
            ci_high = _finite_float(metric_data.get("ci_high"))

            interval = None
            if ci_low is not None and ci_high is not None:
                interval = {
                    "method": method,
                    "lower": ci_low,
                    "upper": ci_high,
                    "confidence_level": confidence_level,
                }

            diag = _compute_diagnostics(
                seed_count=seed_count,
                # Per-cell rank instability is not a field in the source report;
                # honor ``rank_unstable`` when a future/source cell provides it.
                unstable_rank=metric_data.get("rank_unstable"),
                wide_interval=_is_interval_wide(ci_low, ci_high, metric_name),
                advisory_only=advisory_only,
            )

            surfaces.append(
                _build_surface(
                    surface_name="ci_rank_stability",
                    scenario_id=scenario_id,
                    planner_id=planner_id,
                    metric=metric_name,
                    unit=None,
                    seed_count=seed_count,
                    episode_count=None,
                    raw_counts=None,
                    point_estimate=mean,
                    interval=interval,
                    seed_distribution=None,
                    diagnostics=diag,
                    provenance=dict(provenance_base),
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


def _width_is_wide(width: float | None, metric_name: str) -> bool:
    """Classify an interval width as wide for the given metric.

    Returns:
        True when the width exceeds the metric-specific threshold.
    """
    if width is None:
        return False
    threshold = _WIDE_INTERVAL_THRESHOLDS.get(metric_name, _DEFAULT_WIDE_INTERVAL_THRESHOLD)
    try:
        return float(width) > threshold
    except (TypeError, ValueError):
        return False


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
    try:
        return _width_is_wide(float(ci_high) - float(ci_low), metric_name)
    except (TypeError, ValueError):
        return False


def _get_git_commit(campaign_root: Path) -> str | None:
    """Attempt to extract git commit hash from campaign metadata.

    Returns:
        Git commit hash string when available, or ``None``.
    """
    metadata_path = campaign_root / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, encoding="utf-8") as f:
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
        interval = surface.get("interval") or {}

        pe_str = f"{pe:.4f}" if isinstance(pe, (int, float)) else "N/A"
        lo = interval.get("lower")
        hi = interval.get("upper")
        half_width = interval.get("half_width")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            ci_str = f" (CI: [{lo:.4f}, {hi:.4f}])"
        elif isinstance(half_width, (int, float)):
            ci_str = f" (CI half-width: {half_width:.4f})"
        else:
            ci_str = ""

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
            json.dumps(report, indent=2, sort_keys=True, default=str, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        written["json"] = path

    if out_md is not None:
        path = Path(out_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(report), encoding="utf-8")
        written["md"] = path

    return written
