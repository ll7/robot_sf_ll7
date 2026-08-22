"""Aggregation, artifact persistence, and final publication for Full Classic runs."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from .io_utils import write_manifest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class VisualizationError(Exception):
    """Fallback exception type supplied by the facade when optional viz is unavailable."""


def update_scaling_efficiency(manifest, cfg) -> dict[str, Any]:
    """Update runtime, throughput, and explicitly diagnostic scaling fields.

    Returns:
        The updated manifest scaling-efficiency mapping.
    """
    now = time.time()
    manifest.runtime_sec = max(0.0, now - manifest.created_at)
    manifest.workers = int(getattr(cfg, "workers", 1) or 1)
    if manifest.runtime_sec > 0:
        manifest.episodes_per_second = manifest.executed_jobs / manifest.runtime_sec
    throughput_per_worker = (
        manifest.episodes_per_second / manifest.workers if manifest.workers > 0 else 0.0
    )
    compatibility_efficiency = (
        1.0 / manifest.workers
        if manifest.workers > 0 and manifest.episodes_per_second > 0.0
        else 0.0
    )
    evidence_status = (
        "smoke_only_non_evidence" if bool(getattr(cfg, "smoke", False)) else "diagnostic_only"
    )
    manifest.scaling_efficiency = {
        "runtime_sec": manifest.runtime_sec,
        "executed_jobs": manifest.executed_jobs,
        "skipped_jobs": manifest.skipped_jobs,
        "episodes_per_second": manifest.episodes_per_second,
        "workers": manifest.workers,
        "throughput_per_worker": throughput_per_worker,
        "parallel_efficiency": "not_available",
        "parallel_efficiency_basis": "requires measured sequential baseline",
        "evidence_status": evidence_status,
        "parallel_efficiency_placeholder": compatibility_efficiency,
        "parallel_efficiency_placeholder_deprecated": True,
        "parallel_efficiency_placeholder_note": (
            "Deprecated compatibility alias; not benchmark-strength evidence."
        ),
    }
    return manifest.scaling_efficiency


def write_json(path: Path, obj) -> None:
    """Write JSON atomically through a same-directory temporary file."""
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, sort_keys=True)
        tmp.replace(path)
    except (OSError, TypeError) as exc:
        logger.warning("Failed writing JSON artifact {}: {}", path, exc)


def serialize_groups(groups) -> list[dict[str, Any]]:
    """Serialize aggregation group objects into the established JSON schema.

    Returns:
        List of JSON-compatible aggregation group dictionaries.
    """
    return [
        {
            "archetype": group.archetype,
            "density": group.density,
            "count": group.count,
            "metrics": {
                key: {
                    "mean": metric.mean,
                    "median": metric.median,
                    "p95": metric.p95,
                    "mean_ci": metric.mean_ci,
                    "median_ci": metric.median_ci,
                }
                for key, metric in group.metrics.items()
            },
        }
        for group in groups
    ]


def serialize_effects(effects) -> list[dict[str, Any]]:
    """Serialize effect-size reports into the established JSON schema.

    Returns:
        List of JSON-compatible effect report dictionaries.
    """
    return [
        {
            "archetype": report.archetype,
            "comparisons": [
                {
                    "metric": comparison.metric,
                    "density_low": comparison.density_low,
                    "density_high": comparison.density_high,
                    "diff": comparison.diff,
                    "standardized": comparison.standardized,
                }
                for comparison in report.comparisons
            ],
        }
        for report in effects
    ]


def serialize_precision(report) -> dict[str, Any]:
    """Serialize a precision report into a JSON-friendly dictionary.

    Returns:
        JSON-compatible precision report dictionary.
    """
    return {
        "final_pass": report.final_pass,
        "evaluations": [
            {
                "scenario_id": evaluation.scenario_id,
                "archetype": evaluation.archetype,
                "density": evaluation.density,
                "episodes": evaluation.episodes,
                "all_pass": evaluation.all_pass,
                "metric_status": [
                    {
                        "metric": status.metric,
                        "half_width": status.half_width,
                        "target": status.target,
                        "passed": status.passed,
                    }
                    for status in evaluation.metric_status
                ],
            }
            for evaluation in report.evaluations
        ],
    }


def write_iteration_artifacts(root: Path, groups, effects, precision_report) -> None:
    """Persist one adaptive-loop aggregation/report snapshot."""
    write_json(root / "aggregates" / "summary.json", serialize_groups(groups))
    write_json(root / "reports" / "effect_sizes.json", serialize_effects(effects))
    write_json(
        root / "reports" / "statistical_sufficiency.json",
        serialize_precision(precision_report),
    )


def publish_visual_artifacts(  # noqa: PLR0913
    root: Path,
    cfg,
    groups,
    all_records,
    *,
    visual_generator: Callable[..., Any],
    visualization_available: bool,
    plot_generator: Callable[..., Any] | None = None,
    validation_fn: Callable[..., Any] | None = None,
    visualization_error: type[Exception] = VisualizationError,
) -> None:
    """Run the post-loop visual publication pass with optional dependencies injected."""
    try:
        visual_generator(root, cfg, groups, all_records)
        if visualization_available and not getattr(cfg, "smoke", False):
            logger.info("Generating additional real visualizations from episode data")
            try:
                plots_dir = root / "plots"
                videos_dir = root / "videos"
                plots_dir.mkdir(exist_ok=True)
                videos_dir.mkdir(exist_ok=True)
                plot_artifacts = plot_generator(all_records, str(root)) if plot_generator else []
                logger.info("Generated {} real plots into {}", len(plot_artifacts), plots_dir)
                video_artifacts = []
                logger.debug(
                    "Skipping legacy episode video generation (sim-view videos already produced)"
                )
                logger.info("Generated sim_view videos into {}", videos_dir)
                all_artifacts = plot_artifacts + video_artifacts
                if validation_fn is not None:
                    validation = validation_fn(all_artifacts)
                    if validation.passed:
                        logger.info("All real visualizations validated successfully")
                    else:
                        logger.warning(
                            "Some visualizations failed validation: {} failed artifacts",
                            len(validation.failed_artifacts),
                        )
            except (visualization_error, FileNotFoundError) as vis_exc:
                logger.warning("Real visualization generation failed (non-fatal): {}", vis_exc)
    except (VisualizationError, FileNotFoundError) as exc:
        logger.warning("Visual artifact generation failed (non-fatal): {}", exc)


def finalize_run(  # noqa: PLR0913
    root: Path,
    cfg,
    manifest,
    *,
    groups,
    all_records,
    write_run_meta_files_fn: Callable[..., None],
    visual_generator: Callable[..., Any],
    visualization_available: bool,
    plot_generator: Callable[..., Any] | None = None,
    validation_fn: Callable[..., Any] | None = None,
    visualization_error: type[Exception] = VisualizationError,
) -> None:
    """Close the run, persist manifest metadata, and publish final artifacts."""
    update_scaling_efficiency(manifest, cfg)
    manifest.scaling_efficiency.setdefault("finalized", True)
    write_manifest(manifest, str(root / "manifest.json"))
    write_run_meta_files_fn(root, cfg, manifest)
    publish_visual_artifacts(
        root,
        cfg,
        groups,
        all_records,
        visual_generator=visual_generator,
        visualization_available=visualization_available,
        plot_generator=plot_generator,
        validation_fn=validation_fn,
        visualization_error=visualization_error,
    )


# Private aliases keep historical imports working while the owner moves to this module.
_update_scaling_efficiency = update_scaling_efficiency
_write_json = write_json
_serialize_groups = serialize_groups
_serialize_effects = serialize_effects
_serialize_precision = serialize_precision
_write_iteration_artifacts = write_iteration_artifacts


__all__ = [
    "finalize_run",
    "publish_visual_artifacts",
    "serialize_effects",
    "serialize_groups",
    "serialize_precision",
    "update_scaling_efficiency",
    "write_iteration_artifacts",
    "write_json",
]
