"""Analysis helpers for imitation learning training runs.

Loads baseline and pretrained training manifests, computes sample-efficiency
statistics, generates comparison figures, and emits a summary compatible with
``training_summary.schema.json`` for downstream reporting.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from loguru import logger

from robot_sf.benchmark.imitation_manifest import get_training_run_manifest_path
from robot_sf.common.artifact_paths import get_imitation_report_dir
from robot_sf.common.metrics_utils import metric_samples
from robot_sf.research.aggregation import aggregate_metrics, export_metrics_json
from robot_sf.research.figures import (
    plot_distributions,
    plot_improvement_summary,
    plot_sample_efficiency,
)
from robot_sf.training.multi_extractor_models import (
    ExtractorRunRecord,
    HardwareProfile,
    TrainingRunSummary,
)
from robot_sf.training.multi_extractor_summary import write_summary_artifacts


def _load_training_run(run_id: str) -> tuple[dict[str, Any], Path]:
    """Load a training manifest for run_id with fallbacks to prefixed/nested variants.

    Returns:
        tuple[dict[str, Any], Path]: The manifest payload and the resolved manifest path.
    """
    manifest_path = get_training_run_manifest_path(run_id)
    base_runs_dir = manifest_path.parent
    if not manifest_path.exists():
        candidates = sorted(
            base_runs_dir.glob(f"{run_id}*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if candidates:
            manifest_path = candidates[0]
            logger.warning("Canonical manifest missing; using closest match {}", manifest_path.name)
        else:
            nested = sorted(
                get_imitation_report_dir().glob(f"**/runs/{run_id}*.json"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True,
            )
            if nested:
                manifest_path = nested[0]
                logger.warning("Canonical manifest missing; using {}", manifest_path)
            else:
                raise FileNotFoundError(
                    f"Training run manifest not found for {run_id} (searched {base_runs_dir})"
                )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training run manifest for {run_id} must be an object")
    return payload, manifest_path


def _metric_mean(manifest: dict[str, Any], key: str) -> float:
    """Return the mean value for a metric in a manifest, defaulting to 0.0."""
    metrics = manifest.get("metrics") or {}
    entry = metrics.get(key) or {}
    if isinstance(entry, dict) and "mean" in entry:
        try:
            return float(entry["mean"])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _status_to_summary(status: str | None) -> str:
    """Normalize manifest statuses into schema-accepted success/failed values.

    Returns:
        str: "success" or "failed" according to schema expectations.
    """
    if status == "failed":
        return "failed"
    if status == "partial":
        return "failed"
    return "success"


def _hardware_profile_from_manifest(manifest: dict[str, Any]) -> HardwareProfile:
    """Extract hardware metadata from a manifest or return an 'unknown' placeholder.

    Returns:
        HardwareProfile: Normalized hardware profile extracted from the manifest.
    """

    def _as_profile(payload: dict[str, Any]) -> HardwareProfile | None:
        """TODO docstring. Document this function.

        Args:
            payload: TODO docstring.

        Returns:
            TODO docstring.
        """
        try:
            platform = str(payload.get("platform", "unknown"))
            arch = str(payload.get("arch", "unknown"))
            python_version = str(payload.get("python_version", "unknown"))
            workers = int(payload.get("workers", 1) or 1)
            gpu_model = payload.get("gpu_model")
            cuda_version = payload.get("cuda_version")
            return HardwareProfile(
                platform=platform,
                arch=arch,
                python_version=python_version,
                workers=workers,
                gpu_model=gpu_model,
                cuda_version=cuda_version,
            )
        except (TypeError, ValueError):
            return None

    if "hardware_profile" in manifest and isinstance(manifest["hardware_profile"], dict):
        candidate = _as_profile(manifest["hardware_profile"])
        if candidate:
            return candidate
    if "hardware_overview" in manifest and isinstance(manifest["hardware_overview"], list):
        for item in manifest["hardware_overview"]:
            if isinstance(item, dict):
                candidate = _as_profile(item)
                if candidate:
                    return candidate

    return HardwareProfile(
        platform="unknown",
        arch="unknown",
        python_version="unknown",
        workers=1,
        gpu_model=None,
        cuda_version=None,
    )


def _build_record(
    *,
    run_id: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    hardware: HardwareProfile,
    sample_efficiency_ratio: float,
) -> ExtractorRunRecord:
    """Convert a training manifest into an ExtractorRunRecord for summary emission.

    Returns:
        ExtractorRunRecord: Structured record for inclusion in the run summary.
    """
    now = datetime.now(UTC).isoformat()
    timesteps = _metric_mean(manifest, "timesteps_to_convergence")
    success_rate = _metric_mean(manifest, "success_rate")
    collision_rate = _metric_mean(manifest, "collision_rate")
    snqi = _metric_mean(manifest, "snqi")
    return ExtractorRunRecord(
        config_name=run_id,
        status=_status_to_summary(manifest.get("status")),
        start_time=manifest.get("created_at", now),
        end_time=manifest.get("created_at", now),
        duration_seconds=None,
        hardware_profile=hardware,
        worker_mode=manifest.get("worker_mode", "single-thread"),
        training_steps=int(timesteps) if timesteps else 0,
        metrics={
            "timesteps_to_convergence": timesteps,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "snqi": snqi,
            "sample_efficiency_ratio": sample_efficiency_ratio,
        },
        artifacts={"manifest": str(manifest_path)},
        reason=None,
    )


def _generate_figures(
    *,
    baseline_metrics: dict[str, Any],
    pretrained_metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate comparison figures using research-standard helpers.

    Returns:
        dict[str, Path]: Mapping of figure identifiers to file paths.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"n_seeds": len(baseline_metrics.get("timesteps_samples") or [])}
    paths: dict[str, Path] = {}

    baseline_ts_samples = baseline_metrics.get("timesteps_samples") or []
    pretrained_ts_samples = pretrained_metrics.get("timesteps_samples") or []
    if baseline_ts_samples and pretrained_ts_samples:
        se = plot_sample_efficiency(
            baseline_ts_samples,
            pretrained_ts_samples,
            output_dir,
            metadata=metadata,
        )
        paths["fig-sample-efficiency"] = se["paths"]["png"]

    for metric_name in ("success_rate", "collision_rate", "snqi"):
        base_samples = baseline_metrics.get(f"{metric_name}_samples") or []
        pre_samples = pretrained_metrics.get(f"{metric_name}_samples") or []
        if base_samples and pre_samples:
            dist = plot_distributions(
                base_samples,
                pre_samples,
                metric_name,
                output_dir,
                metadata,
            )
            paths[f"fig-{metric_name}-distribution"] = dist["paths"]["png"]

    improvement = plot_improvement_summary(
        {
            "timesteps_to_convergence": baseline_metrics.get("timesteps_to_convergence", 0.0),
            "success_rate": baseline_metrics.get("success_rate", 0.0),
            "collision_rate": baseline_metrics.get("collision_rate", 0.0),
            "snqi": baseline_metrics.get("snqi", 0.0),
        },
        {
            "timesteps_to_convergence": pretrained_metrics.get("timesteps_to_convergence", 0.0),
            "success_rate": pretrained_metrics.get("success_rate", 0.0),
            "collision_rate": pretrained_metrics.get("collision_rate", 0.0),
            "snqi": pretrained_metrics.get("snqi", 0.0),
        },
        output_dir,
        metadata=metadata,
    )
    if improvement["paths"]:
        paths["fig-improvement-summary"] = improvement["paths"]["png"]

    return paths


def _build_metric_records(
    baseline_metrics: dict[str, Any],
    pretrained_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand per-metric sample lists into aggregator-friendly records.

    Returns:
        list[dict[str, Any]]: Records suitable for aggregation by condition.
    """

    records: list[dict[str, Any]] = []
    for condition, metrics in (
        ("baseline", baseline_metrics),
        ("pretrained", pretrained_metrics),
    ):
        sample_keys = [key for key in metrics if key.endswith("_samples")]
        if not sample_keys:
            continue
        max_len = max(len(metrics[key] or []) for key in sample_keys)
        for idx in range(max_len):
            record: dict[str, Any] = {"condition": condition, "seed": idx}
            for key in sample_keys:
                values = metrics.get(key) or []
                if idx < len(values) and isinstance(values[idx], (int, float)):
                    record[key.replace("_samples", "")] = float(values[idx])
            records.append(record)
    return records


def analyze_imitation_results(
    *,
    group_id: str,
    baseline_run_id: str,
    pretrained_run_id: str,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Summarize baseline vs pretrained runs with metrics, figures, and schema-compliant JSON.

    Returns:
        dict[str, Path]: Paths to produced artifacts and figures plus aggregate metrics.
    """
    baseline_manifest, baseline_manifest_path = _load_training_run(baseline_run_id)
    pretrained_manifest, pretrained_manifest_path = _load_training_run(pretrained_run_id)

    baseline_metrics = {
        "timesteps_to_convergence": _metric_mean(baseline_manifest, "timesteps_to_convergence"),
        "timesteps_samples": metric_samples(baseline_manifest, "timesteps_to_convergence"),
        "success_rate": _metric_mean(baseline_manifest, "success_rate"),
        "success_rate_samples": metric_samples(baseline_manifest, "success_rate"),
        "collision_rate": _metric_mean(baseline_manifest, "collision_rate"),
        "collision_rate_samples": metric_samples(baseline_manifest, "collision_rate"),
        "snqi": _metric_mean(baseline_manifest, "snqi"),
        "snqi_samples": metric_samples(baseline_manifest, "snqi"),
    }
    pretrained_metrics = {
        "timesteps_to_convergence": _metric_mean(pretrained_manifest, "timesteps_to_convergence"),
        "timesteps_samples": metric_samples(pretrained_manifest, "timesteps_to_convergence"),
        "success_rate": _metric_mean(pretrained_manifest, "success_rate"),
        "success_rate_samples": metric_samples(pretrained_manifest, "success_rate"),
        "collision_rate": _metric_mean(pretrained_manifest, "collision_rate"),
        "collision_rate_samples": metric_samples(pretrained_manifest, "collision_rate"),
        "snqi": _metric_mean(pretrained_manifest, "snqi"),
        "snqi_samples": metric_samples(pretrained_manifest, "snqi"),
    }

    if not output_dir:
        output_dir = get_imitation_report_dir() / "analysis" / group_id
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_ts = baseline_metrics["timesteps_to_convergence"]
    pretrained_ts = pretrained_metrics["timesteps_to_convergence"]
    sample_efficiency_ratio = baseline_ts / pretrained_ts if pretrained_ts > 0 else 0.0

    baseline_hardware = _hardware_profile_from_manifest(baseline_manifest)
    pretrained_hardware = _hardware_profile_from_manifest(pretrained_manifest)
    now = datetime.now(UTC).isoformat()

    baseline_record = _build_record(
        run_id=baseline_run_id,
        manifest=baseline_manifest,
        manifest_path=baseline_manifest_path,
        hardware=baseline_hardware,
        sample_efficiency_ratio=1.0,
    )
    pretrained_record = _build_record(
        run_id=pretrained_run_id,
        manifest=pretrained_manifest,
        manifest_path=pretrained_manifest_path,
        hardware=pretrained_hardware,
        sample_efficiency_ratio=sample_efficiency_ratio,
    )

    aggregate_summary = {
        "sample_efficiency_ratio": sample_efficiency_ratio,
        "baseline_timesteps_to_convergence": baseline_ts,
        "pretrained_timesteps_to_convergence": pretrained_ts,
        "success_rate_delta": pretrained_metrics["success_rate"] - baseline_metrics["success_rate"],
        "collision_rate_delta": pretrained_metrics["collision_rate"]
        - baseline_metrics["collision_rate"],
        "snqi_delta": pretrained_metrics["snqi"] - baseline_metrics["snqi"],
    }

    summary = TrainingRunSummary(
        run_id=group_id,
        created_at=now,
        output_root=str(output_dir),
        hardware_overview=[baseline_hardware, pretrained_hardware],
        extractor_results=[baseline_record, pretrained_record],
        aggregate_metrics=aggregate_summary,
        notes=None,
    )

    metric_records = _build_metric_records(baseline_metrics, pretrained_metrics)
    aggregated = aggregate_metrics(metric_records, group_by="condition") if metric_records else []
    aggregated_json = data_dir / "aggregated_metrics.json"
    export_metrics_json(aggregated, str(aggregated_json))

    artifacts = write_summary_artifacts(summary=summary, destination=output_dir)
    figures = _generate_figures(
        baseline_metrics=baseline_metrics,
        pretrained_metrics=pretrained_metrics,
        output_dir=figures_dir,
    )

    return {
        "summary_json": artifacts["json"],
        "summary_markdown": artifacts["markdown"],
        "figures_dir": figures_dir,
        "aggregated_metrics_json": aggregated_json,
        "aggregate_metrics": aggregate_summary,
        "figures": figures,
    }
