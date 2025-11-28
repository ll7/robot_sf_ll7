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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from robot_sf.benchmark.imitation_manifest import get_training_run_manifest_path
from robot_sf.common.artifact_paths import get_imitation_report_dir
from robot_sf.training.hardware_probe import collect_hardware_profile
from robot_sf.training.multi_extractor_models import (
    ExtractorRunRecord,
    HardwareProfile,
    TrainingRunSummary,
)
from robot_sf.training.multi_extractor_summary import write_summary_artifacts


def _load_training_run(run_id: str) -> tuple[dict[str, Any], Path]:
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
    metrics = manifest.get("metrics") or {}
    entry = metrics.get(key) or {}
    if isinstance(entry, dict) and "mean" in entry:
        try:
            return float(entry["mean"])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _status_to_summary(status: str | None) -> str:
    if status == "failed":
        return "failed"
    if status == "partial":
        return "failed"
    return "success"


def _build_record(
    *,
    run_id: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    hardware: HardwareProfile,
    sample_efficiency_ratio: float,
) -> ExtractorRunRecord:
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
        worker_mode="single-thread",
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
    baseline_metrics: dict[str, float],
    pretrained_metrics: dict[str, float],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Timesteps comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["baseline", "pretrained"],
        [
            baseline_metrics.get("timesteps_to_convergence", 0.0),
            pretrained_metrics.get("timesteps_to_convergence", 0.0),
        ],
        color=["#4c78a8", "#f58518"],
    )
    ax.set_ylabel("Timesteps to convergence")
    ax.set_title("Convergence comparison")
    plt.tight_layout()
    path = output_dir / "timesteps_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["timesteps_comparison"] = path

    # Performance metrics
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["success_rate", "collision_rate", "snqi"]
    baseline_vals = [baseline_metrics.get(m, 0.0) for m in metrics]
    pretrained_vals = [pretrained_metrics.get(m, 0.0) for m in metrics]
    x_pos = range(len(metrics))
    ax.bar([x - 0.15 for x in x_pos], baseline_vals, width=0.3, label="baseline")
    ax.bar([x + 0.15 for x in x_pos], pretrained_vals, width=0.3, label="pretrained")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(["Success", "Collisions", "SNQI"])
    ax.set_ylabel("Metric value")
    ax.set_title("Performance metrics")
    ax.legend()
    plt.tight_layout()
    perf_path = output_dir / "performance_metrics.png"
    fig.savefig(perf_path, dpi=150)
    plt.close(fig)
    paths["performance_metrics"] = perf_path

    return paths


def analyze_imitation_results(
    *,
    group_id: str,
    baseline_run_id: str,
    pretrained_run_id: str,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    baseline_manifest, baseline_manifest_path = _load_training_run(baseline_run_id)
    pretrained_manifest, pretrained_manifest_path = _load_training_run(pretrained_run_id)

    baseline_metrics = {
        "timesteps_to_convergence": _metric_mean(baseline_manifest, "timesteps_to_convergence"),
        "success_rate": _metric_mean(baseline_manifest, "success_rate"),
        "collision_rate": _metric_mean(baseline_manifest, "collision_rate"),
        "snqi": _metric_mean(baseline_manifest, "snqi"),
    }
    pretrained_metrics = {
        "timesteps_to_convergence": _metric_mean(pretrained_manifest, "timesteps_to_convergence"),
        "success_rate": _metric_mean(pretrained_manifest, "success_rate"),
        "collision_rate": _metric_mean(pretrained_manifest, "collision_rate"),
        "snqi": _metric_mean(pretrained_manifest, "snqi"),
    }

    if not output_dir:
        output_dir = get_imitation_report_dir() / "analysis" / group_id
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_ts = baseline_metrics["timesteps_to_convergence"]
    pretrained_ts = pretrained_metrics["timesteps_to_convergence"]
    sample_efficiency_ratio = baseline_ts / pretrained_ts if pretrained_ts > 0 else 0.0

    hardware = collect_hardware_profile(worker_count=1, skip_gpu=True)
    now = datetime.now(UTC).isoformat()

    baseline_record = _build_record(
        run_id=baseline_run_id,
        manifest=baseline_manifest,
        manifest_path=baseline_manifest_path,
        hardware=hardware,
        sample_efficiency_ratio=1.0,
    )
    pretrained_record = _build_record(
        run_id=pretrained_run_id,
        manifest=pretrained_manifest,
        manifest_path=pretrained_manifest_path,
        hardware=hardware,
        sample_efficiency_ratio=sample_efficiency_ratio,
    )

    aggregate_metrics = {
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
        hardware_overview=[hardware],
        extractor_results=[baseline_record, pretrained_record],
        aggregate_metrics=aggregate_metrics,
        notes=None,
    )

    artifacts = write_summary_artifacts(summary=summary, destination=output_dir)
    figures = _generate_figures(
        baseline_metrics=baseline_metrics,
        pretrained_metrics=pretrained_metrics,
        output_dir=output_dir / "figures",
    )

    return {
        "summary_json": artifacts["json"],
        "summary_markdown": artifacts["markdown"],
        "figures_dir": output_dir / "figures",
        "timesteps_figure": figures["timesteps_comparison"],
        "performance_figure": figures["performance_metrics"],
    }
