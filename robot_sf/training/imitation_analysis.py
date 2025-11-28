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
from robot_sf.training.multi_extractor_models import (
    ExtractorRunRecord,
    HardwareProfile,
    TrainingRunSummary,
)
from robot_sf.training.multi_extractor_summary import write_summary_artifacts


def _load_training_run(run_id: str) -> tuple[dict[str, Any], Path]:
    """Load a training manifest for run_id with fallbacks to prefixed/nested variants."""
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
    """Normalize manifest statuses into schema-accepted success/failed values."""
    if status == "failed":
        return "failed"
    if status == "partial":
        return "failed"
    return "success"


def _hardware_profile_from_manifest(manifest: dict[str, Any]) -> HardwareProfile:
    """Extract hardware metadata from a manifest or return an 'unknown' placeholder."""

    def _as_profile(payload: dict[str, Any]) -> HardwareProfile | None:
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
    """Convert a training manifest into an ExtractorRunRecord for summary emission."""
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


def _metric_samples(manifest: dict[str, Any], key: str) -> list[float]:
    """Return a list of float samples for a metric, if available."""

    metrics = manifest.get("metrics") or {}
    samples = metrics.get(f"{key}_samples") or metrics.get(key)
    if isinstance(samples, list):
        return [float(v) for v in samples if isinstance(v, (int, float))]
    return []


def _plot_timesteps_bar(
    baseline_metrics: dict[str, Any], pretrained_metrics: dict[str, Any], out: Path
) -> Path:
    """Plot a simple bar chart comparing timesteps to convergence."""
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
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_performance_bars(
    baseline_metrics: dict[str, Any], pretrained_metrics: dict[str, Any], out: Path
) -> Path:
    """Plot success, collision, and SNQI bars for baseline vs pretrained."""
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
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_learning_curve(
    base_samples: list[float], pre_samples: list[float], out: Path
) -> Path | None:
    """Plot a line-based learning curve from timesteps samples."""
    if not base_samples and not pre_samples:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    if base_samples:
        ax.plot(range(len(base_samples)), base_samples, marker="o", label="baseline")
    if pre_samples:
        ax.plot(range(len(pre_samples)), pre_samples, marker="o", label="pretrained")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Timesteps to convergence")
    ax.set_title("Learning curve (timesteps samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_success_collision_over_time(
    success_base: list[float],
    success_pre: list[float],
    coll_base: list[float],
    coll_pre: list[float],
    out: Path,
) -> Path | None:
    """Plot success and collision rates across samples for both runs."""
    if not (success_base or success_pre or coll_base or coll_pre):
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    if success_base:
        ax.plot(range(len(success_base)), success_base, label="baseline success", marker="o")
    if success_pre:
        ax.plot(range(len(success_pre)), success_pre, label="pretrained success", marker="o")
    if coll_base:
        ax.plot(range(len(coll_base)), coll_base, label="baseline collisions", marker="x")
    if coll_pre:
        ax.plot(range(len(coll_pre)), coll_pre, label="pretrained collisions", marker="x")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Rate")
    ax.set_title("Success/Collision rates over samples")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_snqi_distribution(
    snqi_base: list[float], snqi_pre: list[float], out: Path
) -> Path | None:
    """Plot histograms of SNQI distributions for baseline and pretrained runs."""
    if not (snqi_base or snqi_pre):
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    if snqi_base:
        ax.hist(snqi_base, bins=min(20, max(5, len(snqi_base))), alpha=0.6, label="baseline")
    if snqi_pre:
        ax.hist(snqi_pre, bins=min(20, max(5, len(snqi_pre))), alpha=0.6, label="pretrained")
    ax.set_xlabel("SNQI")
    ax.set_ylabel("Frequency")
    ax.set_title("Performance distribution (SNQI)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _generate_figures(
    *,
    baseline_metrics: dict[str, Any],
    pretrained_metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate comparison figures (timesteps, performance, distributions) for the analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["timesteps_comparison"] = _plot_timesteps_bar(
        baseline_metrics, pretrained_metrics, output_dir / "timesteps_comparison.png"
    )
    paths["performance_metrics"] = _plot_performance_bars(
        baseline_metrics, pretrained_metrics, output_dir / "performance_metrics.png"
    )

    lc_path = _plot_learning_curve(
        baseline_metrics.get("timesteps_samples") or [],
        pretrained_metrics.get("timesteps_samples") or [],
        output_dir / "learning_curve.png",
    )
    if lc_path:
        paths["learning_curve"] = lc_path

    paths["sample_efficiency"] = _plot_timesteps_bar(
        baseline_metrics, pretrained_metrics, output_dir / "sample_efficiency.png"
    )

    sc_path = _plot_success_collision_over_time(
        baseline_metrics.get("success_rate_samples") or [],
        pretrained_metrics.get("success_rate_samples") or [],
        baseline_metrics.get("collision_rate_samples") or [],
        pretrained_metrics.get("collision_rate_samples") or [],
        output_dir / "success_collision_over_time.png",
    )
    if sc_path:
        paths["success_collision_over_time"] = sc_path

    dist_path = _plot_snqi_distribution(
        baseline_metrics.get("snqi_samples") or [],
        pretrained_metrics.get("snqi_samples") or [],
        output_dir / "performance_distribution.png",
    )
    if dist_path:
        paths["performance_distribution"] = dist_path

    return paths


def analyze_imitation_results(
    *,
    group_id: str,
    baseline_run_id: str,
    pretrained_run_id: str,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Summarize baseline vs pretrained runs with metrics, figures, and schema-compliant JSON."""
    baseline_manifest, baseline_manifest_path = _load_training_run(baseline_run_id)
    pretrained_manifest, pretrained_manifest_path = _load_training_run(pretrained_run_id)

    baseline_metrics = {
        "timesteps_to_convergence": _metric_mean(baseline_manifest, "timesteps_to_convergence"),
        "timesteps_samples": _metric_samples(baseline_manifest, "timesteps_to_convergence"),
        "success_rate": _metric_mean(baseline_manifest, "success_rate"),
        "success_rate_samples": _metric_samples(baseline_manifest, "success_rate"),
        "collision_rate": _metric_mean(baseline_manifest, "collision_rate"),
        "collision_rate_samples": _metric_samples(baseline_manifest, "collision_rate"),
        "snqi": _metric_mean(baseline_manifest, "snqi"),
        "snqi_samples": _metric_samples(baseline_manifest, "snqi"),
    }
    pretrained_metrics = {
        "timesteps_to_convergence": _metric_mean(pretrained_manifest, "timesteps_to_convergence"),
        "timesteps_samples": _metric_samples(pretrained_manifest, "timesteps_to_convergence"),
        "success_rate": _metric_mean(pretrained_manifest, "success_rate"),
        "success_rate_samples": _metric_samples(pretrained_manifest, "success_rate"),
        "collision_rate": _metric_mean(pretrained_manifest, "collision_rate"),
        "collision_rate_samples": _metric_samples(pretrained_manifest, "collision_rate"),
        "snqi": _metric_mean(pretrained_manifest, "snqi"),
        "snqi_samples": _metric_samples(pretrained_manifest, "snqi"),
    }

    if not output_dir:
        output_dir = get_imitation_report_dir() / "analysis" / group_id
    output_dir.mkdir(parents=True, exist_ok=True)

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
        hardware_overview=[baseline_hardware, pretrained_hardware],
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
