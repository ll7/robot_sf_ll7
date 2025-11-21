"""
Research report orchestrator (User Story 1)
Coordinates report generation: collect_metadata, generate_report
"""

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.research.aggregation import (
    aggregate_metrics,
    export_metrics_csv,
    export_metrics_json,
)
from robot_sf.research.figures import (
    plot_distributions,
    plot_learning_curve,
    plot_sample_efficiency,
)
from robot_sf.research.report_template import MarkdownReportRenderer
from robot_sf.research.statistics import evaluate_hypothesis


class ReportOrchestrator:
    """Orchestrates end-to-end research report generation."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_metadata(self) -> dict[str, Any]:
        """
        Collect reproducibility metadata: git hash, packages, hardware.
        Returns dict matching ReproducibilityMetadata schema.
        """
        metadata: dict[str, Any] = {}

        # Git metadata
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            )
            git_dirty = bool(git_status.strip())
        except subprocess.CalledProcessError:
            git_commit = "unknown"
            git_branch = "unknown"
            git_dirty = True

        metadata["git_commit"] = git_commit if len(git_commit) == 40 else "0" * 40
        metadata["git_branch"] = git_branch
        metadata["git_dirty"] = git_dirty

        # Python version
        metadata["python_version"] = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

        # Key packages (subset for brevity)
        key_packages = {}
        try:
            import numpy

            key_packages["numpy"] = numpy.__version__
        except ImportError:
            pass
        try:
            import scipy

            key_packages["scipy"] = scipy.__version__
        except ImportError:
            pass
        try:
            import matplotlib

            key_packages["matplotlib"] = matplotlib.__version__
        except ImportError:
            pass

        metadata["key_packages"] = key_packages

        # Hardware
        import psutil

        hardware = {
            "cpu_model": platform.processor() or "Unknown",
            "cpu_cores": psutil.cpu_count(logical=True),
            "memory_gb": int(psutil.virtual_memory().total / (1024**3)),
            "gpu_model": None,
            "gpu_memory_gb": None,
        }

        # Try to detect GPU (optional)
        try:
            import pynvml  # type: ignore  # Optional dependency

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            hardware["gpu_model"] = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            hardware["gpu_memory_gb"] = int(mem_info.total / (1024**3))
            pynvml.nvmlShutdown()
        except (ImportError, RuntimeError):
            # pynvml not installed or GPU not available
            pass

        metadata["hardware"] = hardware
        metadata["timestamp"] = datetime.now().isoformat()

        return metadata

    def generate_report(
        self,
        experiment_name: str,
        metric_records: list[dict[str, Any]],
        run_id: str,
        seeds: list[int],
        baseline_timesteps: list[float] | None = None,
        pretrained_timesteps: list[float] | None = None,
        baseline_rewards: list[list[float]] | None = None,
        pretrained_rewards: list[list[float]] | None = None,
        threshold: float = 40.0,
    ) -> Path:
        """
        Generate complete research report.

        Args:
            experiment_name: Human-readable experiment label
            metric_records: List of per-seed metric dicts (MetricRecord format)
            run_id: Unique run identifier
            seeds: Random seeds used
            baseline_timesteps: Timesteps to convergence for baseline (for hypothesis eval)
            pretrained_timesteps: Timesteps to convergence for pretrained (for hypothesis eval)
            baseline_rewards: Learning curves for baseline (for figure)
            pretrained_rewards: Learning curves for pretrained (for figure)
            threshold: Hypothesis threshold percentage

        Returns:
            Path to generated report.md
        """
        logger.info(f"Starting report generation for '{experiment_name}'")

        # Create output structure
        figures_dir = self.output_dir / "figures"
        data_dir = self.output_dir / "data"
        configs_dir = self.output_dir / "configs"
        for d in [figures_dir, data_dir, configs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Aggregate metrics
        logger.info("Aggregating metrics...")
        aggregated = aggregate_metrics(
            metric_records, group_by="policy_type", ci_samples=1000, seed=42
        )

        # Export metrics
        export_metrics_json(aggregated, str(data_dir / "metrics.json"))
        export_metrics_csv(aggregated, str(data_dir / "metrics.csv"))

        # Evaluate hypothesis
        logger.info("Evaluating hypothesis...")
        if baseline_timesteps and pretrained_timesteps:
            hypothesis_result = evaluate_hypothesis(
                baseline_timesteps, pretrained_timesteps, threshold
            )
        else:
            hypothesis_result = {"decision": "INCOMPLETE", "note": "Timesteps data not provided"}

        # Export hypothesis
        with open(data_dir / "hypothesis.json", "w", encoding="utf-8") as f:
            json.dump({"schema_version": "1.0.0", "hypotheses": [hypothesis_result]}, f, indent=2)

        # Generate figures
        logger.info("Generating figures...")
        figures = []

        if baseline_rewards and pretrained_rewards and len(baseline_rewards[0]) > 0:
            timesteps = [float(i) for i in range(len(baseline_rewards[0]))]
            fig_lc = plot_learning_curve(
                timesteps,
                baseline_rewards,
                pretrained_rewards,
                figures_dir,
                {"n_seeds": len(seeds)},
            )
            figures.append(fig_lc)

        if baseline_timesteps and pretrained_timesteps:
            fig_se = plot_sample_efficiency(
                baseline_timesteps, pretrained_timesteps, figures_dir, {"n_seeds": len(seeds)}
            )
            figures.append(fig_se)

        # Distribution plots for success/collision rates
        baseline_success = [
            float(r["success_rate"])
            for r in metric_records
            if r.get("policy_type") == "baseline" and "success_rate" in r
        ]
        pretrained_success = [
            float(r["success_rate"])
            for r in metric_records
            if r.get("policy_type") == "pretrained" and "success_rate" in r
        ]

        if baseline_success and pretrained_success:
            fig_dist_success = plot_distributions(
                baseline_success,
                pretrained_success,
                "success_rate",
                figures_dir,
                {"n_seeds": len(seeds)},
            )
            figures.append(fig_dist_success)

        baseline_collision = [
            float(r["collision_rate"])
            for r in metric_records
            if r.get("policy_type") == "baseline" and "collision_rate" in r
        ]
        pretrained_collision = [
            float(r["collision_rate"])
            for r in metric_records
            if r.get("policy_type") == "pretrained" and "collision_rate" in r
        ]

        if baseline_collision and pretrained_collision:
            fig_dist_collision = plot_distributions(
                baseline_collision,
                pretrained_collision,
                "collision_rate",
                figures_dir,
                {"n_seeds": len(seeds)},
            )
            figures.append(fig_dist_collision)

        # Collect reproducibility metadata
        logger.info("Collecting reproducibility metadata...")
        repro_metadata = self.collect_metadata()

        # Assemble full metadata
        full_metadata = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "seeds": seeds,
            "reproducibility": repro_metadata,
            "artifacts": [],  # Will be populated with file paths
        }

        # Save metadata
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(full_metadata, f, indent=2)

        # Render report
        logger.info("Rendering Markdown report...")
        renderer = MarkdownReportRenderer(self.output_dir)
        report_path = renderer.render(
            experiment_name, hypothesis_result, aggregated, figures, full_metadata
        )

        # Export LaTeX (optional)
        logger.info("Exporting LaTeX...")
        renderer.export_latex(report_path)

        logger.info(f"Report generation complete: {report_path}")
        return report_path
