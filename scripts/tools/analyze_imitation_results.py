"""Automated results collection and analysis for imitation learning runs.

Loads baseline and pretrained training manifests, computes sample-efficiency
statistics, generates comparison figures, and writes a summary compliant with
``training_summary.schema.json``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from loguru import logger

from robot_sf.research.cli_args import add_imitation_report_common_args
from robot_sf.research.imitation_report import (
    ImitationReportConfig,
    generate_imitation_report,
)
from robot_sf.telemetry import (
    ManifestWriter,
    PipelineRunRecord,
    PipelineRunStatus,
    PipelineStepDefinition,
    ProgressTracker,
    RunTrackerConfig,
    TelemetrySampler,
    TensorBoardAdapter,
    generate_run_id,
)
from robot_sf.training import analyze_imitation_results


class AnalysisTracker:
    """Optional run tracker session for the analysis/report generation workflow."""

    def __init__(
        self,
        *,
        tracker_root: Path | None,
        run_id_hint: str | None,
        initiator: str | None,
        enable_report_step: bool,
        enable_tensorboard: bool,
        tensorboard_logdir: Path | None,
    ) -> None:
        """TODO docstring. Document this function.

        Args:
            tracker_root: TODO docstring.
            run_id_hint: TODO docstring.
            initiator: TODO docstring.
            enable_report_step: TODO docstring.
            enable_tensorboard: TODO docstring.
            tensorboard_logdir: TODO docstring.
        """
        self.enabled = bool(tracker_root or run_id_hint or enable_tensorboard)
        self.config: RunTrackerConfig | None = None
        self.writer: ManifestWriter | None = None
        self.progress: ProgressTracker | None = None
        self._adapter: TensorBoardAdapter | None = None
        self._sampler: TelemetrySampler | None = None
        self._tensorboard_step = "tensorboard" if enable_tensorboard else None
        self.run_id = run_id_hint or generate_run_id("imitation-analysis")
        self.initiator = initiator
        self.started_at = datetime.now(UTC)
        if not self.enabled:
            return
        self.config = RunTrackerConfig(artifact_root=tracker_root)
        self.run_id = self.run_id or generate_run_id("imitation-analysis")
        self.writer = ManifestWriter(self.config, self.run_id)
        steps = [PipelineStepDefinition("analysis", "Aggregate & Figures")]
        if enable_report_step:
            steps.append(PipelineStepDefinition("report", "Report Generation"))
        if self._tensorboard_step:
            steps.append(PipelineStepDefinition(self._tensorboard_step, "TensorBoard Export"))
        self.progress = ProgressTracker(steps, writer=self.writer, log_fn=logger.info)
        self.progress.enable_failure_guard(heartbeat=self._heartbeat)
        self._write_run_record(PipelineRunStatus.RUNNING)

        self._sampler = TelemetrySampler(
            self.writer,
            progress_tracker=self.progress,
            started_at=self.started_at,
            interval_seconds=self.config.telemetry_interval_seconds,
        )
        if enable_tensorboard:
            logdir = tensorboard_logdir or (self.writer.run_directory / "tensorboard")
            adapter = TensorBoardAdapter(log_dir=logdir)
            if adapter.is_available:
                self._adapter = adapter
                self._sampler.add_consumer(adapter.consume_snapshot)
            else:
                logger.warning(
                    "TensorBoard adapter unavailable; install torch or tensorboardX to enable mirroring."
                )
                self._tensorboard_step = None
        self._sampler.start()

    def has_step(self, step_id: str) -> bool:
        """TODO docstring. Document this function.

        Args:
            step_id: TODO docstring.

        Returns:
            TODO docstring.
        """
        return bool(
            self.enabled
            and self.progress
            and any(entry.step_id == step_id for entry in self.progress.entries)
        )

    @property
    def run_directory(self) -> Path | None:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        if not self.enabled or not self.writer:
            return None
        return self.writer.run_directory

    @contextmanager
    def step(self, step_id: str):
        """TODO docstring. Document this function.

        Args:
            step_id: TODO docstring.
        """
        if not self.enabled or not self.progress or not self.has_step(step_id):
            yield
            return
        self.progress.start_step(step_id)
        try:
            yield
        except Exception as exc:  # pragma: no cover - propagated upstream
            self.progress.fail_step(step_id, reason=str(exc))
            self.fail({"error": str(exc)})
            raise
        else:
            self.progress.complete_step(step_id)

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        """TODO docstring. Document this function.

        Args:
            summary: TODO docstring.
        """
        if not self.enabled:
            return
        self._write_run_record(PipelineRunStatus.COMPLETED, summary=summary)
        self._close()

    def fail(self, summary: dict[str, Any] | None = None) -> None:
        """TODO docstring. Document this function.

        Args:
            summary: TODO docstring.
        """
        if not self.enabled:
            return
        self._write_run_record(PipelineRunStatus.FAILED, summary=summary or {})
        self._close()

    def _heartbeat(self, status: PipelineRunStatus) -> None:
        """TODO docstring. Document this function.

        Args:
            status: TODO docstring.
        """
        if not self.enabled:
            return
        self._write_run_record(status)

    def _write_run_record(
        self,
        status: PipelineRunStatus,
        *,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """TODO docstring. Document this function.

        Args:
            status: TODO docstring.
            summary: TODO docstring.
        """
        if not self.enabled or not self.progress:
            return
        record = PipelineRunRecord(
            run_id=self.run_id,
            created_at=self.started_at,
            status=status,
            enabled_steps=[entry.step_id for entry in self.progress.entries],
            artifact_dir=self.writer.run_directory,
            initiator=self.initiator,
            summary=summary or {},
            steps=self.progress.clone_entries(),
        )
        self.writer.append_run_record(record)

    def _close(self) -> None:
        """TODO docstring. Document this function."""
        if self._sampler:
            self._sampler.stop(flush_final=True)
            self._sampler = None
        if self._adapter:
            self._adapter.close()
            self._adapter = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for imitation results analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        required=True,
        help="Identifier for this comparison group (becomes the summary run_id).",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline training run ID.",
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        help="Pre-trained training run ID.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory (defaults under imitation report root).",
    )
    add_imitation_report_common_args(
        parser,
        alpha_flag="--significance-level",
        alpha_dest="significance_level",
        include_threshold=False,
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate Markdown/LaTeX report after analysis.",
    )
    parser.add_argument(
        "--report-output-root",
        type=Path,
        default=Path("output/research_reports"),
        help="Output root for generated reports (when --generate-report is used).",
    )
    parser.add_argument(
        "--tracker-root",
        type=Path,
        default=None,
        help="Optional artifact root for run tracker outputs.",
    )
    parser.add_argument(
        "--tracker-run-id",
        type=str,
        default=None,
        help="Override run tracker run_id (default: auto-generated).",
    )
    parser.add_argument(
        "--tracker-initiator",
        type=str,
        default=None,
        help="Label recorded as the run initiator in tracker manifests.",
    )
    parser.add_argument(
        "--enable-tensorboard",
        action="store_true",
        help="Mirror telemetry to TensorBoard during analysis/report generation.",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        type=Path,
        default=None,
        help="Custom TensorBoard log directory (defaults to run directory/tensorboard).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _export_metrics_to_tensorboard(metrics: dict[str, float], log_dir: Path) -> Path | None:
    """Write scalar metrics to TensorBoard for interactive inspection."""

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # pragma: no cover - optional dependency
        try:
            from tensorboardX import SummaryWriter  # type: ignore
        except ImportError:
            logger.warning(
                "TensorBoard SummaryWriter unavailable; install torch or tensorboardX to export scalars."
            )
            return None

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))  # type: ignore[misc]
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"analysis/{key}", float(value), 0)
    writer.flush()
    writer.close()
    return log_dir


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    """Entry point: run analysis and emit summary artifacts."""
    args = parse_args(argv)
    tracker = AnalysisTracker(
        tracker_root=args.tracker_root,
        run_id_hint=args.tracker_run_id,
        initiator=args.tracker_initiator,
        enable_report_step=args.generate_report,
        enable_tensorboard=args.enable_tensorboard,
        tensorboard_logdir=args.tensorboard_logdir,
    )
    tracker_summary: dict[str, str] = {}
    tb_logged: Path | None = None
    try:
        with tracker.step("analysis"):
            artifacts = analyze_imitation_results(
                group_id=args.group,
                baseline_run_id=args.baseline,
                pretrained_run_id=args.pretrained,
                output_dir=args.output,
            )
        logger.success("Analysis complete", summary=str(artifacts["summary_json"]))
        tracker_summary["summary_json"] = str(artifacts["summary_json"])
        aggregated_path = artifacts.get("aggregated_metrics_json")
        if aggregated_path:
            tracker_summary["aggregated_metrics_json"] = str(aggregated_path)

        if (args.enable_tensorboard or args.tensorboard_logdir) and artifacts.get(
            "aggregate_metrics"
        ):
            log_dir = args.tensorboard_logdir
            if log_dir is None and tracker.run_directory is not None:
                log_dir = tracker.run_directory / "tensorboard-scalars"
            elif log_dir is None:
                log_dir = Path("output") / "tensorboard" / args.group
            tb_logged = _export_metrics_to_tensorboard(artifacts["aggregate_metrics"], log_dir)
            if tb_logged:
                tracker_summary["tensorboard_scalars"] = str(tb_logged)

        if args.generate_report:
            with tracker.step("report"):
                cfg = ImitationReportConfig(
                    experiment_name=args.experiment_name,
                    hypothesis=args.hypothesis,
                    alpha=args.significance_level,
                    export_latex=args.export_latex,
                    baseline_run_id=args.baseline,
                    pretrained_run_id=args.pretrained,
                    num_seeds=args.num_seeds,
                )
                report_paths = generate_imitation_report(
                    summary_path=artifacts["summary_json"],
                    output_root=args.report_output_root,
                    config=cfg,
                )
            logger.success("Report generated", report=str(report_paths["report"]))
            tracker_summary["report"] = str(report_paths["report"])
            if report_paths.get("report_pdf"):
                tracker_summary["report_pdf"] = str(report_paths["report_pdf"])

        if tracker.has_step("tensorboard"):
            with tracker.step("tensorboard"):
                if tb_logged:
                    logger.info("TensorBoard scalars written to {}", tb_logged)

        tracker.finish(summary=tracker_summary)
        return 0
    except Exception as exc:
        tracker.fail({"error": str(exc)})
        raise


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
