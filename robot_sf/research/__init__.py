"""Research reporting module for automated imitation learning analysis.

This module provides tools for generating publication-ready research reports from
multi-seed imitation learning experiments, including:
- Multi-seed metric aggregation with bootstrap confidence intervals
- Statistical analysis (paired t-tests, effect sizes, hypothesis evaluation)
- Automated figure generation (learning curves, distributions, comparisons)
- Markdown and LaTeX report rendering
- Reproducibility metadata tracking

Public API:
    aggregate_metrics / bootstrap_ci: Metric aggregation utilities with bootstrap CIs
    evaluate_hypothesis / paired_t_test / cohen_d: Statistical helpers
    plot_learning_curve / plot_sample_efficiency / plot_distributions: Figure utilities
    MarkdownReportRenderer: Render Markdown/LaTeX reports from templates
    ReportOrchestrator: End-to-end report generation coordinator
    AblationOrchestrator: Ablation study analysis coordinator

Example:
    >>> from robot_sf.research import ReportOrchestrator
    >>> orchestrator = ReportOrchestrator()
    >>> orchestrator.generate_report(
    ...     experiment_name="Demo",
    ...     metric_records=[],
    ...     run_id="demo_run",
    ...     seeds=[],
    ... )
"""

from __future__ import annotations

__version__ = "0.1.0"

# Eager imports: All components imported immediately for API availability
# (All names in __all__ are defined via direct imports below)
from robot_sf.research.aggregation import (
    aggregate_metrics,
    bootstrap_ci,
    export_metrics_csv,
    export_metrics_json,
)
from robot_sf.research.extractor_report import ReportConfig, generate_extractor_report
from robot_sf.research.figures import (
    configure_matplotlib_backend,
    plot_distributions,
    plot_learning_curve,
    plot_sample_efficiency,
)
from robot_sf.research.orchestrator import AblationOrchestrator, ReportOrchestrator
from robot_sf.research.report_template import MarkdownReportRenderer
from robot_sf.research.statistics import cohen_d, evaluate_hypothesis, paired_t_test

__all__ = [
    "AblationOrchestrator",
    "MarkdownReportRenderer",
    "ReportConfig",
    "ReportOrchestrator",
    "aggregate_metrics",
    "bootstrap_ci",
    "cohen_d",
    "configure_matplotlib_backend",
    "evaluate_hypothesis",
    "export_metrics_csv",
    "export_metrics_json",
    "generate_extractor_report",
    "paired_t_test",
    "plot_distributions",
    "plot_learning_curve",
    "plot_sample_efficiency",
]
