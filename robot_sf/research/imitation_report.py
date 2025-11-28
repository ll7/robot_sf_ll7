"""Generate research-ready reports for imitation learning comparisons.

Consumes a `summary.json` produced by the imitation analysis tooling and emits
Markdown/optional LaTeX reports with metrics tables, hypothesis evaluation,
figures, and reproducibility metadata.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from scipy.stats import t

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.research.metadata import collect_reproducibility_metadata
from robot_sf.research.statistics import (
    cohen_d,
    evaluate_hypothesis,
    format_test_results,
    paired_t_test,
)


@dataclass
class ImitationReportConfig:
    experiment_name: str
    hypothesis: str | None = "BC pre-training reduces timesteps by ≥30%"
    alpha: float = 0.05
    improvement_threshold_pct: float = 30.0
    export_latex: bool = False
    baseline_run_id: str | None = None
    pretrained_run_id: str | None = None
    config_paths: dict[str, Path] | None = None
    ablation_label: str | None = None
    hyperparameters: dict[str, str] | None = None


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary.json must contain an object")
    return payload


def _select_records(
    summary: dict[str, Any],
    baseline_id: str | None,
    pretrained_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose baseline and pretrained records, validating ambiguity."""

    records = summary.get("extractor_results") or []
    if not records or not isinstance(records, list):
        raise ValueError("summary.extractor_results must contain at least two entries")
    baseline = None
    if baseline_id:
        baseline = next((r for r in records if r.get("config_name") == baseline_id), None)
    if baseline is None:
        baseline = records[0]
    if pretrained_id:
        pretrained = next((r for r in records if r.get("config_name") == pretrained_id), None)
        if pretrained is None:
            raise ValueError(f"Pretrained run id '{pretrained_id}' not found in summary")
    elif len(records) == 2:
        pretrained = next((r for r in records if r is not baseline), None)
    else:
        raise ValueError(
            "summary contains multiple records; specify pretrained_run_id to disambiguate"
        )
    if pretrained is None or pretrained is baseline:
        raise ValueError("summary must contain a pretrained record distinct from baseline")
    return baseline, pretrained


def _metric(record: dict[str, Any], key: str) -> float:
    metrics = record.get("metrics") or {}
    val = metrics.get(key)
    return float(val) if isinstance(val, (int, float)) else 0.0


def _metric_samples(record: dict[str, Any], key: str) -> list[float]:
    """Return a list of sample values for a metric if present."""

    metrics = record.get("metrics") or {}
    samples = metrics.get(f"{key}_samples") or metrics.get(key)
    if isinstance(samples, list):
        return [float(v) for v in samples if isinstance(v, (int, float))]
    return []


def _ci_from_samples(samples: list[float]) -> tuple[float, float] | str:
    """Return mean +/- critical * SE; t-distribution for small n, z for large n.

    Returns "n/a" when fewer than 2 samples are available.
    """

    n = len(samples)
    if n < 2:
        return "n/a"
    mean_val = sum(samples) / n
    variance = sum((x - mean_val) ** 2 for x in samples) / (n - 1)
    se = math.sqrt(variance) / math.sqrt(n)
    critical = t.ppf(0.975, df=n - 1) if n < 30 else 1.96
    delta = critical * se
    return (mean_val - delta, mean_val + delta)


def _fmt_stat(value: float | None | str) -> str:
    return "n/a" if value is None or value == "n/a" else f"{value:.4f}"


def _render_markdown(
    *,
    config: ImitationReportConfig,
    summary: dict[str, Any],
    baseline: dict[str, Any],
    pretrained: dict[str, Any],
    stats: dict[str, Any],
    hypothesis_result: dict[str, Any],
    figures: dict[str, Path],
    metadata_path: Path,
) -> str:
    """Render a Markdown report from summary + stats."""

    run_id = summary.get("run_id", "unknown")
    baseline_id = baseline.get("config_name", "baseline")
    pretrained_id = pretrained.get("config_name", "pretrained")
    lines = [
        f"# {config.experiment_name} Report",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated: {_timestamp()}",
        f"- Baseline: `{baseline_id}`",
        f"- Pre-trained: `{pretrained_id}`",
        f"- Ablation: {config.ablation_label or 'N/A'}",
        f"- Hypothesis: {config.hypothesis or 'N/A'}",
        f"- Significance level: {config.alpha}",
        f"- Metadata: `{metadata_path.name}`",
        "",
        "## Metric Comparison",
        "| Metric | Baseline | Pre-trained | Delta |",
        "| --- | --- | --- | --- |",
    ]
    metric_keys = (
        ("timesteps_to_convergence", "Timesteps to convergence"),
        ("success_rate", "Success rate"),
        ("collision_rate", "Collision rate"),
        ("snqi", "SNQI"),
        ("sample_efficiency_ratio", "Sample efficiency ratio"),
    )
    for key, label in metric_keys:
        base_val = _metric(baseline, key)
        pre_val = _metric(pretrained, key)
        delta = pre_val - base_val
        lines.append(f"| {label} | {base_val:.4f} | {pre_val:.4f} | {delta:.4f} |")

    lines.extend(
        [
            "",
            "## Statistical Test (timesteps to convergence)",
            f"- p-value: {_fmt_stat(stats['p_value'])}",
            f"- Cohen's d: {_fmt_stat(stats['effect_size'])}",
            f"- Significance: {stats['significance']}",
            f"- Interpretation: {stats['interpretation']}",
            f"- Baseline timesteps CI (95%): {stats.get('baseline_ci') or 'n/a'}",
            f"- Pre-trained timesteps CI (95%): {stats.get('pretrained_ci') or 'n/a'}",
            "",
            "## Hypothesis Evaluation",
            f"- Decision: {hypothesis_result.get('decision')}",
            (
                f"- Improvement: {hypothesis_result.get('measured_value'):.2f}% "
                f"(threshold ≥ {config.improvement_threshold_pct}%)"
            )
            if hypothesis_result.get("measured_value") is not None
            else "- Improvement: (insufficient data for improvement calculation)",
            f"- Note: {hypothesis_result.get('note', '')}",
            "",
            "## Figures",
        ]
    )
    for label, path in figures.items():
        lines.append(f"![{label}]({path.name})")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(f"- Metadata: `{metadata_path.name}` (git, hardware, packages, configs)")
    if config.hyperparameters:
        lines.append("## Hyperparameters")
        for key, value in sorted(config.hyperparameters.items()):
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    return (
        text.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _render_latex(
    *,
    config: ImitationReportConfig,
    summary: dict[str, Any],
    baseline: dict[str, Any],
    pretrained: dict[str, Any],
    stats: dict[str, Any],
    hypothesis_result: dict[str, Any],
    figures: dict[str, Path],
    metadata_path: Path,
    output_path: Path,
) -> None:
    """Render a LaTeX report from summary + stats."""

    baseline_id = _latex_escape(str(baseline.get("config_name", "baseline")))
    pretrained_id = _latex_escape(str(pretrained.get("config_name", "pretrained")))
    lines = [
        r"\documentclass{article}",
        r"\usepackage{graphicx}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        rf"\section*{{{_latex_escape(config.experiment_name)} Report}}",
        rf"\textbf{{Baseline}}: \texttt{{{baseline_id}}}\\",
        rf"\textbf{{Pre-trained}}: \texttt{{{pretrained_id}}}\\",
        rf"\textbf{{Hypothesis}}: {_latex_escape(config.hypothesis or 'N/A')}\\",
        rf"\textbf{{Significance level}}: {config.alpha}\\",
        rf"\textbf{{Metadata}}: \texttt{{{_latex_escape(metadata_path.name)}}}\\",
        r"\section*{Metric Comparison}",
        r"\begin{tabular}{llll}",
        r"\textbf{Metric} & \textbf{Baseline} & \textbf{Pre-trained} & \textbf{Delta} \\",
        r"\hline",
    ]
    metric_keys = (
        ("timesteps_to_convergence", "Timesteps to convergence"),
        ("success_rate", "Success rate"),
        ("collision_rate", "Collision rate"),
        ("snqi", "SNQI"),
        ("sample_efficiency_ratio", "Sample efficiency ratio"),
    )
    for key, label in metric_keys:
        base_val = _metric(baseline, key)
        pre_val = _metric(pretrained, key)
        delta = pre_val - base_val
        lines.append(f"{_latex_escape(label)} & {base_val:.4f} & {pre_val:.4f} & {delta:.4f} \\\\")
    p_line = rf"p-value: {_fmt_stat(stats['p_value'])}\\"
    d_line = rf"Cohen's d: {_fmt_stat(stats['effect_size'])}\\"
    sig_line = rf"Significance: {_latex_escape(str(stats['significance']))}\\"
    interp_line = rf"Interpretation: {_latex_escape(str(stats['interpretation']))}\\"
    ci_base = stats.get("baseline_ci")
    ci_pre = stats.get("pretrained_ci")
    lines.extend(
        [
            r"\end{tabular}",
            r"\section*{Statistical Test (timesteps to convergence)}",
            p_line,
            d_line,
            sig_line,
            interp_line,
            rf"Baseline CI (95\%): {_latex_escape(str(ci_base))}\\",
            rf"Pre-trained CI (95\%): {_latex_escape(str(ci_pre))}\\",
            r"\section*{Hypothesis Evaluation}",
            rf"Decision: {_latex_escape(str(hypothesis_result.get('decision')))}\\",
        ]
    )
    if hypothesis_result.get("measured_value") is not None:
        lines.append(
            rf"Improvement: {hypothesis_result['measured_value']:.2f}\% "
            rf"(threshold $\geq$ {config.improvement_threshold_pct}\%)\\"
        )
    lines.append(rf"Note: {_latex_escape(str(hypothesis_result.get('note', '')))}\\")
    if figures:
        lines.append(r"\section*{Figures}")
        for label, path in figures.items():
            lines.append(rf"\subsection*{{{_latex_escape(label)}}}")
            lines.append(rf"\includegraphics[width=\linewidth]{{{path.name}}}")
    if stats.get("t_stat") is None:
        lines.append(r"\section*{Notes}")
        lines.append(
            r"Insufficient paired samples for statistical testing (need $\geq$ 2 per run)."
        )
    if config.hyperparameters:
        lines.append(r"\section*{Hyperparameters}")
        for key, value in sorted(config.hyperparameters.items()):
            lines.append(rf"{_latex_escape(key)}: {_latex_escape(str(value))}\\")
    lines.append(r"\end{document}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _figure_paths(summary_path: Path) -> dict[str, Path]:
    """Collect available analysis figures relative to summary path."""

    fig_dir = summary_path.parent / "figures"
    if not fig_dir.exists():
        return {}
    figures: dict[str, Path] = {}
    for candidate in ("timesteps_comparison.png", "performance_metrics.png"):
        path = fig_dir / candidate
        if path.exists():
            figures[path.stem] = path
    return figures


def generate_imitation_report(
    *,
    summary_path: Path,
    output_root: Path,
    config: ImitationReportConfig,
) -> dict[str, Path | None]:
    """Generate Markdown/optional LaTeX imitation report from a training summary."""

    summary = _load_summary(summary_path)
    baseline_rec, pretrained_rec = _select_records(
        summary, config.baseline_run_id, config.pretrained_run_id
    )
    figures = _figure_paths(summary_path)

    baseline_ts = _metric(baseline_rec, "timesteps_to_convergence")
    pretrained_ts = _metric(pretrained_rec, "timesteps_to_convergence")
    baseline_samples = _metric_samples(baseline_rec, "timesteps_to_convergence")
    pretrained_samples = _metric_samples(pretrained_rec, "timesteps_to_convergence")
    baseline_ci = _ci_from_samples(baseline_samples)
    pretrained_ci = _ci_from_samples(pretrained_samples)

    # Only run paired tests when we have >=2 paired samples
    if len(baseline_samples) == len(pretrained_samples) and len(baseline_samples) >= 2:
        t_res = paired_t_test(baseline_samples, pretrained_samples)
        effect = cohen_d(baseline_samples, pretrained_samples)
    else:
        t_res = {
            "t_stat": None,
            "p_value": None,
            "n": min(len(baseline_samples), len(pretrained_samples)),
        }
        effect = None
    stats = format_test_results(t_res, effect, alpha=config.alpha)
    stats["baseline_ci"] = baseline_ci
    stats["pretrained_ci"] = pretrained_ci

    improvement_baseline = baseline_samples or ([baseline_ts] if baseline_ts else [])
    improvement_pretrained = pretrained_samples or ([pretrained_ts] if pretrained_ts else [])
    hypothesis_result = evaluate_hypothesis(
        improvement_baseline, improvement_pretrained, threshold=config.improvement_threshold_pct
    )

    run_id = summary.get("run_id", "unknown")
    output_dir = output_root / f"{config.experiment_name}_{run_id}"
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Copy summary for traceability
    shutil.copy2(summary_path, data_dir / "summary.json")

    metadata = collect_reproducibility_metadata(
        seeds=[],
        config_paths=config.config_paths,
    )
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")

    report_md = _render_markdown(
        config=config,
        summary=summary,
        baseline=baseline_rec,
        pretrained=pretrained_rec,
        stats=stats,
        hypothesis_result=hypothesis_result,
        figures=figures,
        metadata_path=metadata_path,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")

    latex_path: Path | None = None
    if config.export_latex:
        latex_path = output_dir / "report.tex"
        _render_latex(
            config=config,
            summary=summary,
            baseline=baseline_rec,
            pretrained=pretrained_rec,
            stats=stats,
            hypothesis_result=hypothesis_result,
            figures=figures,
            metadata_path=metadata_path,
            output_path=latex_path,
        )

    return {
        "report": report_path,
        "metadata": metadata_path,
        "figures_dir": summary_path.parent / "figures",
        "data_dir": data_dir,
        "latex": latex_path,
    }
