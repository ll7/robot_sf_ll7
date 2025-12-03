"""Generate research-ready reports from multi-extractor training summaries.

Consumes `summary.json` emitted by `scripts/multi_extractor_training.py` and
produces a structured report folder with Markdown/optional LaTeX, figures, and
reproducibility metadata.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.research.figures import configure_matplotlib_backend
from robot_sf.research.metadata import collect_reproducibility_metadata
from robot_sf.research.statistics import cohen_d_independent, welch_t_test

configure_matplotlib_backend()


def _get_pyplot():
    """TODO docstring. Document this function.

    Returns:
        matplotlib.pyplot module.
    """
    return importlib.import_module("matplotlib.pyplot")


def _latex_escape(text: str) -> str:
    """TODO docstring. Document this function.

    Args:
        text: TODO docstring.

    Returns:
        TODO docstring.
    """
    return (
        text.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


@dataclass
class ReportConfig:
    """TODO docstring. Document this class."""

    experiment_name: str
    hypothesis: str | None
    significance_level: float
    export_latex: bool
    baseline_extractor: str | None


def _timestamp() -> str:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _load_summary(path: Path) -> dict[str, Any]:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.

    Returns:
        TODO docstring.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary.json must contain an object")
    return payload


def _maybe_copy_config(config_path: Path | None, target_dir: Path) -> Path | None:
    """TODO docstring. Document this function.

    Args:
        config_path: TODO docstring.
        target_dir: TODO docstring.

    Returns:
        TODO docstring.
    """
    if config_path is None or not config_path.exists():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / config_path.name
    shutil.copy2(config_path, dest)
    return dest


def _git_hash() -> str:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        OSError,
    ):  # pragma: no cover
        return "unknown"


def _extract_metric(records: list[dict[str, Any]], key: str) -> list[float]:
    """TODO docstring. Document this function.

    Args:
        records: TODO docstring.
        key: TODO docstring.

    Returns:
        TODO docstring.
    """
    vals: list[float] = []
    for rec in records:
        metrics = rec.get("metrics") or {}
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            vals.append(float(val))
    return vals


def _generate_figures(records: list[dict[str, Any]], figures_dir: Path) -> dict[str, Path]:
    """TODO docstring. Document this function.

    Args:
        records: TODO docstring.
        figures_dir: TODO docstring.

    Returns:
        TODO docstring.
    """
    plt = _get_pyplot()

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _collect_named_metric(metric_key: str) -> list[tuple[str, float]]:
        """TODO docstring. Document this function.

        Args:
            metric_key: TODO docstring.

        Returns:
            TODO docstring.
        """
        aligned: list[tuple[str, float]] = []
        for idx, rec in enumerate(records):
            metrics = rec.get("metrics") or {}
            raw_val = metrics.get(metric_key)
            if isinstance(raw_val, (int, float)):
                name = rec.get("config_name", f"extractor_{idx}")
                aligned.append((name, float(raw_val)))
        return aligned

    reward_pairs = _collect_named_metric("best_mean_reward")
    if reward_pairs:
        names, rewards = zip(*reward_pairs, strict=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, rewards, color="#4c78a8")
        ax.set_ylabel("Best mean reward")
        ax.set_title("Final performance")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = figures_dir / "final_performance.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["final_performance"] = path

    sample_pairs = _collect_named_metric("sample_efficiency_ratio")
    if sample_pairs:
        names, sample_eff = zip(*sample_pairs, strict=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, sample_eff, color="#f58518")
        ax.set_ylabel("Sample efficiency ratio (baseline/candidate)")
        ax.set_title("Sample efficiency")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = figures_dir / "sample_efficiency.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["sample_efficiency"] = path

    return paths


def _render_markdown(
    *,
    summary: dict[str, Any],
    config: ReportConfig,
    stats: dict[str, Any],
    figures: dict[str, Path],
    metadata_path: Path,
) -> str:
    """TODO docstring. Document this function.

    Args:
        summary: TODO docstring.
        config: TODO docstring.
        stats: TODO docstring.
        figures: TODO docstring.
        metadata_path: TODO docstring.

    Returns:
        TODO docstring.
    """
    run_id = summary.get("run_id", "unknown")
    output_dir = metadata_path.parent

    def _fmt_stat(value: float | None) -> str:
        """TODO docstring. Document this function.

        Args:
            value: TODO docstring.

        Returns:
            TODO docstring.
        """
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        f"# {config.experiment_name} Report",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated: {_timestamp()}",
        f"- Git: `{_git_hash()}`",
        f"- Hypothesis: {config.hypothesis or 'N/A'}",
        f"- Significance level: {config.significance_level}",
        f"- Metadata: `{metadata_path.name}`",
        "",
        "## Extractor Summary",
    ]
    lines.append("| Extractor | Status | Best Reward | Convergence (ts) | Sample Eff. Ratio |")
    lines.append("| --- | --- | --- | --- | --- |")
    for rec in summary.get("extractor_results", []):
        name = rec.get("config_name", "unknown")
        status = rec.get("status", "unknown")
        metrics = rec.get("metrics") or {}
        best_reward = metrics.get("best_mean_reward", "n/a")
        conv = metrics.get("convergence_timestep", "n/a")
        ratio = metrics.get("sample_efficiency_ratio", "n/a")
        lines.append(f"| {name} | {status} | {best_reward} | {conv} | {ratio} |")

    lines.extend(
        [
            "",
            "## Statistical Comparison",
            f"- Baseline: `{stats.get('baseline')}`",
            "- Candidates vs baseline (best_mean_reward): "
            f"p={_fmt_stat(stats['reward_p_value'])} cohen_d={_fmt_stat(stats['reward_cohen_d'])}",
            "- Candidates vs baseline (sample_efficiency_ratio): "
            f"p={_fmt_stat(stats['sample_p_value'])} cohen_d={_fmt_stat(stats['sample_cohen_d'])}",
            "",
            "## Figures",
        ]
    )
    for label, path in figures.items():
        try:
            figure_rel = path.relative_to(output_dir)
        except ValueError:  # fallback if a figure is outside the report directory
            figure_rel = path
        lines.append(f"![{label}]({figure_rel.as_posix()})")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Reports are generated from `summary.json` and enriched metrics.")
    return "\n".join(lines)


def _render_latex(
    *,
    summary: dict[str, Any],
    config: ReportConfig,
    stats: dict[str, Any],
    figures: dict[str, Path],
    metadata_path: Path,
    output_path: Path,
) -> None:
    """TODO docstring. Document this function.

    Args:
        summary: TODO docstring.
        config: TODO docstring.
        stats: TODO docstring.
        figures: TODO docstring.
        metadata_path: TODO docstring.
        output_path: TODO docstring.
    """

    def _fmt_stat(value: float | None) -> str:
        """TODO docstring. Document this function.

        Args:
            value: TODO docstring.

        Returns:
            TODO docstring.
        """
        return "n/a" if value is None else f"{value:.4f}"

    run_id = summary.get("run_id", "unknown")
    output_dir = metadata_path.parent
    body_lines = [
        r"\documentclass{article}",
        r"\usepackage{graphicx}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        rf"\section*{{{_latex_escape(config.experiment_name)} Report}}",
        r"\begin{itemize}",
        rf"\item Run ID: \texttt{{{_latex_escape(str(run_id))}}}",
        rf"\item Generated: {_timestamp()}",
        rf"\item Git: \texttt{{{_latex_escape(_git_hash())}}}",
        rf"\item Hypothesis: {_latex_escape(config.hypothesis or 'N/A')}",
        rf"\item Significance level: {config.significance_level}",
        rf"\item Metadata: \texttt{{{_latex_escape(metadata_path.name)}}}",
        r"\end{itemize}",
        r"\section*{Extractors}",
        r"\begin{tabular}{lllll}",
        r"\textbf{Extractor} & \textbf{Status} & \textbf{Best Reward} & \textbf{Convergence (ts)} & \textbf{Sample Eff. Ratio} \\",
        r"\hline",
    ]

    for rec in summary.get("extractor_results", []):
        name = _latex_escape(str(rec.get("config_name", "unknown")))
        status = _latex_escape(str(rec.get("status", "unknown")))
        metrics = rec.get("metrics") or {}
        best_reward = metrics.get("best_mean_reward", "n/a")
        conv = metrics.get("convergence_timestep", "n/a")
        ratio = metrics.get("sample_efficiency_ratio", "n/a")
        body_lines.append(f"{name} & {status} & {best_reward} & {conv} & {ratio} \\\\")

    reward_stats = (
        rf"best\_mean\_reward: p={_fmt_stat(stats['reward_p_value'])} "
        rf"\; d={_fmt_stat(stats['reward_cohen_d'])}\\"
    )
    sample_stats = (
        rf"sample\_efficiency\_ratio: p={_fmt_stat(stats['sample_p_value'])} "
        rf"\; d={_fmt_stat(stats['sample_cohen_d'])}\\"
    )

    body_lines.extend(
        [
            r"\end{tabular}",
            r"\section*{Statistical Comparison}",
            rf"\textbf{{Baseline}}: \texttt{{{_latex_escape(str(stats.get('baseline')))}}}\\",
            reward_stats,
            sample_stats,
        ]
    )

    if figures:
        body_lines.append(r"\section*{Figures}")
        for label, path in figures.items():
            try:
                figure_rel = path.relative_to(output_dir)
            except ValueError:
                figure_rel = path
            body_lines.append(rf"\subsection*{{{_latex_escape(label)}}}")
            body_lines.append(rf"\includegraphics[width=\linewidth]{{{figure_rel.as_posix()}}}")
    body_lines.append(r"\end{document}")
    output_path.write_text("\n".join(body_lines), encoding="utf-8")


def _stats_against_baseline(
    baseline_vals: list[float],
    candidate_vals: list[float],
) -> tuple[float | None, float | None]:
    """TODO docstring. Document this function.

    Args:
        baseline_vals: TODO docstring.
        candidate_vals: TODO docstring.

    Returns:
        TODO docstring.
    """
    test_result = welch_t_test(baseline_vals, candidate_vals)
    effect = cohen_d_independent(baseline_vals, candidate_vals)
    return test_result.get("p_value"), effect


def generate_extractor_report(
    *,
    summary_path: Path,
    output_root: Path,
    config: ReportConfig,
    config_path: Path | None = None,
) -> dict[str, Path | None]:
    """TODO docstring. Document this function.

    Args:
        summary_path: TODO docstring.
        output_root: TODO docstring.
        config: TODO docstring.
        config_path: TODO docstring.

    Returns:
        TODO docstring.
    """
    summary = _load_summary(summary_path)
    run_id = summary.get("run_id", "unknown")
    output_dir = output_root / f"{config.experiment_name}_{run_id}"
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    configs_dir = output_dir / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(summary_path, data_dir / "summary.json")
    copied_config = _maybe_copy_config(config_path, configs_dir)

    metadata = collect_reproducibility_metadata(
        seeds=[],
        config_paths={"config": copied_config} if copied_config else None,
    )
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")

    figures = _generate_figures(summary.get("extractor_results", []), figures_dir)

    baseline_name = config.baseline_extractor or (
        summary.get("extractor_results", [{}])[0].get("config_name")
        if summary.get("extractor_results")
        else "baseline"
    )
    baseline_rewards = _extract_metric(
        [r for r in summary.get("extractor_results", []) if r.get("config_name") == baseline_name],
        "best_mean_reward",
    )
    candidate_rewards = _extract_metric(
        [r for r in summary.get("extractor_results", []) if r.get("config_name") != baseline_name],
        "best_mean_reward",
    )
    baseline_sample = _extract_metric(
        [r for r in summary.get("extractor_results", []) if r.get("config_name") == baseline_name],
        "sample_efficiency_ratio",
    )
    candidate_sample = _extract_metric(
        [r for r in summary.get("extractor_results", []) if r.get("config_name") != baseline_name],
        "sample_efficiency_ratio",
    )

    reward_p, reward_d = _stats_against_baseline(baseline_rewards, candidate_rewards)
    sample_p, sample_d = _stats_against_baseline(baseline_sample, candidate_sample)

    stats_payload = {
        "baseline": baseline_name,
        "reward_p_value": reward_p,
        "reward_cohen_d": reward_d,
        "sample_p_value": sample_p,
        "sample_cohen_d": sample_d,
    }

    report_md = _render_markdown(
        summary=summary,
        config=config,
        stats=stats_payload,
        figures=figures,
        metadata_path=metadata_path,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")

    if config.export_latex:
        latex_path = output_dir / "report.tex"
        _render_latex(
            summary=summary,
            config=config,
            stats=stats_payload,
            figures=figures,
            metadata_path=metadata_path,
            output_path=latex_path,
        )
    else:
        latex_path = None

    return {
        "report": report_path,
        "metadata": metadata_path,
        "figures_dir": figures_dir,
        "data_dir": data_dir,
        "latex": latex_path,
    }
