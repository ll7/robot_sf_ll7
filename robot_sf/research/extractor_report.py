"""Generate research-ready reports from multi-extractor training summaries.

Consumes `summary.json` emitted by `scripts/multi_extractor_training.py` and
produces a structured report folder with Markdown/optional LaTeX, figures, and
reproducibility metadata.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.research.figures import configure_matplotlib_backend
from robot_sf.research.metadata import collect_reproducibility_metadata
from robot_sf.research.statistics import cohen_d, paired_t_test

configure_matplotlib_backend()


@dataclass
class ReportConfig:
    experiment_name: str
    hypothesis: str | None
    significance_level: float
    export_latex: bool
    baseline_extractor: str | None


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary.json must contain an object")
    return payload


def _maybe_copy_config(config_path: Path | None, target_dir: Path) -> Path | None:
    if config_path is None or not config_path.exists():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / config_path.name
    shutil.copy2(config_path, dest)
    return dest


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover
        return "unknown"


def _extract_metric(records: list[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for rec in records:
        metrics = rec.get("metrics") or {}
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            vals.append(float(val))
    return vals


def _generate_figures(records: list[dict[str, Any]], figures_dir: Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    rewards = _extract_metric(records, "best_mean_reward")
    names = [rec.get("config_name", f"extractor_{i}") for i, rec in enumerate(records)]
    if rewards:
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

    sample_eff = _extract_metric(records, "sample_efficiency_ratio")
    if sample_eff:
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
    run_id = summary.get("run_id", "unknown")
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
            f"- Candidates vs baseline (best_mean_reward): p={stats['reward_p_value']} cohen_d={stats['reward_cohen_d']}",
            f"- Candidates vs baseline (sample_efficiency_ratio): p={stats['sample_p_value']} cohen_d={stats['sample_cohen_d']}",
            "",
            "## Figures",
        ]
    )
    for label, path in figures.items():
        lines.append(f"![{label}]({path.name})")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Reports are generated from `summary.json` and enriched metrics.")
    return "\n".join(lines)


def _render_latex(report_md: str, figures: dict[str, Path], output_path: Path) -> None:
    body_lines = [r"\documentclass{article}", r"\usepackage{graphicx}", r"\begin{document}"]
    body_lines.append(r"\section*{Extractors}")
    body_lines.append(r"\begin{itemize}")
    for line in report_md.splitlines():
        if line.startswith("| Extractor"):
            continue
        if line.startswith("| ---"):
            continue
        if line.startswith("| "):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 5:
                body_lines.append(
                    rf"\item {cells[0]} -- status {cells[1]}, best reward {cells[2]}, convergence {cells[3]}, sample eff. {cells[4]}"
                )
    body_lines.append(r"\end{itemize}")
    if figures:
        body_lines.append(r"\section*{Figures}")
    for label, path in figures.items():
        body_lines.append(rf"\subsection*{{{label}}}")
        body_lines.append(rf"\includegraphics[width=\linewidth]{{{path.name}}}")
    body_lines.append(r"\end{document}")
    output_path.write_text("\n".join(body_lines), encoding="utf-8")


def _stats_against_baseline(
    baseline_vals: list[float],
    candidate_vals: list[float],
) -> tuple[float | None, float]:
    if len(baseline_vals) == len(candidate_vals) and len(baseline_vals) >= 2:
        p_val = paired_t_test(baseline_vals, candidate_vals)
    else:
        p_val = None
    effect = cohen_d(baseline_vals, candidate_vals)
    return p_val, effect


def generate_extractor_report(
    *,
    summary_path: Path,
    output_root: Path,
    config: ReportConfig,
    config_path: Path | None = None,
) -> dict[str, Path]:
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
        config_paths={"config": str(copied_config)} if copied_config else {},
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
        _render_latex(report_md, figures, latex_path)
    else:
        latex_path = None

    return {
        "report": report_path,
        "metadata": metadata_path,
        "figures_dir": figures_dir,
        "data_dir": data_dir,
        "latex": latex_path if latex_path else Path(),
    }
