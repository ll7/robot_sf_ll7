"""
Report template renderer for research reporting (User Story 1)
Implements: MarkdownReportRenderer, export_latex, _render_abstract
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class MarkdownReportRenderer:
    """Renders research report as Markdown with optional LaTeX export."""

    def __init__(self, output_dir: Path):
        """Init.

        Args:
            output_dir: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        experiment_name: str,
        hypothesis_result: dict[str, Any],
        aggregated_metrics: list[dict[str, Any]],
        figures: list[dict[str, Any]],
        metadata: dict[str, Any],
        seed_status: Optional[list[dict[str, Any]]] = None,
        completeness: Optional[dict[str, Any]] = None,
        ablation_variants: Optional[list[dict[str, Any]]] = None,
        telemetry: Optional[dict[str, Any]] = None,
    ) -> Path:
        """
        Render full Markdown report.
        Returns path to report.md file.
        """
        sections = []

        # Title and metadata
        sections.append(f"# {experiment_name}\n")
        sections.append(f"**Generated**: {datetime.now().isoformat()}\n")
        sections.append(f"**Run ID**: {metadata.get('run_id', 'N/A')}\n")

        # Abstract
        sections.append(self._render_abstract(hypothesis_result, aggregated_metrics))

        # Hypothesis Evaluation
        sections.append(self._render_hypothesis_section(hypothesis_result))

        # Ablation table (optional)
        if ablation_variants:
            sections.append(self._render_ablation_table(ablation_variants))

        # Seed summary/completeness
        seed_summary_section = self._render_seed_summary(seed_status, completeness)
        if seed_summary_section:
            sections.append(seed_summary_section)

        # Telemetry (Phase 7)
        if telemetry:
            sections.append(self._render_telemetry(telemetry))

        # Results
        sections.append(self._render_results_section(aggregated_metrics))

        # Figures
        sections.append(self._render_figures_section(figures))

        # Reproducibility
        sections.append(self._render_reproducibility_section(metadata))

        # Write to file
        report_path = self.output_dir / "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sections))

        return report_path

    def _render_abstract(
        self, hypothesis_result: dict[str, Any], aggregated_metrics: list[dict[str, Any]]
    ) -> str:
        """Auto-populate Abstract section based on hypothesis and key metrics."""
        decision = hypothesis_result.get("decision", "INCOMPLETE")
        measured = hypothesis_result.get("measured_value")
        threshold = hypothesis_result.get("threshold_value", 40.0)

        # Extract key metrics
        baseline_success = next(
            (
                m["mean"]
                for m in aggregated_metrics
                if m["condition"] == "baseline" and m["metric_name"] == "success_rate"
            ),
            None,
        )
        pretrained_success = next(
            (
                m["mean"]
                for m in aggregated_metrics
                if m["condition"] == "pretrained" and m["metric_name"] == "success_rate"
            ),
            None,
        )

        abstract = "## Abstract\n\n"
        if decision == "PASS":
            abstract += f"This experiment demonstrates that behavioral cloning pre-training significantly improves PPO sample efficiency, achieving a **{measured:.1f}% reduction** in timesteps to convergence (threshold: {threshold}%). "
        elif decision == "FAIL":
            abstract += f"This experiment found that behavioral cloning pre-training did not meet the target sample efficiency improvement (achieved: {measured:.1f}%, threshold: {threshold}%). "
        else:
            abstract += "This experiment evaluated the impact of behavioral cloning pre-training on PPO sample efficiency. Results are incomplete. "

        if baseline_success is not None and pretrained_success is not None:
            abstract += f"Success rates: baseline {baseline_success:.2%}, pretrained {pretrained_success:.2%}."

        return abstract + "\n"

    def _render_hypothesis_section(self, hypothesis_result: dict[str, Any]) -> str:
        """Render hypothesis evaluation results."""
        section = "## Hypothesis Evaluation\n\n"
        section += f"**Hypothesis**: {hypothesis_result.get('description', 'N/A')}\n\n"
        section += f"**Decision**: **{hypothesis_result.get('decision', 'INCOMPLETE')}**\n\n"

        measured = hypothesis_result.get("measured_value")
        if measured is not None:
            section += f"**Measured Improvement**: {measured:.1f}%\n\n"

        note = hypothesis_result.get("note")
        if note:
            section += f"*{note}*\n"

        return section + "\n"

    def _render_seed_summary(
        self,
        seed_status: Optional[list[dict[str, Any]]],
        completeness: Optional[dict[str, Any]],
    ) -> str:
        """Render seed completeness table.

        Returns empty string when no seed data is provided to avoid spurious
        sections in reports that do not orchestrate multiple seeds.
        """

        if not seed_status and not completeness:
            return ""

        section = "## Seed Summary\n\n"
        if completeness:
            section += (
                f"**Completeness**: {completeness.get('score', 0)}% "
                f"({completeness.get('completed', 0)}/"
                f"{completeness.get('expected', 0)} seeds)\n\n"
            )
            missing = completeness.get("missing_seeds") or []
            failed = completeness.get("failed_seeds") or []
            if missing:
                section += f"Missing seeds: {', '.join(map(str, missing))}\n\n"
            if failed:
                section += f"Failed seeds: {', '.join(map(str, failed))}\n\n"

        if seed_status:
            section += "| Seed | Baseline | Pretrained | Notes |\n"
            section += "|------|----------|------------|-------|\n"
            for entry in seed_status:
                notes = entry.get("note") or "-"
                section += (
                    f"| {entry.get('seed', 'N/A')} | "
                    f"{entry.get('baseline_status', 'N/A')} | "
                    f"{entry.get('pretrained_status', 'N/A')} | {notes} |\n"
                )

        return section + "\n"

    def _render_results_section(self, aggregated_metrics: list[dict[str, Any]]) -> str:
        """Render aggregated metrics as table."""
        section = "## Results\n\n"
        # Optional stats table if effect sizes present
        stats_table = self._render_stats_table(aggregated_metrics)
        if stats_table:
            section += stats_table + "\n"
        section += "### Aggregated Metrics\n\n"
        section += "| Condition | Metric | Mean | Median | P95 | CI (95%) |\n"
        section += "|-----------|--------|------|--------|-----|----------|\n"

        for m in aggregated_metrics:
            condition = m["condition"]
            metric_name = m["metric_name"].replace("_", " ").title()
            mean = m["mean"]
            median = m["median"]
            p95 = m["p95"]
            ci_low = m.get("ci_low")
            ci_high = m.get("ci_high")
            ci_str = (
                f"[{ci_low:.2f}, {ci_high:.2f}]"
                if ci_low is not None and ci_high is not None
                else "N/A"
            )

            section += f"| {condition} | {metric_name} | {mean:.2f} | {median:.2f} | {p95:.2f} | {ci_str} |\n"

        return section + "\n"

    def _render_stats_table(self, aggregated_metrics: list[dict[str, Any]]) -> str:
        """Render statistical summary table if any entries contain effect_size.

        Expects aggregated_metrics entries possibly containing keys:
        - metric_name
        - condition
        - effect_size (optional)
        - mean, ci_low, ci_high (optional)
        Returns empty string if no effect sizes present.
        """
        effect_entries = [m for m in aggregated_metrics if m.get("effect_size") is not None]
        if not effect_entries:
            return ""
        table = "### Statistical Summary (Effect Sizes)\n\n"
        table += "| Metric | Condition | Effect Size (d) | Interpretation |\n"
        table += "|--------|-----------|-----------------|----------------|\n"
        for m in effect_entries:
            metric = m.get("metric_name", "N/A").replace("_", " ").title()
            cond = m.get("condition", "N/A")
            d = m.get("effect_size")
            # Interpret Cohen's d using established thresholds (Cohen 1988)
            # Label assignment follows standard effect size conventions
            if d is None:
                label = "N/A"
            else:
                ad = abs(d)
                if ad < 0.2:
                    label = "negligible"
                elif ad < 0.5:
                    label = "small"
                elif ad < 0.8:
                    label = "medium"
                else:
                    label = "large"
            table += (
                f"| {metric} | {cond} | {d:.2f} | {label} |\n"
                if d is not None
                else f"| {metric} | {cond} | N/A | N/A |\n"
            )
        return table

    def _render_figures_section(self, figures: list[dict[str, Any]]) -> str:
        """Render figures section with captions and links."""
        section = "## Figures\n\n"
        for fig in figures:
            caption = fig.get("caption", "Figure")
            pdf_path = fig["paths"].get("pdf")
            png_path = fig["paths"].get("png")

            if png_path:
                png = Path(png_path)
                pdf = Path(pdf_path) if pdf_path else None

                def _rel_safe(path: Path) -> str:
                    """Rel safe.

                    Args:
                        path: Auto-generated placeholder description.

                    Returns:
                        str: Auto-generated placeholder description.
                    """
                    if not path.is_absolute():
                        return path.as_posix()
                    try:
                        return path.relative_to(self.output_dir).as_posix()
                    except ValueError:
                        # Fallback to basename if outside output_dir
                        return path.name

                rel_png = _rel_safe(png)
                rel_pdf = _rel_safe(pdf) if pdf else None

                section += f"![{caption}]({rel_png})\n\n"
                section += f"*{caption}*\n\n"
                if rel_pdf:
                    section += f"[PDF version]({rel_pdf})\n\n"

        return section

    def _render_reproducibility_section(self, metadata: dict[str, Any]) -> str:
        """Render reproducibility metadata."""
        section = "## Reproducibility\n\n"
        repro = metadata.get("reproducibility", {})

        section += f"- **Git Commit**: `{repro.get('git_commit', 'N/A')}`\n"
        section += f"- **Git Branch**: `{repro.get('git_branch', 'N/A')}`\n"
        section += f"- **Git Dirty**: {repro.get('git_dirty', False)}\n"
        section += f"- **Python Version**: {repro.get('python_version', 'N/A')}\n"

        packages = repro.get("key_packages", {})
        if packages:
            section += "\n### Key Packages\n\n"
            for pkg, version in packages.items():
                section += f"- `{pkg}`: {version}\n"

        hardware = repro.get("hardware", {})
        if hardware:
            section += "\n### Hardware\n\n"
            section += f"- **CPU**: {hardware.get('cpu_model', 'N/A')} ({hardware.get('cpu_cores', 'N/A')} cores)\n"
            section += f"- **Memory**: {hardware.get('memory_gb', 'N/A')} GB\n"
            gpu_model = hardware.get("gpu_model")
            if gpu_model:
                section += (
                    f"- **GPU**: {gpu_model} ({hardware.get('gpu_memory_gb', 'N/A')} GB VRAM)\n"
                )

        return section + "\n"

    def _render_ablation_table(self, variants: list[dict[str, Any]]) -> str:
        """Render ablation comparison table.

        Expected variant dict keys: variant_id, improvement_pct, decision (PASS/FAIL/INCOMPLETE),
        bc_epochs (optional), dataset_size (optional).
        """
        if not variants:
            return ""
        section = "## Ablation Matrix\n\n"
        section += "| Variant | BC Epochs | Dataset Size | Improvement (%) | Decision |\n"
        section += "|---------|-----------|--------------|------------------|----------|\n"
        for v in variants:
            imp = v.get("improvement_pct")
            imp_str = f"{imp:.1f}" if isinstance(imp, int | float) else "N/A"
            section += (
                f"| {v.get('variant_id', 'N/A')} | {v.get('bc_epochs', '-')} | {v.get('dataset_size', '-')} | "
                f"{imp_str} | {v.get('decision', 'N/A')} |\n"
            )
        return section + "\n"

    def _render_telemetry(self, telemetry: dict[str, Any]) -> str:
        """Render telemetry metrics collected during runs (Phase 7 T078).

        Expects a flat dict of key→value pairs (already aggregated). Missing
        telemetry data returns an empty section upstream.
        """
        if not telemetry:
            return ""
        section = "## Telemetry\n\n"
        section += "| Metric | Value |\n|--------|-------|\n"
        for k, v in telemetry.items():
            section += f"| {k} | {v} |\n"
        return section + "\n"

    def export_latex(self, markdown_path: Path) -> Optional[Path]:
        """
        Export Markdown report to LaTeX format.
        Returns path to .tex file, or None if conversion fails.
        """
        # Placeholder: would use pandoc or manual conversion
        # For now, create a simple LaTeX wrapper
        latex_path = markdown_path.with_suffix(".tex")

        with open(markdown_path, encoding="utf-8") as f:
            md_content = f.read()

        # Simple conversion (real implementation would use pandoc)
        latex_content = "\\documentclass{article}\n"
        latex_content += "\\usepackage{graphicx}\n"
        latex_content += "\\begin{document}\n"
        latex_content += "% Converted from Markdown\n"
        latex_content += "% Note: Manual cleanup required for publication\n\n"
        latex_content += md_content.replace("#", "\\section{").replace("\n\n", "}\n\n")
        latex_content += "\\end{document}\n"

        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex_content)

        return latex_path
