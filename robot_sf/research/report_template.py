"""
Report template renderer for research reporting (User Story 1)
Implements: MarkdownReportRenderer, export_latex, _render_abstract
"""

from datetime import datetime
from pathlib import Path
from typing import Any


class MarkdownReportRenderer:
    """Renders research report as Markdown with optional LaTeX export."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        experiment_name: str,
        hypothesis_result: dict[str, Any],
        aggregated_metrics: list[dict[str, Any]],
        figures: list[dict[str, Any]],
        metadata: dict[str, Any],
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

    def _render_results_section(self, aggregated_metrics: list[dict[str, Any]]) -> str:
        """Render aggregated metrics as table."""
        section = "## Results\n\n"
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

    def _render_figures_section(self, figures: list[dict[str, Any]]) -> str:
        """Render figures section with captions and links."""
        section = "## Figures\n\n"
        for fig in figures:
            caption = fig.get("caption", "Figure")
            pdf_path = fig["paths"].get("pdf")
            png_path = fig["paths"].get("png")

            if png_path:
                # Relative path for markdown embedding
                rel_path = Path(png_path).relative_to(self.output_dir)
                section += f"![{caption}]({rel_path})\n\n"
                section += f"*{caption}*\n\n"
                section += f"[PDF version]({Path(pdf_path).relative_to(self.output_dir)})\n\n"

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

    def export_latex(self, markdown_path: Path) -> Path | None:
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
