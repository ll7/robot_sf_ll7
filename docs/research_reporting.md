# Research Reporting: Automated Analysis for Imitation Learning

**Feature**: 270-imitation-report  
**Status**: In Development  
**Purpose**: Transform multi-seed imitation learning experiments into publication-ready research reports

[← Back to Documentation Index](./README.md)

## Overview

The research reporting system automates the generation of comprehensive research reports from imitation learning pipeline runs. It orchestrates multi-seed experiments, aggregates metrics with statistical rigor, generates publication-quality figures, and exports structured reports (Markdown + LaTeX) with complete reproducibility metadata.

**Key Capabilities:**
- Multi-seed metric aggregation with bootstrap confidence intervals
- Statistical analysis (paired t-tests, effect sizes, hypothesis evaluation)
- Automated figure generation (learning curves, distributions, comparisons)
- Markdown and LaTeX report rendering
- Reproducibility metadata tracking (git hash, packages, hardware)
- Ablation study support (BC epochs, dataset sizes)

## Quick Start

### Basic Report Generation

```bash
# 1. Run imitation learning pipeline with tracker enabled
uv run python examples/advanced/16_imitation_learning_pipeline.py \
  --policy-id my_experiment \
  --dataset-id my_trajectories \
  --enable-tracker \
  --tracker-output my_run

# 2. Generate research report
uv run python scripts/research/generate_report.py \
  --tracker-run my_run \
  --experiment-name "BC Pretraining Demo" \
  --output output/research_reports/demo

# 3. View report
open output/research_reports/demo/report.md
```

### Report Output Structure

```
output/research_reports/<timestamp>_<experiment_name>/
├── report.md              # Primary Markdown report
├── report.tex             # Optional LaTeX export
├── metadata.json          # Reproducibility manifest
├── figures/               # All figures (PDF + PNG)
│   ├── fig-learning-curve.{pdf,png}
│   ├── fig-sample-efficiency.{pdf,png}
│   ├── fig-success-distribution.{pdf,png}
│   ├── fig-collision-distribution.{pdf,png}
│   └── fig-improvement-summary.{pdf,png}
├── data/                  # Raw and aggregated metrics
│   ├── metrics.json       # Structured (nested CIs, metadata)
│   ├── metrics.csv        # Flattened (Excel-friendly)
│   └── hypothesis.json    # Evaluation results
└── configs/               # Captured configuration files
    ├── expert_ppo.yaml
    ├── bc_pretrain.yaml
    └── ppo_finetune.yaml
```

## Programmatic API

### Example: Manual Report Generation

```python
from pathlib import Path
from robot_sf.research import (
    MetricAggregator,
    StatisticalAnalyzer,
    FigureGenerator,
    ReportRenderer,
)

# 1. Load episode data
episodes_baseline = read_jsonl("output/benchmarks/baseline_episodes.jsonl")
episodes_pretrained = read_jsonl("output/benchmarks/pretrained_episodes.jsonl")

# 2. Aggregate metrics
aggregator = MetricAggregator(bootstrap_samples=1000, confidence=0.95, seed=42)
metrics_baseline = aggregator.aggregate(episodes_baseline, group_by="seed")
metrics_pretrained = aggregator.aggregate(episodes_pretrained, group_by="seed")

# 3. Statistical analysis
analyzer = StatisticalAnalyzer()
comparison = analyzer.paired_t_test(
    baseline=metrics_baseline["timesteps_to_convergence"],
    treatment=metrics_pretrained["timesteps_to_convergence"]
)
effect_size = analyzer.cohen_d(metrics_baseline, metrics_pretrained)

# 4. Hypothesis evaluation
hypothesis_result = analyzer.evaluate_hypothesis(
    baseline_mean=metrics_baseline["mean"],
    treatment_mean=metrics_pretrained["mean"],
    threshold=40.0,
    metric_name="timesteps_to_convergence"
)

# 5. Generate figures
fig_gen = FigureGenerator(output_dir=Path("output/research_reports/manual/figures"))
fig_gen.learning_curve(episodes_baseline, episodes_pretrained, seeds=[42,43,44])
fig_gen.sample_efficiency_bar(metrics_baseline, metrics_pretrained)

# 6. Render report
renderer = ReportRenderer(template="default")
report_md = renderer.render(
    experiment_name="Manual Report",
    metrics={"baseline": metrics_baseline, "pretrained": metrics_pretrained},
    hypothesis=hypothesis_result,
    statistical_tests=comparison,
    figures=fig_gen.get_artifact_list()
)

# Save
(Path("output/research_reports/manual") / "report.md").write_text(report_md)
```

## CLI Reference

### generate_report.py

```bash
uv run python scripts/research/generate_report.py [OPTIONS]

Options:
  --tracker-run TEXT          Run tracker ID (from output/run-tracker/)
  --experiment-name TEXT      Human-readable experiment label
  --output PATH               Report output directory
  --hypothesis-threshold FLOAT Improvement threshold % (default: 40.0)
  --bootstrap-samples INT     Bootstrap iterations (default: 1000)
  --confidence FLOAT          CI confidence level (default: 0.95)
  --export-latex              Also generate LaTeX report
  --skip-figures              Skip figure generation (metadata only)
  --help                      Show help message
```

### compare_ablations.py

```bash
uv run python scripts/research/compare_ablations.py [OPTIONS]

Options:
  --ablation-config PATH      YAML file defining ablation matrix
  --experiment-name TEXT      Experiment identifier
  --seeds INT...              Random seeds (space-separated)
  --output PATH               Report output directory
  --parallel-workers INT      Parallel execution workers (default: 1)
  --help                      Show help message
```

## Report Sections

Generated `report.md` includes:

1. **Abstract**: Auto-populated with hypothesis, primary findings, quantified improvement
2. **Experimental Setup**: Seeds, configs, hardware profile, software versions
3. **Results**: 
   - Learning curves (timesteps vs success rate)
   - Sample efficiency comparison (bar chart + table)
   - Success/collision rate distributions
4. **Statistical Analysis**:
   - Paired t-test results (p-value, effect size)
   - Bootstrap confidence intervals
   - Hypothesis evaluation (PASS/FAIL with measured improvement)
5. **Ablation Results** (if applicable):
   - Comparison table across all variants
   - Sensitivity plots
6. **Conclusions**: Summary of key findings
7. **Reproducibility**: Git hash, packages, hardware, commands to reproduce

## Architecture

### Module Structure

```
robot_sf/research/
├── __init__.py              # Public API exports
├── aggregation.py           # MetricAggregator: multi-seed aggregation + bootstrap CIs
├── statistics.py            # StatisticalAnalyzer: t-tests, effect sizes, hypothesis eval
├── figures.py               # FigureGenerator: matplotlib plots (PDF + PNG)
├── report_template.py       # ReportRenderer: Markdown/LaTeX rendering
├── orchestrator.py          # ReportOrchestrator: end-to-end coordination
├── metadata.py              # ReproducibilityMetadata: git/hardware tracking
├── artifact_paths.py        # Path management for output/research_reports/
├── schema_loader.py         # JSON schema validation
├── logging_config.py        # Loguru logging setup
└── exceptions.py            # Custom exceptions
```

### Data Flow

```
Pipeline Runs (tracker manifests)
        ↓
MetricAggregator (per-seed → aggregated stats)
        ↓
StatisticalAnalyzer (t-tests, effect sizes, hypothesis)
        ↓
FigureGenerator (learning curves, distributions, comparisons)
        ↓
ReportRenderer (Markdown/LaTeX from templates)
        ↓
Output (report.md, figures/, data/, metadata.json)
```

## Validation

```bash
# Validate report artifacts
uv run python scripts/tools/validate_report.py \
  output/research_reports/demo

# Checks:
# ✓ All required files present
# ✓ JSON schemas validate
# ✓ Figures readable as PDF/PNG
# ✓ Metadata completeness ≥95%
# ✓ CSV parseable by pandas
```

## Configuration

### Hypothesis Threshold

Default: 40% reduction in PPO timesteps to convergence

Override via CLI:
```bash
--hypothesis-threshold 50.0  # Require 50% improvement
```

### Bootstrap Confidence Intervals

Default: 1000 samples, 95% confidence

Override via CLI:
```bash
--bootstrap-samples 5000 --confidence 0.99
```

### Figure Styles

Figures follow `docs/dev_guide.md` guidelines:
- Vector PDF + PNG (300 dpi) dual export
- LaTeX-compatible fonts (matplotlib PGF backend)
- 9pt labels, 8pt ticks/legend
- Publication-ready captions with sample size notation

## Troubleshooting

### Issue: Missing tracker manifest

**Solution**: Ensure `--enable-tracker` flag used during pipeline execution.

```bash
ls output/run-tracker/<run_id>/manifest.json
```

### Issue: LaTeX export fails

**Solution**: Verify pandoc installed or skip LaTeX export.

```bash
which pandoc
# Or skip LaTeX
--no-export-latex
```

### Issue: Figures not generated

**Solution**: Confirm matplotlib backend supports PDF export.

```bash
python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.figure(); plt.savefig('/tmp/test.pdf'); print('OK')"
```

### Issue: Hypothesis shows INCOMPLETE

**Cause**: Missing baseline or pretrained metrics.

**Solution**: Verify both conditions ran successfully; check tracker manifest for failed steps.

```bash
cat output/run-tracker/<run_id>/manifest.json | jq '.steps[] | select(.status != "completed")'
```

## Performance

**Target**: < 120 seconds for 3-seed run (Success Criterion SC-001)

Bottlenecks:
- Bootstrap sampling (1000 iterations)
- Figure generation (5-10 plots)
- LaTeX export (if enabled)

Optimization:
- Reduce bootstrap samples for faster iteration: `--bootstrap-samples 100`
- Skip figures during development: `--skip-figures`
- Disable LaTeX export: (omit `--export-latex`)

## Related Documentation

- **[Imitation Learning Pipeline](./imitation_learning_pipeline.md)** - Complete guide to PPO pre-training
- **[Imitation Learning Quickstart](../specs/001-ppo-imitation-pretrain/quickstart.md)** - Step-by-step workflow
- **[Run Tracker CLI](./dev_guide.md#run-tracker--history-cli)** - Tracker manifest commands
- **[Development Guide](./dev_guide.md)** - Coding standards, testing, quality gates
- **[Specification](../specs/270-imitation-report/spec.md)** - Complete feature specification
- **[Quickstart Guide](../specs/270-imitation-report/quickstart.md)** - Usage examples and workflows

## Contributing

When extending the research reporting system:

1. Follow existing module patterns (see `robot_sf/research/`)
2. Add comprehensive docstrings with examples
3. Write unit tests in `tests/research/`
4. Update this documentation
5. Ensure new figures follow publication guidelines

## License

GPL-3.0-only (same as Robot SF)
