# Quickstart: Automated Research Reporting

**Feature**: 270-imitation-report  
**Phase**: 1 (Design)  
**Date**: 2025-11-21

## Overview

This guide demonstrates how to use the automated research reporting system to transform imitation learning experiments into publication-ready artifacts.

---

## Prerequisites

```bash
# Ensure environment activated and dependencies installed
source .venv/bin/activate
uv sync --all-extras

# Verify key dependencies
python -c "import scipy, matplotlib, pandas; print('Dependencies OK')"
```

---

## Basic Usage: Single Experiment Report

### Step 1: Run Imitation Learning Pipeline

```bash
# Execute full pipeline with tracker enabled
uv run python examples/advanced/16_imitation_learning_pipeline.py \
  --policy-id expert_demo \
  --dataset-id traj_demo \
  --enable-tracker \
  --tracker-output my_experiment
```

This produces:
- Expert policy: `output/benchmarks/expert_policies/expert_demo.zip`
- Trajectories: `output/benchmarks/expert_trajectories/traj_demo.npz`
- BC policy: `output/models/expert/bc_expert_demo.zip`
- Fine-tuned policy: `output/benchmarks/expert_policies/finetuned_expert_demo.zip`
- Run tracker manifest: `output/run-tracker/my_experiment/manifest.json`

### Step 2: Generate Research Report

```bash
# Generate report from tracker manifest
uv run python scripts/research/generate_report.py \
  --tracker-run my_experiment \
  --experiment-name "BC Pretraining Demo" \
  --output output/research_reports/demo
```

**Output** (in `output/research_reports/demo/`):
```
demo/
├── report.md              # Main Markdown report
├── report.tex             # Optional LaTeX export
├── metadata.json          # Reproducibility metadata
├── figures/
│   ├── fig-learning-curve.pdf
│   ├── fig-learning-curve.png
│   ├── fig-sample-efficiency.pdf
│   ├── fig-sample-efficiency.png
│   └── ... (5+ figures)
├── data/
│   ├── metrics.json       # Structured aggregated metrics
│   ├── metrics.csv        # Flattened for Excel/pandas
│   └── hypothesis.json    # Evaluation results
└── configs/
    ├── expert_ppo.yaml
    ├── bc_pretrain.yaml
    └── ppo_finetune.yaml
```

### Step 3: View Report

```bash
# Open Markdown report
open output/research_reports/demo/report.md

# Or preview in VS Code
code output/research_reports/demo/report.md
```

---

## Advanced Usage: Ablation Studies

### Define Ablation Matrix

```python
# scripts/research/compare_ablations.py
from robot_sf.research.orchestrator import AblationOrchestrator

ablation_params = {
    "bc_epochs": [5, 10, 20],
    "dataset_size": [100, 200, 300]
}

orchestrator = AblationOrchestrator(
    experiment_name="BC_Ablation_Study",
    seeds=[42, 43, 44],
    ablation_params=ablation_params,
    hypothesis_threshold=40.0
)

# Run all 9 variants (3 BC epochs × 3 dataset sizes)
orchestrator.run_ablation_matrix()

# Generate comparison report
orchestrator.generate_report(output_dir="output/research_reports/ablation")
```

### Output Includes

- **Comparison table**: All 9 variants with improvement %, p-values, pass/fail
- **Sensitivity plots**: BC epochs vs improvement, dataset size vs improvement
- **Hypothesis summary**: Which variants meet ≥40% threshold

---

## Programmatic API Usage

### Manual Report Generation

```python
from robot_sf.research import (
    MetricAggregator,
    StatisticalAnalyzer,
    FigureGenerator,
    ReportRenderer
)
from pathlib import Path

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
fig_gen.distribution_comparison(metrics_baseline, metrics_pretrained, metric="success_rate")

# 6. Render report
renderer = ReportRenderer(template="default")
report_md = renderer.render(
    experiment_name="Manual Imitation Report",
    metrics={
        "baseline": metrics_baseline,
        "pretrained": metrics_pretrained
    },
    hypothesis=hypothesis_result,
    statistical_tests=comparison,
    figures=fig_gen.get_artifact_list()
)

# Save
output_path = Path("output/research_reports/manual/report.md")
output_path.write_text(report_md)
```

---

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

---

## Output Directory Structure

Every report follows this canonical structure:

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
│   ├── hypothesis.json    # Evaluation results
│   └── aggregated.csv     # Per-condition summary
└── configs/               # Captured configuration files
    ├── expert_ppo.yaml
    ├── bc_pretrain.yaml
    └── ppo_finetune.yaml
```

---

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

---

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

---

## Troubleshooting

### Issue: Missing tracker manifest

**Solution**: Ensure `--enable-tracker` flag used during pipeline execution.

```bash
# Check tracker output exists
ls output/run-tracker/<run_id>/manifest.json
```

### Issue: LaTeX export fails

**Solution**: Verify pandoc installed or skip LaTeX export.

```bash
# Check pandoc
which pandoc

# Or skip LaTeX
uv run python scripts/research/generate_report.py \
  --tracker-run <id> \
  --experiment-name <name> \
  # (omit --export-latex)
```

### Issue: Figures not generated

**Solution**: Confirm matplotlib backend supports PDF export.

```bash
# Test backend
python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.figure(); plt.savefig('/tmp/test.pdf'); print('OK')"
```

### Issue: Hypothesis shows INCOMPLETE

**Cause**: Missing baseline or pretrained metrics.

**Solution**: Verify both conditions ran successfully; check tracker manifest for failed steps.

```bash
# Inspect manifest
cat output/run-tracker/<run_id>/manifest.json | jq '.steps[] | select(.status != "completed")'
```

---

## Next Steps

- **Extend templates**: Customize `robot_sf/research/report_template.py` for domain-specific sections
- **Add metrics**: Implement new aggregators in `robot_sf/research/aggregation.py`
- **Custom figures**: Add plot functions to `robot_sf/research/figures.py`
- **Automate workflows**: Integrate `generate_report.py` into CI for regression tracking

---

## Related Documentation

- Full specification: `specs/270-imitation-report/spec.md`
- Data model: `specs/270-imitation-report/data-model.md`
- JSON schemas: `specs/270-imitation-report/contracts/`
- Imitation pipeline guide: `docs/imitation_learning_pipeline.md`
- Run tracker CLI: `scripts/tools/run_tracker_cli.py`
