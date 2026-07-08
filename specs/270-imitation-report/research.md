# Research & Technical Decisions: Automated Research Reporting

**Feature**: 270-imitation-report  
**Phase**: 0 (Research)  
**Date**: 2025-11-21

## Overview

This document consolidates research findings and technical decisions for the automated research reporting system. All "NEEDS CLARIFICATION" items from the Technical Context have been resolved through analysis of existing codebase patterns, library capabilities, and domain requirements.

---

## Statistical Testing Library Selection

**Decision**: Use **scipy.stats** for paired t-tests and effect size calculations

**Rationale**:
- Already available in scientific Python stack (likely transitive dependency via NumPy/matplotlib)
- `scipy.stats.ttest_rel()` provides paired t-test with automatic p-value computation
- `scipy.stats.cohen_d()` or manual calculation `(mean_diff / pooled_std)` for effect sizes
- Well-documented, stable API with extensive validation in research contexts
- Integrates cleanly with NumPy arrays from aggregated metrics

**Alternatives Considered**:
- **statsmodels**: More comprehensive but heavier dependency; overkill for basic t-tests
- **pingouin**: Research-focused but adds new dependency; scipy sufficient for our needs
- **Manual implementation**: Error-prone for edge cases (unequal variances, small samples)

**Implementation Notes**:
- Fallback to bootstrap-only when sample size <2 (no paired comparison possible)
- Document assumptions (normality of differences) in report disclaimer
- Validate against manual calculations in unit tests

---

## Figure Generation & LaTeX Export

**Decision**: Use **matplotlib with pgf backend** for publication-quality figures

**Rationale**:
- Matplotlib already in dependency tree (via plotting examples)
- PGF backend produces vector PDFs with embedded LaTeX fonts matching paper style
- `savefig.bbox='tight'` and `pdf.fonttype=42` settings ensure compatibility
- Established pattern in `docs/dev_guide.md` figure guidelines

**Alternatives Considered**:
- **Seaborn**: Nice defaults but adds dependency; matplotlib sufficient with rcParams tuning
- **Plotly**: Interactive but not suitable for static PDF export to papers
- **Direct LaTeX/TikZ**: Maximum control but complex; matplotlib PGF adequate

**Configuration**:
```python
# Align with dev_guide.md figure guidelines
matplotlib.rcParams.update({
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,  # TrueType fonts for LaTeX compatibility
    'font.size': 9,      # Labels
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.4,
})
```

---

## Report Template Rendering

**Decision**: Use **Python f-strings with Jinja2-lite pattern** for Markdown/LaTeX generation

**Rationale**:
- Simple templates can use f-strings with multi-line formatting
- Jinja2 (if needed) already available via many Python tools
- Markdown primary; LaTeX export via pandoc-style conversion or direct templates
- Avoid heavyweight templating engines (Django templates, Mako) for simple use case

**Alternatives Considered**:
- **Pandoc**: External dependency; requires subprocess calls; adds complexity
- **Pure f-strings**: Works for simple cases but limited for conditionals/loops
- **Jinja2**: Clean syntax for complex reports; good middle ground

**Implementation Pattern**:
```python
# Simple case: f-strings
report_md = f"""
# {experiment_name}
## Results
- Sample Efficiency: {improvement_pct:.1f}%
"""

# Complex case: Jinja2 for loops/conditionals
from jinja2 import Template
template = Template(REPORT_TEMPLATE)
rendered = template.render(metrics=data, hypothesis=result)
```

---

## Metrics Aggregation with Bootstrap CIs

**Decision**: Reuse **existing `robot_sf.benchmark.aggregate` bootstrap implementation**

**Rationale**:
- `robot_sf/benchmark/aggregate.py` already implements bootstrap resampling for CIs
- `compute_aggregates_with_ci()` function provides mean, median, p95 with [low, high] bounds
- Default 1000 samples aligns with feature requirements
- Consistent with existing benchmark outputs (maintains schema alignment)

**Alternatives Considered**:
- **scipy.stats.bootstrap**: New in scipy 1.7+; clean API but reinvents existing wheel
- **Manual implementation**: Already done; reuse > duplicate
- **statsmodels.stats.bootstrap**: Adds dependency without added value

**Integration Point**:
```python
from robot_sf.benchmark.aggregate import compute_aggregates_with_ci

aggregated = compute_aggregates_with_ci(
    episode_records,
    group_by="policy_type",
    bootstrap_samples=1000,
    bootstrap_confidence=0.95,
    bootstrap_seed=42
)
```

---

## Run Tracker Manifest Integration

**Decision**: **Direct JSON parsing** of telemetry manifests via `robot_sf.telemetry` models

**Rationale**:
- Run tracker already writes structured JSON manifests (feature 124+ integration)
- `PipelineRunRecord` dataclass provides typed access to steps, telemetry, recommendations
- `ManifestWriter` produces deterministic paths under `output/run-tracker/<run_id>/`
- No new parsing logic needed; import existing models

**Alternatives Considered**:
- **Custom parser**: Reinvents manifest schema; violates DRY
- **Database backend**: Out of scope; file-based aligns with Constitution
- **SQLite indexing**: Premature optimization; JSON scan adequate for <100 runs

**Usage Pattern**:
```python
from robot_sf.telemetry.models import PipelineRunRecord
import json

manifest_path = tracker_root / run_id / "manifest.json"
with manifest_path.open() as f:
    data = json.load(f)
    record = PipelineRunRecord(**data)
    
# Extract timing, telemetry samples, recommendations
step_durations = {s.step_id: s.duration_seconds for s in record.steps}
telemetry_count = record.summary.get("telemetry_samples", 0)
```

---

## Data Export Formats

**Decision**: **Dual export: JSON (primary) + CSV (convenience)**

**Rationale**:
- JSON preserves nested structures (CIs as [low, high], metadata as dict)
- CSV enables quick import to pandas/Excel for ad-hoc analysis
- pandas `DataFrame.to_csv()` and `to_json()` handle both formats trivially

**Schema Alignment**:
- JSON follows `training_summary.schema.json` extension pattern
- CSV flattens to one row per metric variant with columns: variant_id, metric_name, mean, median, p95, ci_low, ci_high, sample_size

**Implementation**:
```python
import pandas as pd

# JSON export (full structure)
summary_json = {
    "run_id": run_id,
    "metrics": aggregated_metrics,
    "hypothesis": hypothesis_results,
    # ...
}
with open(output_dir / "summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)

# CSV export (flattened for quick access)
df = pd.DataFrame(flatten_metrics(aggregated_metrics))
df.to_csv(output_dir / "metrics.csv", index=False)
```

---

## Hypothesis Evaluation Logic

**Decision**: **Simple threshold comparison with graceful NaN handling**

**Rationale**:
- Hypothesis: "Pre-training reduces PPO timesteps by ≥40%"
- Calculation: `improvement_pct = 100 * (baseline_steps - pretrained_steps) / baseline_steps`
- Pass: `improvement_pct >= 40.0`
- Handle missing data: report "INCOMPLETE" status if baseline or pretrained metrics unavailable

**Edge Cases**:
- Baseline slower than pretrained (negative improvement): FAIL with note
- Missing seeds: Use available subset; flag completeness ratio
- Zero baseline timesteps: Skip metric; log ERROR level warning

**Output Format**:
```json
{
  "hypothesis": "Pre-training reduces PPO timesteps by ≥40%",
  "threshold": 40.0,
  "improvement_pct": 45.2,
  "ci_low": 38.1,
  "ci_high": 52.3,
  "status": "PASS",  // PASS | FAIL | INCOMPLETE
  "note": "Based on 3/3 successful seeds"
}
```

---

## Ablation Matrix Representation

**Decision**: **Nested dict structure** keyed by parameter combinations

**Rationale**:
- Ablation parameters: `{bc_epochs: [5, 10, 20], dataset_size: [100, 200, 300]}`
- Cartesian product yields 9 variants
- Key format: `"bc5_ds100"`, `"bc10_ds200"`, etc.
- Maps cleanly to Markdown table rows and JSON export

**Data Structure**:
```python
ablation_results = {
    "bc5_ds100": {
        "bc_epochs": 5,
        "dataset_size": 100,
        "improvement_pct": 32.1,
        "hypothesis_pass": False,
        "ci": [28.3, 36.5],
        "seeds_completed": 3
    },
    "bc10_ds200": { ... },
    # ...
}
```

**Table Rendering**:
| Variant | BC Epochs | Dataset Size | Improvement % | CI (95%) | Pass (≥40%) |
|---------|-----------|--------------|---------------|----------|-------------|
| bc5_ds100 | 5 | 100 | 32.1% | [28.3, 36.5] | ❌ |
| bc10_ds200 | 10 | 200 | 45.3% | [41.2, 49.8] | ✅ |

---

## Figure Naming & Caption Standards

**Decision**: Follow **existing `docs/dev_guide.md` figure conventions**

**Rationale**:
- Prefix: `fig-<short-description>.pdf`
- Examples: `fig-learning-curve.pdf`, `fig-sample-efficiency.pdf`
- Captions include: descriptive title, axis explanations, sample size notation `(n=3 seeds)`
- Dual export: PDF (paper) + PNG (slides/web)

**Caption Template**:
```
Figure 1: Learning curves comparing baseline PPO (blue) and BC-pretrained PPO (orange) 
over training timesteps. Shaded regions indicate 95% bootstrap confidence intervals. 
Sample efficiency gain visible as earlier convergence. (n=3 seeds)
```

**Generated Figures List**:
1. `fig-learning-curve.pdf` - Timesteps vs success rate
2. `fig-sample-efficiency.pdf` - Bar chart of convergence timesteps
3. `fig-success-distribution.pdf` - Final success rate distributions
4. `fig-collision-distribution.pdf` - Final collision rate distributions
5. `fig-improvement-summary.pdf` - Effect size and p-value visualization

---

## Reproducibility Metadata Collection

**Decision**: **Composite helper function** gathering git, hardware, package versions

**Rationale**:
- Git: `subprocess.run(["git", "rev-parse", "HEAD"])` for commit hash
- Hardware: `platform.processor()`, `psutil.virtual_memory().total` (existing dep)
- Packages: `uv export --no-hashes` or `importlib.metadata.version()` for key deps
- Combine into single `collect_reproducibility_metadata()` call

**Output Schema**:
```json
{
  "git_commit": "a1b2c3d4...",
  "git_branch": "270-imitation-report",
  "git_dirty": false,
  "python_version": "3.11.5",
  "key_packages": {
    "stable-baselines3": "2.1.0",
    "imitation": "1.0.0",
    "robot-sf": "0.5.0"
  },
  "hardware": {
    "cpu": "Apple M1 Pro",
    "memory_gb": 16,
    "gpu": null
  },
  "timestamp": "2025-11-21T10:30:00Z"
}
```

---

## Directory Structure Convention

**Decision**: **Timestamp + experiment name** format for report directories

**Rationale**:
- Pattern: `output/research_reports/<timestamp>_<experiment_name>/`
- Example: `output/research_reports/20251121_103000_bc_ablation/`
- Subdirectories: `figures/`, `data/`, `configs/`, plus top-level `report.md`, `metadata.json`
- Aligns with existing artifact root policy (Constitution + dev_guide.md)

**Path Resolution**:
```python
from datetime import datetime, UTC
from pathlib import Path

timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
report_dir = Path("output/research_reports") / f"{timestamp}_{experiment_name}"
report_dir.mkdir(parents=True, exist_ok=True)

(report_dir / "figures").mkdir()
(report_dir / "data").mkdir()
(report_dir / "configs").mkdir()
```

---

## Summary of Resolved Clarifications

All technical unknowns resolved:
- ✅ Statistical library: scipy.stats
- ✅ Figure generation: matplotlib with PGF backend
- ✅ Template rendering: f-strings + Jinja2 for complex cases
- ✅ Bootstrap CIs: reuse existing `robot_sf.benchmark.aggregate`
- ✅ Tracker integration: direct JSON parsing via telemetry models
- ✅ Data export: JSON (primary) + CSV (convenience)
- ✅ Hypothesis evaluation: threshold comparison with graceful degradation
- ✅ Ablation representation: nested dict keyed by variant ID
- ✅ Figure naming: follow dev_guide.md conventions
- ✅ Reproducibility metadata: composite helper with git/hardware/packages
- ✅ Directory structure: timestamp + name under canonical artifact root

**Status**: Ready for Phase 1 (Design & Contracts)
