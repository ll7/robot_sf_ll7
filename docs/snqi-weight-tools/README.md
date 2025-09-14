# SNQI Weight Tooling – User Guide

This guide consolidates user‑facing documentation that was previously split across `scripts/QUICK_START.md` and `scripts/README_SNQI_WEIGHTS.md`. It explains how to recompute, optimize, and analyze Social Navigation Quality Index (SNQI) weights.

## Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Scripts & Typical Tasks](#core-scripts--typical-tasks)
- [CLI Arguments (Key Flags)](#cli-arguments-key-flags)
- [Input Data Formats](#input-data-formats)
- [Output JSON Schema (Summary)](#output-json-schema-summary)
- [External / Initial Weights](#external--initial-weights)
- [Recommended Workflows](#recommended-workflows)
- [Interpreting Results](#interpreting-results)
- [Reproducibility & Determinism](#reproducibility--determinism)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Related Design Document](#related-design-document)

## Overview
The Social Navigation Quality Index (SNQI) aggregates multiple navigation metrics (success, time, safety, comfort, smoothness) into a single score:
```
SNQI = w_success * success
       - w_time * time_norm
       - w_collisions * coll_norm
       - w_near * near_norm
       - w_comfort * comfort_exposure
       - w_force_exceed * force_exceed_norm
       - w_jerk * jerk_norm
```
Normalized metrics use median/p95 baseline statistics with clamping to [0,1]. Positive weights penalize negative factors by subtraction.

## Quick Start
```bash
# 1. Validate tooling environment
python scripts/validate_snqi_scripts.py

# 2. Recompute and compare strategies
python scripts/recompute_snqi_weights.py \
  --episodes episodes.jsonl \
  --baseline baseline_stats.json \
  --compare-strategies \
  --output strategy_comparison.json

# 3. Optimize weights via grid + evolution
python scripts/snqi_weight_optimization.py \
  --episodes episodes.jsonl \
  --baseline baseline_stats.json \
  --method both \
  --sensitivity \
  --output optimized_weights.json
```

## Core Scripts & Typical Tasks
| Script | Purpose | Typical Use |
|--------|---------|-------------|
| `recompute_snqi_weights.py` | Evaluate predefined strategies + (optional) normalization comparison | Establish baseline or compare styles (default, safety, efficiency, pareto) |
| `snqi_weight_optimization.py` | Search weight space (grid + differential evolution) | Derive candidate optimized weights |
| `snqi_sensitivity_analysis.py` | Analyze robustness to weight perturbations | Post‑optimization validation |
| `validate_snqi_scripts.py` | Sanity check imports & basic compute | Environment verification |
| `example_snqi_workflow.py` | Demonstration pipeline | Learning / documentation |

## CLI Arguments (Key Flags)
(Flags vary per script; run with `-h` for the authoritative list.)

Common:
- `--episodes <path>`: Episodes JSONL input.
- `--baseline <path>`: Baseline median/p95 stats JSON.
- `--seed <int>`: RNG seed for deterministic sampling / optimization.
- `--output <path>`: Output JSON (or directory for sensitivity).

Recompute specific:
- `--strategy <name>`: One of `default|balanced|safety_focused|efficiency_focused|pareto`.
- `--compare-strategies`: Evaluate all strategies.
- `--compare-normalization`: Include normalization variant comparison.
- `--external-weights-file <path>`: Evaluate user‑provided weight set.

Optimization specific:
- `--method <grid|evolution|both>`: Optimization mode.
- `--grid-resolution <int>`: Candidate points per weight dim (guarded).
- `--max-grid-combinations <int>`: Cap before adaptive pruning/sampling.
- `--initial-weights-file <path>`: Starting point (valid weights JSON).
- `--sensitivity`: Enable sensitivity analysis block.

Sensitivity specific:
- `--weights <path>`: Weights to analyze.
- `--skip-visualizations`: Skip plotting (headless or minimal run).

Planned / future (design doc): `--sample`, `--fail-on-missing-metric`, bootstrap stability flags, progress reporting (`tqdm`).

## Input Data Formats
### Episodes JSONL (one JSON object per line)
```json
{ "episode_id": "scenario_1--seed_42", "metrics": { "success": true, "time_to_goal_norm": 0.75, "collisions": 0, "near_misses": 2, "comfort_exposure": 0.1, "force_exceed_events": 5, "jerk_mean": 0.3 }, "scenario_params": { "algo": "baseline_sf", "density": "med" } }
```

### Baseline Stats JSON
```json
{ "collisions": {"med": 0.0, "p95": 2.0}, "near_misses": {"med": 1.0, "p95": 5.0}, "force_exceed_events": {"med": 2.0, "p95": 15.0}, "jerk_mean": {"med": 0.2, "p95": 1.0} }
```
Missing metrics default to neutral (0 contribution) currently.

### Weights JSON
```json
{ "w_success": 2.0, "w_time": 1.0, "w_collisions": 2.0, "w_near": 1.0, "w_comfort": 1.5, "w_force_exceed": 1.5, "w_jerk": 0.5 }
```

## Output JSON Schema (Summary)
All scripts include `_metadata` and `summary` blocks (schema version 1). Example (optimization):
```json
{
  "recommended": {"weights": {...}, "objective_value": 0.72, "ranking_stability": 0.81, "method_used": "differential_evolution"},
  "grid_search": {"weights": {...}, "objective_value": 0.69, "ranking_stability": 0.78},
  "differential_evolution": {"weights": {...}, "objective_value": 0.72, "ranking_stability": 0.81},
  "sensitivity_analysis": {"w_collisions": {"score_sensitivity": 0.15}},
  "_metadata": {"schema_version": 1, "generated_at": "...", "git_commit": "abc1234", "seed": 42, "provenance": {"episodes_file": "episodes.jsonl", "baseline_file": "baseline_stats.json", "method_requested": "both"}},
  "summary": {"method": "differential_evolution", "objective_value": 0.72, "ranking_stability": 0.81, "weights": {"w_success": 2.0, ...}}
}
```
Recompute output adds: `recommended_weights`, optional `strategy_comparison`, `strategy_correlations`, `external_weights`, `normalization_comparison`.

## External / Initial Weights
Use `--external-weights-file` (recompute) or `--initial-weights-file` (optimization) with a JSON mapping of all required weight keys. Validation enforces:
- Presence of all canonical weight names
- Positivity and finiteness
- Extraneous keys ignored with a warning

## Recommended Workflows
### A. Establish Baseline & Strategy Comparison
```bash
python scripts/recompute_snqi_weights.py \
  --episodes episodes.jsonl \
  --baseline baseline_stats.json \
  --compare-strategies --compare-normalization \
  --output recompute_all.json
```
### B. Optimize Weights Then Validate
```bash
python scripts/snqi_weight_optimization.py \
  --episodes episodes.jsonl --baseline baseline_stats.json \
  --method both --sensitivity --seed 17 \
  --output optimized.json
python scripts/snqi_sensitivity_analysis.py \
  --episodes episodes.jsonl --baseline baseline_stats.json \
  --weights optimized.json --output sensitivity_results/ --skip-visualizations
```
### C. Evaluate External Weight Proposal
```bash
python scripts/recompute_snqi_weights.py \
  --episodes episodes.jsonl --baseline baseline_stats.json \
  --external-weights-file candidate_weights.json \
  --strategy default --output eval_candidate.json
```

## Interpreting Results
- `objective_value`: Combined (currently heuristic) score mixing stability & discriminative power.
- `ranking_stability`: Higher = rankings robust across internal perturbations.
- `strategy_correlations`: How similar strategy rankings are (Spearman or Pearson depending on implementation).
- `normalization_comparison`: Correlation vs canonical median/p95 normalization.
- `sensitivity_analysis`: Which weights most affect ranking variance.

## Reproducibility & Determinism
Pass `--seed` to ensure deterministic: sampling (Pareto), grid sampling fallback, differential evolution initialization. Remaining nondeterminism: potential SciPy internal parallelism.

Metadata captures: seed, git commit, invocation, file provenance, schema version.

## Troubleshooting
| Issue | Cause | Resolution |
|-------|-------|------------|
| Empty / tiny episode set | Insufficient data for stability | Collect ≥ 10–20 episodes (≥ 2 algos improves stability) |
| Missing baseline metric | Key absent in baseline JSON | Add med/p95 entry or accept neutral contribution |
| Large grid runtime | Exponential combination count | Lower `--grid-resolution` or rely on evolution |
| Non-finite output | Bad input metric or division | Check baseline denominators; ensure p95>med |
| Plots not generated | Missing matplotlib/seaborn | Install or use `--skip-visualizations` |

## Future Enhancements
Planned (see design doc): bootstrap stability metric, multi-objective frontier (NSGA-II), progress bars, episode sampling, exit code taxonomy, schema snapshot tests, weight simplex option.

## Related Design Document
For deep architectural details, data contracts, algorithms, and planned roadmap see:  
`docs/dev/issues/snqi-recomputation/DESIGN.md`.

---
This user guide will be kept in sync with implementation changes. Please update both this file and the design doc when modifying schemas or adding major features.
