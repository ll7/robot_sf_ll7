# SNQI Weight Tooling – User Guide

This guide consolidates user‑facing documentation that was previously split across `scripts/QUICK_START.md` and `scripts/README_SNQI_WEIGHTS.md`. It explains how to recompute, optimize, and analyze Social Navigation Quality Index (SNQI) weights.

## Contents
- [Contents](#contents)
- [Bootstrap Examples](#bootstrap-examples)
- [Overview](#overview)
  - [Normalization Rationale (Median / p95)](#normalization-rationale-median--p95)
  - [Clamping and Outliers](#clamping-and-outliers)
- [Installation (uv)](#installation-uv)
- [Quick Start](#quick-start)
- [Core Scripts \& Typical Tasks](#core-scripts--typical-tasks)
- [CLI Arguments (Key Flags)](#cli-arguments-key-flags)
- [Input Data Formats](#input-data-formats)
  - [Episodes JSONL (one JSON object per line)](#episodes-jsonl-one-json-object-per-line)
  - [Baseline Stats JSON](#baseline-stats-json)
  - [Weights JSON](#weights-json)
- [Output JSON Schema (Summary)](#output-json-schema-summary)
  - [Diagnostics Fields](#diagnostics-fields)
- [External / Initial Weights](#external--initial-weights)
- [Recommended Workflows](#recommended-workflows)
  - [A. Establish Baseline \& Strategy Comparison](#a-establish-baseline--strategy-comparison)
  - [B. Optimize Weights Then Validate](#b-optimize-weights-then-validate)
  - [C. Evaluate External Weight Proposal](#c-evaluate-external-weight-proposal)
- [Interpreting Results](#interpreting-results)
- [Reproducibility \& Determinism](#reproducibility--determinism)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Related Design Document](#related-design-document)
- [Unified Benchmark CLI (New)](#unified-benchmark-cli-new)
  - [Inline SNQI during benchmark run](#inline-snqi-during-benchmark-run)
## Bootstrap Examples
Estimate stability and confidence intervals for the recommended weights.

Unified CLI (preferred):

```bash
# Optimize weights with 200 bootstrap samples at 95% confidence
uv run robot_sf_bench snqi optimize \
  --episodes episodes.jsonl \
  --baseline baseline_stats.json \
  --method both \
  --bootstrap-samples 200 \
  --bootstrap-confidence 0.95 \
  --output optimized_with_bootstrap.json

# Recompute strategies and bootstrap the selected (recommended) weights
uv run robot_sf_bench snqi recompute \
  --episodes episodes.jsonl \
  --baseline baseline_stats.json \
  --compare-strategies \
  --bootstrap-samples 200 \
  --bootstrap-confidence 0.95 \
  --output recompute_with_bootstrap.json
```

Result JSON includes:

- `results.bootstrap.recommended_score.samples`
- `results.bootstrap.recommended_score.mean_mean`
- `results.bootstrap.recommended_score.std_mean`
- `results.bootstrap.recommended_score.ci` (two‑sided bounds)
- `results.bootstrap.recommended_score.confidence_level`

Normalization details are documented in `docs/snqi-weight-tools/normalization.md`.

- [Contents](#contents)
- [Bootstrap Examples](#bootstrap-examples)
- [Overview](#overview)
  - [Normalization Rationale (Median / p95)](#normalization-rationale-median--p95)
- [Installation (uv)](#installation-uv)
- [Quick Start](#quick-start)
- [Core Scripts \& Typical Tasks](#core-scripts--typical-tasks)
- [CLI Arguments (Key Flags)](#cli-arguments-key-flags)
- [Input Data Formats](#input-data-formats)
  - [Episodes JSONL (one JSON object per line)](#episodes-jsonl-one-json-object-per-line)
  - [Baseline Stats JSON](#baseline-stats-json)
  - [Weights JSON](#weights-json)
- [Output JSON Schema (Summary)](#output-json-schema-summary)
  - [Diagnostics Fields](#diagnostics-fields)
- [External / Initial Weights](#external--initial-weights)
- [Recommended Workflows](#recommended-workflows)
  - [A. Establish Baseline \& Strategy Comparison](#a-establish-baseline--strategy-comparison)
  - [B. Optimize Weights Then Validate](#b-optimize-weights-then-validate)
  - [C. Evaluate External Weight Proposal](#c-evaluate-external-weight-proposal)
- [Interpreting Results](#interpreting-results)
- [Reproducibility \& Determinism](#reproducibility--determinism)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Related Design Document](#related-design-document)
- [Unified Benchmark CLI (New)](#unified-benchmark-cli-new)
  - [Inline SNQI during benchmark run](#inline-snqi-during-benchmark-run)

### Inline SNQI during benchmark run
You can compute SNQI on the fly while generating episodes via the unified benchmark CLI `run` subcommand by providing a weights JSON and baseline stats JSON:

```bash
uv run robot_sf_bench run \
  --matrix configs/baselines/example_matrix.yaml \
  --out results/episodes.jsonl \
  --schema docs/dev/issues/social-navigation-benchmark/episode_schema.json \
  --snqi-weights model/snqi_canonical_weights_v1.json \
  --snqi-baseline results/baseline_stats.json
```

When both files are provided, each episode record will include `metrics.snqi` computed using the same canonical normalization (median/p95 with clamping). Missing baseline entries fall back to neutral (0) contribution. See the naming note above for `w_near` vs `w_near_misses`.

## Overview
The Social Navigation Quality Index (SNQI) aggregates multiple navigation metrics (success, time, safety, comfort, smoothness) into a single score. It is intentionally bounded and designed for reproducibility and comparative benchmarking:
```
SNQI = w_success * success
       - w_time * time_norm
       - w_collisions * coll_norm
       - w_near * near_norm
       - w_comfort * comfort_exposure
       - w_force_exceed * force_exceed_norm
       - w_jerk * jerk_norm
```
Normalized metrics use median/p95 baseline statistics with clamping to [0,1]. Positive weights penalize adverse factors by subtraction. Metrics below baseline median floor at 0 (no negative reward); values above p95 saturate at 1 (robustness over extreme tail sensitivity). See the dedicated clamping discussion below and `docs/snqi-weight-tools/normalization.md` for details and trade-offs.

### Normalization Rationale (Median / p95)
Chosen for stability + robustness:
- Median resists skew from heavy tails.
- p95 anchors an upper bound without letting rare outliers explode scale.
- Clamping yields a predictable optimization landscape.

Limitations:
- Improvement below median not differentiated (hard floor).
- Severe outliers (>p95) collapsed together.

Alternatives (explorable via normalization comparison in recomputation script):
- Median/p90 (less tail compression)
- IQR scaling (p25/p75) potentially amplifies moderate variance; sensitive with small samples.

### Clamping and Outliers
Clamping is intentional to improve stability and comparability:

- Below median → normalized 0; above p95 → normalized 1.
- Prevents extreme outliers from dominating objective/weights.
- Compresses tails (episodes with very large adverse values become indistinguishable at 1).

If tail ordering matters for your study:

- Enable normalization comparison and review correlations.
- Experiment with a lower upper percentile (p90) or IQR/MAD.
- Keep clamping enabled for benchmark comparability unless a documented alternative is used.

Reference: `docs/snqi-weight-tools/normalization.md#clamping-and-outliers` and the design doc “Normalization Rationale & Limitations”.

## Installation (uv)
We use [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

```bash
# Clone (with submodules if not already pulled)
git clone --recursive <repo-url>
cd robot_sf_ll7

# Sync core dependencies
uv sync

# (Optional) extras:
#   progress -> tqdm progress bars
#   viz      -> plotting only (matplotlib + seaborn)
#   analysis -> seaborn/matplotlib/pandas (full visualization/analysis)
uv sync --extra progress --extra viz

# One‑line environment smoke test
uv run python scripts/validate_snqi_scripts.py
```

Add extras later without re-syncing everything:
```bash
uv sync --extra progress
uv sync --extra viz
```

Install the package in editable mode for external usage (optional):
```bash
uv run python -m pip install -e .
```

## Quick Start
```bash
# 1. Validate tooling environment
uv run python scripts/validate_snqi_scripts.py

# 2. Recompute and compare strategies (with progress bars & sampling 50 episodes)
uv run python scripts/recompute_snqi_weights.py \
  --episodes episodes.jsonl \
  --baseline baseline_stats.json \
  --compare-strategies \
  --compare-normalization \
  --sample 50 \
  --progress \
  --output strategy_comparison.json

# 3. Optimize weights via grid + evolution, run sensitivity, deterministic seed
uv run python scripts/snqi_weight_optimization.py \
  --episodes episodes.jsonl \
  --baseline baseline_stats.json \
  --method both \
  --sensitivity \
  --sample 50 \
  --seed 17 \
  --progress \
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
 - `--sample <int>`: Deterministically subsample the first N pseudo‑randomly shuffled episodes (stable under seed) for faster iteration. Metadata records original vs used counts.

Recompute specific:
- `--strategy <name>`: One of `default|balanced|safety_focused|efficiency_focused|pareto`.
- `--compare-strategies`: Evaluate all strategies.
- `--compare-normalization`: Include normalization variant comparison.
- `--external-weights-file <path>`: Evaluate user‑provided weight set.
 - `--missing-metric-max-list <int>`: Include up to N example episode IDs per missing baseline metric in diagnostics.
 - `--fail-on-missing-metric`: Treat missing baseline metrics (present in episodes) as error (exit code 4).
 - `--progress`: Show progress bars (if `tqdm` installed).
 - `--bootstrap-samples <int>` / `--bootstrap-confidence <float>`: Same semantics as optimization; operates on the recommended (or selected) weight set.

Optimization specific:
- `--method <grid|evolution|both>`: Optimization mode.
- `--grid-resolution <int>`: Candidate points per weight dim (guarded).
- `--max-grid-combinations <int>`: Cap before adaptive pruning/sampling.
- `--initial-weights-file <path>`: Starting point (valid weights JSON).
- `--sensitivity`: Enable sensitivity analysis block.
 - `--missing-metric-max-list <int>` / `--fail-on-missing-metric` / `--progress` analogous to recompute.
 - `--bootstrap-samples <int>` / `--bootstrap-confidence <float>`: Enable bootstrap estimation of the mean recommended episodic SNQI score. Adds `bootstrap.recommended_score` with mean-of-means, std, and percentile CI. Set samples to 0 (default) to disable. Confidence defaults to 0.95.
 - `--ci-placeholder` (deprecated): Legacy scaffold retained for backward compatibility; superseded by real bootstrap fields.

Sensitivity specific:
- `--weights <path>`: Weights to analyze.
- `--skip-visualizations`: Skip plotting (headless or minimal run).

Planned / future (design doc): weight simplex normalization, evolution early stopping, extended bootstrap (BCa, stratified by algorithm group, per-component), drift detection test.

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

Naming note:
- The canonical key for near-miss penalties is `w_near`.
- For backward compatibility, the validator also accepts `w_near_misses` as an alias and maps it to `w_near`. If both are provided, `w_near` wins and the alias is ignored (with a warning).
  - Preferred going forward: use `w_near`.

## Output JSON Schema (Summary)
All scripts include `_metadata` and `summary` blocks (schema version 1). See the human‑readable schema in `docs/snqi-weight-tools/schema.md`.

A machine‑readable JSON Schema is available at `docs/snqi-weight-tools/snqi_output.schema.json`. CI validation is planned as a follow‑up.

Schema stability policy: Additive (backward‑compatible) fields may appear without bumping `schema_version` (consumers should ignore unknown properties). Removals or semantic changes require incrementing `schema_version` and updating snapshot tests.

Validate programmatically using `jsonschema`:
```python
import json, jsonschema
from pathlib import Path
schema = json.loads(Path('docs/snqi-weight-tools/snqi_output.schema.json').read_text())
data = json.loads(Path('optimized_weights.json').read_text())
jsonschema.Draft202012Validator(schema).validate(data)
```

Validate from the shell (Unix) using uv + python -c:
```bash
uv run python - <<'PY'
import json, sys
from pathlib import Path
import jsonschema
schema = json.loads(Path('docs/snqi-weight-tools/snqi_output.schema.json').read_text())
for fp in sys.argv[1:]:
  data = json.loads(Path(fp).read_text())
  jsonschema.Draft202012Validator(schema).validate(data)
  print(f'VALID: {fp}')
PY optimized_weights.json
```

Example (optimization excerpt):
```json
{
  "recommended": {"weights": {...}, "objective_value": 0.72, "ranking_stability": 0.81, "method_used": "differential_evolution", "objective_components": {"stability": 0.81, "discriminative": 0.63}},
  "grid_search": {"weights": {...}, "objective_value": 0.69, "ranking_stability": 0.78, "objective_components": {"stability": 0.78, "discriminative": 0.58}},
  "differential_evolution": {"weights": {...}, "objective_value": 0.72, "ranking_stability": 0.81, "objective_components": {"stability": 0.81, "discriminative": 0.63}},
  "sensitivity_analysis": {"w_collisions": {"score_sensitivity": 0.15}},
  "_metadata": {"schema_version": 1, "generated_at": "...", "git_commit": "abc1234", "seed": 42, "phase_timings": {"load_inputs": 0.12, "grid_search": 1.31, "differential_evolution": 2.71, "sensitivity_analysis": 0.42, "write_output": 0.01}, "original_episode_count": 120, "used_episode_count": 50, "provenance": {"episodes_file": "episodes.jsonl", "baseline_file": "baseline_stats.json", "method_requested": "both"}},
  "summary": {"method": "differential_evolution", "objective_value": 0.72, "ranking_stability": 0.81, "weights": {"w_success": 2.0, ...}}
}
```
Recompute output adds: `recommended_weights`, optional `strategy_comparison`, `strategy_correlations`, `external_weights`, `normalization_comparison`, and `diagnostics`.

### Diagnostics Fields
- `skipped_malformed_lines` (metadata + summary): Count of invalid JSONL lines ignored.
- `baseline_missing_metric_count` (metadata + summary): Number of metrics present in episodes but absent from baseline normalization stats.
- `diagnostics.baseline_missing_metrics.metrics[]`: Per-metric occurrence counts + example episode IDs (bounded by `--missing-metric-max-list`).

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
- `objective_value`: Heuristic (0.6 * stability + 0.4 * discriminative power) – higher is better.
- `ranking_stability`: Spearman-based or variance proxy; ≥0.7 generally acceptable on small sets.
 - `objective_components`: Decomposed contributions (already normalized into [0,1] components) used to form `objective_value` via weighted sum; useful for diagnosing trade‑offs.
- `strategy_correlations`: Overlap in episodic ranking across strategies.
- `normalization_comparison`: Score means + correlation vs canonical median/p95.
- `sensitivity_analysis`: Local one-at-a-time perturbation effects; highlights brittle dimensions.
- `bootstrap.recommended_score`: Empirical distribution of the mean episodic SNQI score under resampling with replacement (episode-level). Fields:\n  - `samples`: Number of bootstrap replicates.\n  - `mean_mean`: Mean of bootstrap replicate means (unbiased estimator of expected mean score).\n  - `std_mean`: Standard deviation across replicate means.\n  - `ci`: Two-element percentile confidence interval `[lower, upper]` at requested confidence.\n  - `confidence_level`: The nominal confidence level used.
- `baseline_missing_metric_count`: Non-zero implies potential silent bias (consider failing CI with flag).
- `skipped_malformed_lines`: Data hygiene indicator; should normally be 0.
 - `phase_timings`: Per‑phase wall clock seconds aiding performance regression detection.

Note on heuristics (experimental):
The current stability and discriminative components are pragmatic heuristics intended for fast, reproducible runs. They will be superseded by a principled bootstrap stability metric and improved discriminative measures. See the design doc sections “Heuristic Components (v1 – Experimental)” and “Proposed Stability (Bootstrap)” for details and the transition plan.

## Reproducibility & Determinism
Pass `--seed` to ensure deterministic: Pareto sampling, grid subset sampling, differential evolution initialization. Sensitivity analysis and progress bars remain deterministic given the seed. Remaining nondeterminism may stem from SciPy internals / BLAS parallelism.

Metadata captures: seed, git commit, invocation, file provenance, schema version.

See also: `docs/snqi-weight-tools/REPRODUCIBILITY.md` for a focused note on sources of nondeterminism and mitigation tips.

## Troubleshooting
| Issue | Cause | Resolution |
|-------|-------|------------|
| EXIT_INPUT_ERROR | File not found / malformed JSONL | Verify paths; inspect failing line |
| EXIT_VALIDATION_ERROR | Output contract drift | Update schema or adapt code + snapshot test |
| EXIT_MISSING_METRIC_ERROR | Missing baseline metric + fail flag | Regenerate or extend baseline stats |
| Empty / tiny episode set | Insufficient variability | Collect ≥10–20 episodes across ≥2 algorithms |
| Large grid runtime | Combinatorial explosion | Reduce `--grid-resolution` or prefer evolution |
| Low discriminative power | Homogeneous scenarios | Add scenario/policy diversity |
| Unstable rankings | Too few episodes per algo | Increase per‑algo sample size |
| Non-finite output | Degenerate baseline (p95≈med) | Inspect baseline stats; ensure spread |
| Plots not generated | Missing matplotlib/seaborn | Install deps or use `--skip-visualizations` |

## Future Enhancements
Implemented items (moved from backlog):
- Objective component breakdown
- Deterministic episode sampling (`--sample`)
- Phase timing instrumentation
- Unified benchmark CLI entrypoint (`robot_sf_bench snqi optimize|recompute`)
- Bootstrap stability + percentile confidence intervals for the recommended mean episodic SNQI score (`bootstrap.recommended_score`)

Still planned (see design doc / backlog):
- Weight simplex option (normalize weights to sum constant before evaluating objective)
- Early stopping for evolution (plateau detection over trailing generations)
- Extended bootstrap enhancements:
  - BCa intervals
  - Stratified bootstrap by algorithm/policy group to control for mix effects
  - Per-strategy bootstrap CIs in recompute outputs
  - Bootstrap distributions for objective components and stability metric
- Drift detection test (continuous regression guard)

## Related Design Document
For deep architectural details, data contracts, algorithms, and planned roadmap see:  
`docs/dev/issues/snqi-recomputation/DESIGN.md`.

---
This user guide will be kept in sync with implementation changes. Please update both this file and the design doc when modifying schemas or adding major features.

## Unified Benchmark CLI (New)
The functionality of the optimization and recomputation scripts is now also exposed via the central benchmark entrypoint:

```
robot_sf_bench snqi optimize   # weight search
robot_sf_bench snqi recompute  # strategy / normalization workflows
```

Example parity (recompute strategy comparison):
```
robot_sf_bench snqi recompute \
  --episodes episodes.jsonl \
  --baseline baseline_stats.json \
  --compare-strategies \
  --compare-normalization \
  --output strategy_comparison.json
```

Optimization with both grid + evolution and sensitivity:
```
robot_sf_bench snqi optimize \
  --episodes episodes.jsonl \
  --baseline baseline_stats.json \
  --method both \
  --sensitivity \
  --seed 17 \
  --output optimized_weights.json
```

Fast smoke-test mode (skips heavy compute):
```
ROBOT_SF_SNQI_LIGHT_TEST=1 robot_sf_bench snqi optimize --episodes e.jsonl --baseline b.json --output w.json
```

Rationale for unified CLI:
- Single discoverable surface (`robot_sf_bench -h`).
- Lazy dynamic loading keeps startup fast for non-SNQI commands.
- Backward compatible: legacy scripts remain supported.

See also: `robot_sf/benchmark/cli.py` implementation notes.
