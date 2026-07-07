# Social Navigation Benchmark Platform - Quickstart Guide

**Purpose**: Complete step-by-step guide to execute all social navigation experiments, generate visualizations, and interpret results using the benchmark platform.

**Last Updated**: June 2026
**Implementation Status**: Most major features operational; some planned workflow sections marked as not yet shipped.

## Table of Contents
1. [Prerequisites and Setup](#prerequisites-and-setup)
2. [Quick Validation](#quick-validation)
3. [Basic Benchmark Workflow](#basic-benchmark-workflow)
4. [Advanced Experiment Execution](#advanced-experiment-execution)
5. [Visualization and Analysis](#visualization-and-analysis)
6. [Interpretation Guidelines](#interpretation-guidelines)
7. [Troubleshooting](#troubleshooting)
8. [Complete Example Workflows](#complete-example-workflows)

## Prerequisites and Setup

### System Requirements
- **OS**: Linux/macOS (headless execution supported)
- **Python**: 3.12+
- **RAM**: 8GB minimum (16GB recommended for parallel execution)
- **Storage**: 2GB for base installation, 10GB+ for experiment outputs

### Installation Steps
```bash
# 1. Clone repository and initialize submodules (CRITICAL)
git clone <repository-url>
cd robot_sf_ll7
git submodule update --init --recursive

# 2. Install dependencies
uv sync --all-extras
source .venv/bin/activate  # or activate via your shell

# 3. Verify installation
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; print('✓ Import successful')"

# 4. Test CLI access
uv run robot_sf_bench --help
```

### Environment Variables (Optional)
For headless execution (recommended for batch processing):
```bash
export DISPLAY=""
export MPLBACKEND="Agg"
export SDL_VIDEODRIVER="dummy"
```

## Quick Validation

Verify your installation with these smoke tests:

```bash
# Test 1: Environment creation (robust printing)
# Some environments return a NumPy array, others a dict (e.g., image or multi-modal).
# This snippet prints shape if available, else keys/type.
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; \
env = make_robot_env(debug=True); obs, info = env.reset(seed=42); \
import numpy as _np; \
print('✓ Environment reset successful.', end=' '); \
print('Obs shape:' , getattr(obs,'shape',None)) if hasattr(obs,'shape') else \
print('Obs keys:' , list(obs.keys())) if isinstance(obs, dict) else \
print('Obs type:', type(obs).__name__)"

# Alternative (clearer) here‑doc form (copy/paste friendly):
uv run python - <<'PY'
from robot_sf.gym_env.environment_factory import make_robot_env
env = make_robot_env(debug=True)
obs, info = env.reset(seed=42)
if hasattr(obs, 'shape'):
  print(f"✓ Environment reset successful. Obs shape: {obs.shape}")
elif isinstance(obs, dict):
  print(f"✓ Environment reset successful. Obs keys: {list(obs.keys())}")
else:
  print(f"✓ Environment reset successful. Obs type: {type(obs).__name__}")
PY

# Test 2: List scenarios from the example matrix
uv run robot_sf_bench list-scenarios --matrix configs/baselines/example_matrix.yaml

# Test 3: Quick baseline run (runs episodes + computes baseline stats)
uv run robot_sf_bench baseline \
  --matrix configs/baselines/example_matrix.yaml \
  --out /tmp/test_baseline.json \
  --jsonl /tmp/test_episode.jsonl
```

Expected output: All commands complete without errors, files generated in `/tmp/`.

## Basic Benchmark Workflow

### Step 1: Define a Scenario Matrix
Create or modify scenario matrix files. The simplest format is a flat list:

```yaml
# Example: configs/scenarios/my_experiment.yaml
- id: basic_navigation
  density: low
  flow: uni
  obstacle: open
  groups: 0.0
  speed_var: low
  goal_topology: point
  robot_context: embedded
  repeats: 5

- id: dense_navigation
  density: high
  flow: bi
  obstacle: open
  groups: 0.0
  speed_var: med
  goal_topology: point
  robot_context: embedded
  repeats: 5
```

See `configs/baselines/example_matrix.yaml` for a minimal working example.

### Step 2: Compute Baseline Statistics
The `baseline` command runs a batch of episodes from a matrix and writes both the episode JSONL and the baseline normalization stats:

```bash
uv run robot_sf_bench baseline \
  --matrix configs/scenarios/my_experiment.yaml \
  --out output/benchmarks/quickstart/my_experiment_baseline.json \
  --jsonl output/benchmarks/quickstart/my_experiment_episodes.jsonl \
  --workers 4
```

**Expected duration**: ~2-5 minutes per scenario depending on episode length and worker count.

### Step 3: Run a Full Experiment (optional, for custom algorithms)
When you need to run episodes without baseline computation (e.g., for a custom algorithm):

```bash
uv run robot_sf_bench run \
  --matrix configs/scenarios/my_experiment.yaml \
  --out output/benchmarks/quickstart/my_experiment_episodes.jsonl \
  --workers 4
```

Resume is enabled by default (existing episodes are skipped via manifest). Disable with `--no-resume`.

### Step 4: Aggregate Results with Confidence Intervals
```bash
# Generate summary statistics with bootstrap CIs
uv run robot_sf_bench aggregate \
  --in output/benchmarks/quickstart/my_experiment_episodes.jsonl \
  --out output/benchmarks/quickstart/my_experiment_summary.json \
  --bootstrap-samples 1000 \
  --bootstrap-confidence 0.95 \
  --bootstrap-seed 42
```

**Output**: JSON file with mean/median/p95 for each metric, plus confidence intervals.

## Advanced Experiment Execution

### Multi-Baseline Comparison
First create `configs/scenarios/comparison_study.yaml` using the Step 1 matrix format.

```bash
# Run multiple algorithms on the same scenarios
for ALGO in "simple_policy" "baseline_sf" "random"; do
  uv run robot_sf_bench run \
    --matrix configs/scenarios/comparison_study.yaml \
    --out output/benchmarks/quickstart/episodes_${ALGO}.jsonl \
    --algo $ALGO \
    --workers 4
done

# Combine results for comparison
cat output/benchmarks/quickstart/episodes_*.jsonl > output/benchmarks/quickstart/episodes_combined.jsonl

# Aggregate by algorithm
uv run robot_sf_bench aggregate \
  --in output/benchmarks/quickstart/episodes_combined.jsonl \
  --out output/benchmarks/quickstart/comparison_summary.json \
  --group-by "scenario_params.algo" \
  --bootstrap-samples 1000
```

### SNQI Weight Analysis
```bash
# Recompute SNQI with custom weights via predefined strategy
uv run robot_sf_bench snqi recompute \
  --episodes output/benchmarks/quickstart/my_experiment_episodes.jsonl \
  --baseline output/benchmarks/quickstart/my_experiment_baseline.json \
  --output output/benchmarks/quickstart/recomputed_weights.json \
  --strategy default

# SNQI ablation: measure rank shifts from one-at-a-time component removal
uv run robot_sf_bench snqi-ablate \
  --in output/benchmarks/quickstart/my_experiment_episodes.jsonl \
  --out output/benchmarks/quickstart/snqi_ablation.md \
  --snqi-weights output/benchmarks/quickstart/recomputed_weights.json \
  --snqi-baseline output/benchmarks/quickstart/my_experiment_baseline.json
```

### SNQI Weight Optimization (Grid / Evolution)
```bash
uv run robot_sf_bench snqi optimize \
  --episodes output/benchmarks/quickstart/my_experiment_episodes.jsonl \
  --baseline output/benchmarks/quickstart/my_experiment_baseline.json \
  --output output/benchmarks/quickstart/optimized_weights.json \
  --method both
```

### Large-Scale Parameter Sweeps
For sweeps, create the matrix YAML manually — there is no generate-scenarios CLI command yet. Use a small script or hand-craft the file with your parameter ranges:

```yaml
# Example: configs/scenarios/full_sweep.yaml (user-created)
- id: sweep_low_slow
  density: low
  flow: uni
  obstacle: open
  groups: 0.0
  speed_var: low
  goal_topology: point
  robot_context: embedded
  repeats: 3
# ... extend with desired parameter combinations
```

Then execute with high parallelism:

```bash
uv run robot_sf_bench run \
  --matrix configs/scenarios/full_sweep.yaml \
  --out output/benchmarks/quickstart/parameter_sweep_episodes.jsonl \
  --workers 8
```

## Visualization and Analysis

### Core CLI Visualizations
Use the dedicated plot subcommands:

```bash
# Pareto front (two metrics grouped by algo)
uv run robot_sf_bench plot-pareto \
  --in output/benchmarks/quickstart/my_experiment_episodes.jsonl \
  --out output/benchmarks/quickstart/figures/pareto.png \
  --x-metric collisions \
  --y-metric comfort_exposure

# Per-metric distribution histograms
uv run robot_sf_bench plot-distributions \
  --in output/benchmarks/quickstart/my_experiment_episodes.jsonl \
  --out-dir output/benchmarks/quickstart/figures/distributions/ \
  --metrics collisions,comfort_exposure,time_to_goal \
  --kde

# Baseline comparison table (Markdown/CSV/LaTeX)
uv run robot_sf_bench table \
  --in output/benchmarks/quickstart/my_experiment_episodes.jsonl \
  --out output/benchmarks/quickstart/figures/baseline_table.md \
  --metrics collisions,comfort_exposure,time_to_goal \
  --format md
```

### Batch Figure Generation Script
For a full set of publication-quality figures, use the dedicated script:

```bash
uv run python scripts/generate_figures.py \
  --episodes output/benchmarks/quickstart/my_experiment_episodes.jsonl \
  --out-dir output/benchmarks/quickstart/figures/ \
  --pareto-x collisions --pareto-y comfort_exposure \
  --dmetrics collisions,comfort_exposure,time_to_goal \
  --table-metrics collisions,comfort_exposure,time_to_goal --table-tex \
  --thumbs-matrix configs/baselines/example_matrix.yaml --thumbs-montage
```

See `scripts/generate_figures.py --help` for all options (Pareto, distributions, force-field, thumbnails, tables).

### Interactive Analysis
```python
# Python script for custom analysis
from robot_sf.benchmark.aggregate import read_jsonl, compute_aggregates_with_ci
import matplotlib.pyplot as plt

# Load results
episodes = read_jsonl("output/benchmarks/quickstart/my_experiment_episodes.jsonl")

# Custom aggregation
summary = compute_aggregates_with_ci(
    episodes,
    group_by="scenario_params.ped_density",
    bootstrap_samples=1000
)

# Custom plotting
densities = list(summary.keys())
snqi_means = [summary[d]["metrics.snqi"]["mean"] for d in densities]
snqi_cis = [summary[d]["metrics.snqi"]["mean_ci"] for d in densities]

plt.errorbar(densities, snqi_means, yerr=[[ci[1]-m for ci, m in zip(snqi_cis, snqi_means)],
                                          [m-ci[0] for ci, m in zip(snqi_cis, snqi_means)]])
plt.xlabel("Pedestrian Density")
plt.ylabel("SNQI Score")
plt.title("Social Navigation Performance vs Pedestrian Density")
plt.savefig("output/benchmarks/quickstart/figures/custom_analysis.png", dpi=300)
```

### Trajectory Analysis (Not Yet Shipped)
There is no `extract-trajectories` CLI command yet. For custom trajectory extraction, use the programmatic API:
```python
from robot_sf.benchmark.aggregate import read_jsonl
episodes = read_jsonl("output/benchmarks/quickstart/my_experiment_episodes.jsonl")
# Manually filter and process episodes with trajectory data
```

## Interpretation Guidelines

### Metric Understanding
**SNQI (Social Navigation Quality Index)**: Composite score (0-1, higher better)
- `> 0.8`: Excellent social navigation
- `0.6-0.8`: Good performance
- `0.4-0.6`: Acceptable performance
- `< 0.4`: Poor social behavior

**Key Component Metrics**:
- `collisions`: Number of collisions (lower better)
- `comfort_exposure`: Pedestrian comfort exposure (0-1, lower better)
- `path_efficiency`: Path efficiency (0-1, higher better)
- `snqi`: Composite SNQI score (0-1, higher better)

### Statistical Significance
- **Bootstrap CIs**: 95% confidence intervals indicate statistical reliability
- **Non-overlapping CIs**: Strong evidence of performance difference
- **Overlapping CIs**: Require additional statistical tests
- **Sample size**: Minimum 20 episodes per condition recommended

### Performance Benchmarks
**Baseline Expectations** (from validation):
- **SocialForce**: SNQI ~0.65, robust but conservative
- **PPO (trained)**: SNQI ~0.75, context-dependent
- **Random**: SNQI ~0.25, useful as lower bound

**Environment Performance**:
- 20-25 steps/second typical
- 500-step episodes complete in 20-25 seconds
- Parallel execution scales linearly up to CPU core count

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Symptom: ModuleNotFoundError
# Solution: Verify virtual environment and submodules
source .venv/bin/activate
git submodule update --init --recursive
uv sync
```

**Display Errors**:
```bash
# Symptom: "can't connect to display"
# Solution: Use headless mode
export DISPLAY=""
export MPLBACKEND="Agg"
export SDL_VIDEODRIVER="dummy"
```

**Performance Issues**:
```bash
# Symptom: Slow episode execution
# Solution: Check resource usage and reduce workers
htop  # Monitor CPU/memory
uv run robot_sf_bench run --matrix configs/baselines/example_matrix.yaml --out /tmp/debug.jsonl --workers 2
```

**File Corruption**:
```bash
# Symptom: JSON decode errors in JSONL files
# Solution: Validate manually with Python
uv run python -c "
import json
with open('output/benchmarks/quickstart/problematic_episodes.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'Line {i}: {e}')
"
```

There is no `validate-episodes` CLI command yet; use direct JSONL validation for now.

### Debugging Commands
```bash
# Verbose logging
uv run robot_sf_bench run \
  --matrix configs/baselines/example_matrix.yaml \
  --out /tmp/debug.jsonl \
  --structured-output json \
  --repeats 1

# Profile performance
uv run python -c "
import cProfile
from robot_sf.benchmark.runner import run_single_episode
# ... profiling code
"
```

## Complete Example Workflows

### Workflow 1: Quick Performance Assessment
**Goal**: Evaluate a new robot policy against baselines
**Time**: ~15 minutes

```bash
# 1. Setup (30 seconds)
export MPLBACKEND="Agg"
cd robot_sf_ll7

# 2. Compute baseline (10 minutes)
uv run robot_sf_bench baseline \
  --matrix configs/baselines/example_matrix.yaml \
  --out output/benchmarks/quickstart/baseline_stats.json \
  --jsonl output/benchmarks/quickstart/policy_comparison.jsonl \
  --workers 4

# 3. Generate summary (2 minutes)
uv run robot_sf_bench aggregate \
  --in output/benchmarks/quickstart/policy_comparison.jsonl \
  --out output/benchmarks/quickstart/policy_summary.json \
  --bootstrap-samples 500

# 4. Visualize (2 minutes)
uv run robot_sf_bench plot-pareto \
  --in output/benchmarks/quickstart/policy_comparison.jsonl \
  --out output/benchmarks/quickstart/policy_pareto.png \
  --x-metric collisions \
  --y-metric comfort_exposure

echo "✓ Results in output/benchmarks/quickstart/policy_summary.json and output/benchmarks/quickstart/policy_pareto.png"
```

### Workflow 2: Comprehensive Research Study
**Goal**: Multi-parameter analysis with publication-quality figures
**Time**: ~2-4 hours

```bash
# 1. Create scenario matrix manually (5 minutes)
# See configs/baselines/example_matrix.yaml for format.
# Edit configs/scenarios/comprehensive_study.yaml with your parameter combinations.

# 2. Execute full study (2-3 hours, can run overnight)
uv run robot_sf_bench run \
  --matrix configs/scenarios/comprehensive_study.yaml \
  --out output/benchmarks/quickstart/comprehensive_episodes.jsonl \
  --workers 8

# 3. Generate all analyses (15 minutes)
uv run robot_sf_bench aggregate \
  --in output/benchmarks/quickstart/comprehensive_episodes.jsonl \
  --out output/benchmarks/quickstart/comprehensive_summary.json \
  --group-by "scenario_params.algo" \
  --bootstrap-samples 2000 \
  --bootstrap-confidence 0.95

# 4. Create publication figures (10 minutes)
uv run python scripts/generate_figures.py \
  --episodes output/benchmarks/quickstart/comprehensive_episodes.jsonl \
  --out-dir output/benchmarks/quickstart/publication_figures/ \
  --pareto-x collisions --pareto-y comfort_exposure --pareto-pdf \
  --dmetrics collisions,comfort_exposure,time_to_goal --dists-pdf \
  --table-metrics collisions,comfort_exposure,time_to_goal --table-tex \
  --force-field

# 5. Generate LaTeX table from aggregate summary
uv run robot_sf_bench table \
  --in output/benchmarks/quickstart/comprehensive_episodes.jsonl \
  --out output/benchmarks/quickstart/publication_figures/results_table.tex \
  --metrics collisions,comfort_exposure,time_to_goal \
  --format tex

echo "✓ Comprehensive study complete. See output/benchmarks/quickstart/publication_figures/"
```

### Workflow 3: SNQI Weight Sensitivity Analysis
**Goal**: Understand how SNQI component weighting affects rankings
**Time**: ~45 minutes

```bash
# 1. Baseline experiment (15 minutes)
uv run robot_sf_bench baseline \
  --matrix configs/baselines/example_matrix.yaml \
  --out output/benchmarks/quickstart/snqi_baseline.json \
  --jsonl output/benchmarks/quickstart/snqi_base_episodes.jsonl \
  --workers 4

# 2. Recompute SNQI weights with different strategies
uv run robot_sf_bench snqi recompute \
  --episodes output/benchmarks/quickstart/snqi_base_episodes.jsonl \
  --baseline output/benchmarks/quickstart/snqi_baseline.json \
  --output output/benchmarks/quickstart/snqi_weights.json \
  --strategy default \
  --compare-strategies

# 3. SNQI ablation: rank stability under component removal
uv run robot_sf_bench snqi-ablate \
  --in output/benchmarks/quickstart/snqi_base_episodes.jsonl \
  --out output/benchmarks/quickstart/weight_ablation/ablation.md \
  --snqi-weights output/benchmarks/quickstart/snqi_weights.json \
  --snqi-baseline output/benchmarks/quickstart/snqi_baseline.json \
  --format md

# 4. Optimize weights via grid search
uv run robot_sf_bench snqi optimize \
  --episodes output/benchmarks/quickstart/snqi_base_episodes.jsonl \
  --baseline output/benchmarks/quickstart/snqi_baseline.json \
  --output output/benchmarks/quickstart/optimized_weights.json \
  --method grid \
  --grid-resolution 5

echo "✓ Weight sensitivity analysis complete. See output/benchmarks/quickstart/"
```

---

## Next Steps

After completing this quickstart:

1. **Explore Advanced Features**: See `docs/` for detailed component documentation
2. **Custom Metrics**: Add domain-specific metrics to the metrics module
3. **New Baselines**: Implement additional planners using the PlannerProtocol interface
4. **Integration**: Incorporate benchmark platform into your research workflow
5. **Community**: Share configurations and results with the research community

**Support**: For technical issues, see troubleshooting section or consult `docs/dev_guide.md` for development guidance.

## Outputs Summary

All artifacts are written under `output/benchmarks/quickstart/`:
- `episodes.jsonl` (raw per-episode lines)
- `episodes.jsonl.manifest.json` (resume index)
- `*_baseline.json` (normalization stats)
- `*_weights.json` (SNQI weights artifact)
- `*_summary.json` (aggregated metrics + optional CIs)
- `figures/...` (visual assets & tables)
- `weight_ablation/...` (component influence)

## Known Gaps

The following workflow areas are **not yet shipped** as CLI commands:
- **Scenario generation**: Create scenario matrices manually or with a helper script
- **Trajectory extraction**: Use the programmatic API (`robot_sf.benchmark.aggregate.read_jsonl`)
- **Episode validation**: Validate JSONL files manually with Python
- **generate-scenarios CLI**: Currently manual; automated generation is planned
- **extract-trajectories CLI**: Currently manual; use the programmatic API
- **validate-episodes CLI**: Currently manual; use built-in Python validation
- **analyze-weight-sensitivity CLI**: SNQI ablation (`robot_sf_bench snqi-ablate`) covers partial scope

---

**Status**: CLI commands validated - interface patterns align with `robot_sf.benchmark.cli` implementation.
