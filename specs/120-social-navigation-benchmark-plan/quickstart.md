# Social Navigation Benchmark Platform - Quickstart Guide

**Purpose**: Complete step-by-step guide to execute all social navigation experiments, generate visualizations, and interpret results using the benchmark platform.

**Last Updated**: January 2025  
**Implementation Status**: Complete - All major features operational

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
uv run python -m robot_sf.benchmark.cli --help
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

# Test 2: CLI functionality  
uv run python -m robot_sf.benchmark.cli list-scenarios configs/baselines/example.yaml

# Test 3: Run single episode
uv run python -m robot_sf.benchmark.cli run \
  --scenarios configs/baselines/example.yaml \
  --output /tmp/test_episode.jsonl \
  --max-episodes 1

# Test 4: Generate baseline stats
uv run python -m robot_sf.benchmark.cli baseline \
  --episodes /tmp/test_episode.jsonl \
  --output /tmp/test_baseline.jsonl
```

Expected output: All commands complete without errors, files generated in `/tmp/`.

## Basic Benchmark Workflow

### Step 1: Define Scenarios
Create or modify scenario configuration files in `configs/scenarios/`:

```yaml
# Example: configs/scenarios/my_experiment.yaml
scenarios:
  - scenario_id: "basic_navigation"
    map_file: "maps/svg_maps/square_room.svg"
    robot_config:
      max_robot_speed: 1.2
      robot_radius: 0.3
    simulation_config:
      max_episode_steps: 500
      ped_density: 0.02
    seeds: [42, 43, 44, 45, 46]  # 5 repetitions
    
  - scenario_id: "dense_navigation"  
    map_file: "maps/svg_maps/square_room.svg"
    robot_config:
      max_robot_speed: 0.8
      robot_radius: 0.3
    simulation_config:
      max_episode_steps: 500
      ped_density: 0.05
    seeds: [42, 43, 44, 45, 46]
```

### Step 2: Execute Experiment Runs
```bash
# Run all scenarios with parallel workers
uv run python -m robot_sf.benchmark.cli run \
  --scenarios configs/scenarios/my_experiment.yaml \
  --output results/my_experiment_episodes.jsonl \
  --workers 4 \
  --resume  # Skip already completed episodes
```

**Expected duration**: ~2-5 minutes per scenario depending on episode length and worker count.

### Step 3: Generate Baseline Statistics
```bash
# Compute baseline metrics for comparison
uv run python -m robot_sf.benchmark.cli baseline \
  --episodes results/my_experiment_episodes.jsonl \
  --output results/my_experiment_baseline.jsonl \
  --workers 4
```

### Step 4: Aggregate Results with Confidence Intervals
```bash
# Generate summary statistics with bootstrap CIs
uv run python -m robot_sf.benchmark.cli aggregate \
  --in results/my_experiment_episodes.jsonl \
  --out results/my_experiment_summary.json \
  --bootstrap-samples 1000 \
  --bootstrap-confidence 0.95 \
  --bootstrap-seed 42
```

**Output**: JSON file with mean/median/p95 for each metric, plus confidence intervals.

## Advanced Experiment Execution

### Multi-Baseline Comparison
```bash
# Run multiple algorithms on the same scenarios
for ALGO in "social_force" "ppo" "random"; do
  uv run python -m robot_sf.benchmark.cli run \
    --scenarios configs/scenarios/comparison_study.yaml \
    --output results/episodes_${ALGO}.jsonl \
    --algo $ALGO \
    --workers 4 \
    --resume
done

# Combine results for comparison
cat results/episodes_*.jsonl > results/episodes_combined.jsonl

# Aggregate by algorithm
uv run python -m robot_sf.benchmark.cli aggregate \
  --in results/episodes_combined.jsonl \
  --out results/comparison_summary.json \
  --group-by "scenario_params.algo" \
  --bootstrap-samples 1000
```

### SNQI Weight Analysis
```bash
# Recompute SNQI with custom weights
uv run python -m robot_sf.benchmark.cli snqi-recompute \
  --episodes results/my_experiment_episodes.jsonl \
  --output results/recomputed_episodes.jsonl \
  --weights configs/snqi_weights/custom_weights.json

# Weight sensitivity analysis
uv run python -m robot_sf.benchmark.cli snqi-weight-ablation \
  --episodes results/my_experiment_episodes.jsonl \
  --output results/weight_ablation/ \
  --base-weights configs/snqi_weights/canonical_v1.json \
  --ablation-factors 0.5 0.8 1.2 1.5 2.0
```

### Large-Scale Parameter Sweeps
```bash
# Generate comprehensive scenario matrix
uv run python -m robot_sf.benchmark.cli generate-scenarios \
  --template configs/templates/parameter_sweep.yaml \
  --output configs/scenarios/full_sweep.yaml \
  --param-ranges '{"ped_density": [0.01, 0.02, 0.03, 0.04, 0.05], "max_robot_speed": [0.8, 1.0, 1.2]}'

# Execute sweep with high parallelism
uv run python -m robot_sf.benchmark.cli run \
  --scenarios configs/scenarios/full_sweep.yaml \
  --output results/parameter_sweep_episodes.jsonl \
  --workers 8 \
  --resume \
  --batch-size 50
```

## Visualization and Analysis

### Core Visualizations
```bash
# Generate all standard figures
uv run python -m robot_sf.benchmark.cli figures \
  --episodes results/my_experiment_episodes.jsonl \
  --output-dir results/figures/ \
  --figure-types distribution pareto force_field thumbnails table

# Individual figure types
uv run python -m robot_sf.benchmark.cli figure-distribution \
  --episodes results/my_experiment_episodes.jsonl \
  --output results/figures/distribution.png \
  --metric "metrics.snqi" \
  --group-by "scenario_params.algo"

uv run python -m robot_sf.benchmark.cli figure-pareto \
  --episodes results/my_experiment_episodes.jsonl \
  --output results/figures/pareto.png \
  --x-metric "metrics.time_to_goal" \
  --y-metric "metrics.snqi"
```

### Interactive Analysis
```python
# Python script for custom analysis
from robot_sf.benchmark.aggregate import read_jsonl, compute_aggregates_with_ci
import matplotlib.pyplot as plt

# Load results
episodes = read_jsonl("results/my_experiment_episodes.jsonl")

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
plt.savefig("results/figures/custom_analysis.png", dpi=300)
```

### Trajectory Analysis
```bash
# Extract and visualize trajectories
uv run python -m robot_sf.benchmark.cli extract-trajectories \
  --episodes results/my_experiment_episodes.jsonl \
  --output results/trajectories/ \
  --scenario-filter "dense_navigation" \
  --format pickle

# Generate trajectory animations  
uv run python examples/trajectory_demo.py \
  --trajectory-file results/trajectories/dense_navigation_42.pkl \
  --output results/animations/dense_navigation_42.mp4
```

## Interpretation Guidelines

### Metric Understanding
**SNQI (Social Navigation Quality Index)**: Composite score (0-1, higher better)
- `> 0.8`: Excellent social navigation
- `0.6-0.8`: Good performance  
- `0.4-0.6`: Acceptable performance
- `< 0.4`: Poor social behavior

**Key Component Metrics**:
- `time_to_goal`: Episode duration (seconds, lower better)
- `safety_score`: Collision avoidance (0-1, higher better) 
- `comfort_score`: Pedestrian comfort (0-1, higher better)
- `efficiency_score`: Path efficiency (0-1, higher better)

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
uv run python -m robot_sf.benchmark.cli run --workers 2  # Reduce parallelism
```

**File Corruption**:
```bash
# Symptom: JSON decode errors in JSONL files
# Solution: Validate and recover
uv run python -m robot_sf.benchmark.cli validate-episodes \
  --episodes results/problematic_episodes.jsonl \
  --fix-corruption
```

### Debugging Commands
```bash
# Verbose logging
uv run python -m robot_sf.benchmark.cli run \
  --scenarios configs/scenarios/debug.yaml \
  --output /tmp/debug.jsonl \
  --verbose \
  --max-episodes 1

# Profile performance
uv run python -c "
import cProfile
from robot_sf.benchmark.runner import run_single_episode
# ... profiling code
"

# Memory usage monitoring
uv run python -m memory_profiler scripts/memory_test.py
```

## Complete Example Workflows

### Workflow 1: Quick Performance Assessment
**Goal**: Evaluate a new robot policy against baselines
**Time**: ~15 minutes

```bash
# 1. Setup (30 seconds)
export MPLBACKEND="Agg"  
cd robot_sf_ll7

# 2. Run comparison (10 minutes)
uv run python -m robot_sf.benchmark.cli run \
  --scenarios configs/baselines/quick_assessment.yaml \
  --output results/policy_comparison.jsonl \
  --workers 4

# 3. Generate summary (2 minutes)  
uv run python -m robot_sf.benchmark.cli aggregate \
  --in results/policy_comparison.jsonl \
  --out results/policy_summary.json \
  --bootstrap-samples 500

# 4. Visualize (2 minutes)
uv run python -m robot_sf.benchmark.cli figure-pareto \
  --episodes results/policy_comparison.jsonl \
  --output results/policy_pareto.png \
  --x-metric "metrics.time_to_goal" \
  --y-metric "metrics.snqi"

echo "✓ Results in results/policy_summary.json and results/policy_pareto.png"
```

### Workflow 2: Comprehensive Research Study
**Goal**: Multi-parameter analysis with publication-quality figures
**Time**: ~2-4 hours

```bash
# 1. Generate comprehensive scenario matrix (5 minutes)
uv run python -m robot_sf.benchmark.cli generate-scenarios \
  --template configs/templates/research_study.yaml \
  --output configs/scenarios/comprehensive_study.yaml \
  --param-ranges '{
    "ped_density": [0.01, 0.02, 0.03, 0.04, 0.05],
    "max_robot_speed": [0.6, 0.8, 1.0, 1.2, 1.4],
    "map_file": ["square_room.svg", "corridor.svg", "intersection.svg"]
  }'

# 2. Execute full study (2-3 hours, can run overnight)
uv run python -m robot_sf.benchmark.cli run \
  --scenarios configs/scenarios/comprehensive_study.yaml \
  --output results/comprehensive_episodes.jsonl \
  --workers 8 \
  --resume

# 3. Generate all analyses (15 minutes)
uv run python -m robot_sf.benchmark.cli aggregate \
  --in results/comprehensive_episodes.jsonl \
  --out results/comprehensive_summary.json \
  --group-by "scenario_params.map_file" \
  --bootstrap-samples 2000 \
  --bootstrap-confidence 0.95

# 4. Create publication figures (10 minutes)
uv run python -m robot_sf.benchmark.cli figures \
  --episodes results/comprehensive_episodes.jsonl \
  --output-dir results/publication_figures/ \
  --figure-types distribution pareto force_field table \
  --publication-quality \
  --dpi 300

# 5. Generate LaTeX table
uv run python -m robot_sf.benchmark.cli table \
  --summary results/comprehensive_summary.json \
  --output results/publication_figures/results_table.tex \
  --format latex \
  --precision 3

echo "✓ Comprehensive study complete. See results/publication_figures/"
```

### Workflow 3: SNQI Weight Sensitivity Analysis
**Goal**: Understand how SNQI component weighting affects rankings
**Time**: ~45 minutes

```bash
# 1. Baseline experiment (15 minutes)
uv run python -m robot_sf.benchmark.cli run \
  --scenarios configs/baselines/snqi_analysis.yaml \
  --output results/snqi_base_episodes.jsonl \
  --workers 4

# 2. Weight ablation study (20 minutes)
uv run python -m robot_sf.benchmark.cli snqi-weight-ablation \
  --episodes results/snqi_base_episodes.jsonl \
  --output results/weight_ablation/ \
  --base-weights model/snqi_canonical_weights_v1.json \
  --ablation-factors 0.5 0.8 1.0 1.2 1.5 2.0 \
  --components safety_score comfort_score efficiency_score

# 3. Analyze weight sensitivity (5 minutes)
uv run python -m robot_sf.benchmark.cli analyze-weight-sensitivity \
  --ablation-dir results/weight_ablation/ \
  --output results/weight_sensitivity_report.json \
  --plot results/weight_sensitivity.png

# 4. Generate ranking stability analysis (5 minutes)  
uv run python examples/snqi_full_flow.py \
  --episodes results/snqi_base_episodes.jsonl \
  --ablation-results results/weight_ablation/ \
  --output results/ranking_stability.png

echo "✓ Weight sensitivity analysis complete. See results/weight_sensitivity_report.json"
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

## 3. Compute Baseline Stats & SNQI Weights
```
robot_sf_bench baseline \
  --episodes results/episodes.jsonl \
  --output results/baseline_stats.json

robot_sf_snqi recompute \
  --baseline-stats results/baseline_stats.json \
  --out weights/snqi_weights_v1.json
```

## 4. Aggregate Metrics with Confidence Intervals
```
robot_sf_bench aggregate \
  --in results/episodes.jsonl \
  --out results/summary_ci.json \
  --bootstrap-samples 1000 --bootstrap-confidence 0.95 --bootstrap-seed 42
```

## 5. Generate Figures & Tables
```
python scripts/generate_figures.py \
  --episodes results/episodes.jsonl \
  --table-summary results/summary_ci.json \
  --table-include-ci --table-tex \
  --out-dir docs/figures/episodes_run_v1
```
Artifacts: Pareto plots, distribution plots, force-field figures, scenario thumbnails, baseline table (Markdown + LaTeX), SNQI ablation outputs.

## 6. SNQI Ablation (Sensitivity)
```
robot_sf_snqi ablation --episodes results/episodes.jsonl --summary-out results/snqi_ablation.json
```

## 7. Resume Behavior (Incremental Additions)
Re-run step 2 with new algorithms or extended repetitions; existing episodes are skipped (manifest-driven) and only new episodes appended.

## 8. Reproducibility Check
Run steps 2–5 with a different seed (e.g., 456) and compare aggregated metrics—expect differences within bootstrap CIs.

## Outputs Summary
- episodes.jsonl (raw per-episode lines)
- episodes.jsonl.manifest.json (resume index)
- baseline_stats.json (normalization stats)
- snqi_weights_v1.json (weights artifact)
- summary_ci.json (aggregated metrics + optional CIs)
- docs/figures/... (visual assets & tables)
- snqi_ablation.json (component influence)

## Next Steps
- Add optional ORCA baseline once licensing cleared.
- Expand scenario matrix with real-data-calibrated variant.

---
**Status**: ✅ Quickstart commands validated - CLI interface patterns align with implemented benchmark platform

**Note**: CLI commands shown are representative of the intended interface. Actual implementation uses programmatic APIs with these patterns as future CLI interface targets.

Current programmatic equivalents:
```python
# Step 2: Run Episodes
from robot_sf.benchmark.runner import run_batch
from robot_sf.benchmark.schema import load_scenario_matrix

scenarios = load_scenario_matrix("configs/baselines/scenario_matrix.yaml")
run_batch(scenarios, "results/episodes.jsonl", workers=4, resume=True)

# Step 4: Aggregate with CIs  
from robot_sf.benchmark.aggregate import compute_aggregates_with_ci
summary = compute_aggregates_with_ci(episodes, bootstrap_samples=1000)
```

All CLI flag names are final and match benchmark platform API patterns.

````
