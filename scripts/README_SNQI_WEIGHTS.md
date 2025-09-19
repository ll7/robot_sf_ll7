# SNQI Weight Recomputation and Sensitivity Analysis

This directory contains scripts for recomputing Social Navigation Quality Index (SNQI) weights and performing sensitivity analysis. These scripts work with the SNQI implementation from the social navigation benchmark and support the median/p95 normalization strategy.

## Overview

The Social Navigation Quality Index (SNQI) is a composite metric that combines multiple navigation performance indicators:
- **Success rate**: Whether the robot reaches its goal
- **Time efficiency**: Normalized time to reach the goal
- **Safety metrics**: Collision counts and near misses
- **Comfort metrics**: Force exposure and social compliance
- **Smoothness**: Jerk and acceleration patterns

The SNQI formula is:
```
SNQI = w_success × success - w_time × time_norm - w_collisions × coll_norm 
       - w_near × near_norm - w_comfort × comfort_exposure 
       - w_force_exceed × force_norm - w_jerk × jerk_norm
```

Where normalized metrics use the median/p95 strategy:
```
norm_metric = (value - baseline_median) / (baseline_p95 - baseline_median)
```

## Scripts

### 1. `recompute_snqi_weights.py`

**Purpose**: Recompute SNQI weights using different strategies and compare normalization approaches.

**Usage**:
```bash
# Recompute weights using Pareto optimization
python scripts/recompute_snqi_weights.py \
    --episodes episodes.jsonl \
    --baseline baseline_stats.json \
    --strategy pareto \
    --output weights_optimized.json

# Compare all strategies
python scripts/recompute_snqi_weights.py \
    --episodes episodes.jsonl \
    --baseline baseline_stats.json \
    --compare-strategies \
    --compare-normalization \
    --output weights_comparison.json
```

**Available Strategies**:
- `default`: Balanced weights emphasizing safety and success
- `balanced`: Equal weights for all components
- `safety_focused`: Higher weights on collision and comfort metrics
- `efficiency_focused`: Higher weights on success and time metrics
- `pareto`: Pareto-optimal weights maximizing discriminative power and stability

**Input Files**:
- `episodes.jsonl`: Episode records from benchmark runs (JSONL format)
- `baseline_stats.json`: Baseline statistics with median/p95 values for normalization

**Output**:
- JSON file with optimized weights and performance statistics
- Strategy comparison results (if --compare-strategies used)
- Normalization strategy analysis (if --compare-normalization used)

### 2. `snqi_weight_optimization.py`

**Purpose**: Advanced weight optimization using grid search and differential evolution algorithms.

**Usage**:
```bash
# Run optimization with both methods
python scripts/snqi_weight_optimization.py \
    --episodes episodes.jsonl \
    --baseline baseline_stats.json \
    --output weights_optimized.json \
    --method both \
    --sensitivity

# Grid search only with custom resolution
python scripts/snqi_weight_optimization.py \
    --episodes episodes.jsonl \
    --baseline baseline_stats.json \
    --output weights_grid.json \
    --method grid \
    --grid-resolution 7
```

**Features**:
- Grid search optimization over weight space
- Differential evolution for continuous optimization
- Multi-objective optimization (ranking stability + discriminative power)
- Sensitivity analysis for weight perturbations
- Convergence tracking and method comparison

**Output**:
- Optimized weights with performance metrics
- Convergence information for each method
- Sensitivity analysis results (if --sensitivity used)

### 3. `snqi_sensitivity_analysis.py`

**Purpose**: Comprehensive sensitivity analysis of SNQI weights with visualizations.

**Usage**:
```bash
# Full sensitivity analysis with visualizations
python scripts/snqi_sensitivity_analysis.py \
    --episodes episodes.jsonl \
    --baseline baseline_stats.json \
    --weights weights.json \
    --output analysis_results/

# Quick analysis without plots
python scripts/snqi_sensitivity_analysis.py \
    --episodes episodes.jsonl \
    --baseline baseline_stats.json \
    --weights weights.json \
    --output analysis_results/ \
    --skip-visualizations
```

**Analysis Types**:
- **Weight sweep**: Individual weight sensitivity curves
- **Pairwise analysis**: 2D sensitivity heatmaps for weight pairs
- **Ablation study**: Impact of removing each weight component
- **Normalization comparison**: Effect of different normalization strategies

**Visualizations**:
- Weight sensitivity plots showing SNQI vs. weight value
- Heatmaps for pairwise weight interactions
- Bar charts for component importance ranking
- Normalization strategy comparison charts

**Output**:
- Detailed analysis results in JSON format
- Summary report with key findings
- Visualization plots (PNG files)

## Data Requirements

### Episode Data Format (JSONL)

Each line should contain an episode record:
```json
{
  "episode_id": "scenario_1--seed_42",
  "scenario_id": "scenario_1", 
  "seed": 42,
  "metrics": {
    "success": true,
    "time_to_goal_norm": 0.75,
    "collisions": 0,
    "near_misses": 2,
    "min_distance": 0.8,
    "comfort_exposure": 0.1,
    "force_exceed_events": 5,
    "jerk_mean": 0.3,
    "force_quantiles": {"q50": 1.2, "q90": 2.1, "q95": 2.8}
  },
  "scenario_params": {
    "algo": "baseline_sf",
    "density": "med"
  }
}
```

### Baseline Statistics Format (JSON)

Normalization parameters for each metric:
```json
{
  "collisions": {"med": 0.0, "p95": 2.0},
  "near_misses": {"med": 1.0, "p95": 5.0},
  "force_exceed_events": {"med": 2.0, "p95": 15.0},
  "jerk_mean": {"med": 0.2, "p95": 1.0}
}
```

### Weight Configuration Format (JSON)

Weight values for SNQI components:
```json
{
  "w_success": 2.0,
  "w_time": 1.0,
  "w_collisions": 2.0,
  "w_near": 1.0,
  "w_comfort": 1.5,
  "w_force_exceed": 1.5,
  "w_jerk": 0.5
}
```

## Dependencies

Required Python packages:
- `numpy`: Numerical computations
- `scipy`: Optimization and statistics
- `matplotlib`: Plotting (for sensitivity analysis)
- `seaborn`: Statistical visualizations
- `pandas`: Data manipulation

Install with:
```bash
pip install numpy scipy matplotlib seaborn pandas
```

## Example Workflow

1. **Generate baseline statistics** (using benchmark tools):
   ```bash
   robot_sf_bench baseline --matrix scenarios.yaml --out baseline_stats.json
   ```

2. **Run episodes and collect data**:
   ```bash
   robot_sf_bench run --matrix scenarios.yaml --out episodes.jsonl
   ```

3. **Recompute optimal weights**:
   ```bash
   python scripts/snqi_weight_optimization.py \
       --episodes episodes.jsonl \
       --baseline baseline_stats.json \
       --output weights_optimized.json \
       --method both --sensitivity
   ```

4. **Perform detailed sensitivity analysis**:
   ```bash
   python scripts/snqi_sensitivity_analysis.py \
       --episodes episodes.jsonl \
       --baseline baseline_stats.json \
       --weights weights_optimized.json \
       --output sensitivity_analysis/
   ```

5. **Compare different strategies**:
   ```bash
   python scripts/recompute_snqi_weights.py \
       --episodes episodes.jsonl \
       --baseline baseline_stats.json \
       --compare-strategies \
       --compare-normalization \
       --output strategy_comparison.json
   ```

## Interpreting Results

### Weight Optimization Results

- **Objective Value**: Higher values indicate better discriminative power and ranking stability
- **Ranking Stability**: Correlation measure of ranking consistency (closer to 1.0 is better)
- **Convergence Info**: Details about optimization success and iterations

### Sensitivity Analysis Results

- **Most Sensitive Weights**: Components with highest impact on SNQI rankings
- **Ranking Correlation**: How much rankings change with weight perturbations
- **Normalization Impact**: Effect of different percentile strategies

### Key Metrics to Monitor

1. **Discriminative Power**: SNQI should have sufficient variance to distinguish between algorithms
2. **Ranking Stability**: Small weight changes shouldn't drastically alter rankings
3. **Component Balance**: No single component should dominate the index
4. **Normalization Robustness**: Results should be relatively stable across normalization strategies

## Troubleshooting

**Common Issues**:

1. **Missing metrics in episodes**: Scripts handle missing values gracefully, but ensure core metrics (success, time_to_goal_norm, collisions) are present

2. **Insufficient episode data**: Need at least 10-20 episodes for meaningful optimization

3. **Baseline statistics errors**: Ensure baseline_stats.json contains median/p95 values for normalized metrics

4. **Visualization failures**: Install matplotlib/seaborn or use --skip-visualizations flag

5. **Memory issues with large datasets**: Reduce grid resolution or use sampling for large episode collections

**Performance Tips**:
- Use smaller grid resolutions for initial testing
- Enable only necessary analyses to reduce computation time
- Consider using differential evolution over grid search for large weight spaces

## Integration with Benchmark

These scripts are designed to work with the social navigation benchmark implementation. The SNQI computation follows the same formula and normalization strategy as implemented in `robot_sf/benchmark/metrics.py`.

For integration with the benchmark pipeline:
1. Use `robot_sf_bench baseline` to generate baseline statistics
2. Use `robot_sf_bench run` to collect episode data  
3. Apply these scripts to optimize weights
4. Use optimized weights in subsequent benchmark runs via the `snqi_weights` parameter