# SNQI Weight Recomputation - Quick Start

This directory contains scripts for recomputing and analyzing Social Navigation Quality Index (SNQI) weights based on the median/p95 normalization strategy.

## Quick Test

Run the validation script to verify everything works:

```bash
python scripts/validate_snqi_scripts.py
```

## Demo Workflow

See a complete example with generated data:

```bash
python scripts/example_snqi_workflow.py
```

## Basic Usage

### 1. Simple Weight Recomputation

```bash
# Compare all weight strategies
python scripts/recompute_snqi_weights.py \
    --episodes your_episodes.jsonl \
    --baseline your_baseline_stats.json \
    --compare-strategies \
    --output weight_comparison.json
```

### 2. Advanced Optimization

```bash
# Run differential evolution optimization with sensitivity analysis
python scripts/snqi_weight_optimization.py \
    --episodes your_episodes.jsonl \
    --baseline your_baseline_stats.json \
    --output optimized_weights.json \
    --method evolution \
    --sensitivity
```

### 3. Detailed Sensitivity Analysis

```bash
# Full sensitivity analysis with visualizations
python scripts/snqi_sensitivity_analysis.py \
    --episodes your_episodes.jsonl \
    --baseline your_baseline_stats.json \
    --weights optimized_weights.json \
    --output sensitivity_results/
```

## Expected Output

After running the scripts, you'll have:

- **Optimized weights**: JSON file with recommended SNQI weights
- **Performance metrics**: Statistics showing how well different strategies perform
- **Sensitivity analysis**: Detailed analysis of how weight changes affect rankings
- **Visualizations**: Plots showing weight sensitivity and interactions (if matplotlib available)

## Data Format

Your input files should follow the benchmark format:

- **episodes.jsonl**: One episode record per line with metrics
- **baseline_stats.json**: Median/p95 values for normalization

See `README_SNQI_WEIGHTS.md` for detailed format specifications.

## Dependencies

Required:
- `numpy` - for numerical computations
- `scipy` - for optimization algorithms

Optional (for visualizations):
- `matplotlib` - for plotting
- `seaborn` - for statistical plots
- `pandas` - for data manipulation

Install with:
```bash
pip install numpy scipy matplotlib seaborn pandas
```

## Key Features

- **Multiple Strategies**: Default, balanced, safety-focused, efficiency-focused, Pareto-optimal
- **Normalization Support**: Median/p95 strategy with comparisons to alternatives
- **Optimization Methods**: Grid search and differential evolution
- **Sensitivity Analysis**: Weight sweep, pairwise interactions, ablation studies
- **Ranking Stability**: Measures how robust rankings are to weight changes
- **Discriminative Power**: Ensures SNQI can distinguish between different algorithms

The scripts automatically handle missing data, provide fallbacks for missing dependencies, and generate detailed reports with actionable insights.