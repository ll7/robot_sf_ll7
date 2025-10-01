# Quickstart: Fix Benchmark Placeholder Outputs

**Date**: 2025-09-24
**Feature**: 133-all-generated-plots

## Overview
This feature fixes the benchmark system to generate real plots and videos instead of placeholders. After implementation, benchmark runs will produce actual visualizations of performance data and simulation replays.

## Quick Start

### 1. Run a Complete Benchmark
```bash
# Run the full classic benchmark with real visualizations
uv run python scripts/classic_benchmark_full.py
```

### 2. Check the Results
After completion, check the output directory:
```
results/full_classic_run_YYYY-MM-DD_HH-MM-SS/
├── episodes.jsonl          # Raw benchmark data
├── plots/                  # Real PDF plots (not placeholders!)
│   ├── metrics_distribution.pdf
│   ├── trajectory_comparison.pdf
│   └── performance_summary.pdf
└── videos/                 # Real MP4 videos (not dummies!)
    ├── scenario_001_socialforce.mp4
    └── scenario_002_random.mp4
```

### 3. Verify Real Outputs
The system now validates that outputs are real:
- **Plots**: Show actual metric distributions, not placeholder images
- **Videos**: Display actual robot navigation scenarios, not dummy footage
- **Validation**: Automatic checks ensure outputs contain real data

## What Changed

### Before (Placeholders)
- Plots: Generic placeholder PDFs with "TODO" messages
- Videos: Static dummy MP4 files
- No validation of output quality

### After (Real Visualizations)
- Plots: Statistical plots of actual benchmark metrics
- Videos: Rendered replays of simulation episodes
- Validation ensures outputs are meaningful

## Troubleshooting

### Missing Dependencies
If you see errors about missing packages:
```bash
# Install visualization dependencies
uv add matplotlib moviepy
```

### Videos Not Generating
Video rendering requires:
- MoviePy library
- Working environment factory functions
- Trajectory data in episode records

### Plots Not Generating
Plot generation requires:
- Matplotlib library
- Valid metrics in episode JSONL files
- Writable output directory

## Advanced Usage

### Custom Visualization Scripts
```python
from robot_sf.benchmark.visualization import generate_benchmark_plots, generate_benchmark_videos

# Generate only plots for specific scenarios
plots = generate_benchmark_plots(
    episodes_path="results/episodes.jsonl",
    output_dir="results/",
    scenario_filter="classic_interactions"
)

# Generate videos with custom settings
videos = generate_benchmark_videos(
    episodes_path="results/episodes.jsonl",
    output_dir="results/",
    fps=24,
    max_duration=15.0
)
```

### Validation
```python
from robot_sf.benchmark.visualization import validate_visual_artifacts

# Check if generated artifacts are real
result = validate_visual_artifacts(artifacts)
if result.passed:
    print("All visualizations are real!")
else:
    print(f"Found {len(result.failed_artifacts)} placeholder artifacts")
```

## Integration with Existing Workflows

The changes are backward compatible:
- Existing benchmark scripts work unchanged
- New visualization functions are optional additions
- Fallback to placeholders if dependencies missing
- Clear error messages guide troubleshooting

## Performance Notes

- **Plot generation**: < 30 seconds for typical benchmark sizes
- **Video generation**: < 60 seconds per scenario
- **Memory usage**: Scales with episode count and trajectory length
- **Dependencies**: Matplotlib and MoviePy add minimal overhead

## Next Steps

1. Run your benchmark with `scripts/classic_benchmark_full.py`
2. Open the generated PDFs in `results/plots/`
3. Play the MP4 videos in `results/videos/`
4. Verify they show real benchmark data, not placeholders