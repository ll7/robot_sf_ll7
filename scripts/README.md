# Scripts Directory - Quick Overview

This directory contains all executable scripts for the Robot SF project, organized by functional area. This guide provides a quick reference to help you find the right tool for your task.

## Table of Contents

- [Quick Navigation by Task](#quick-navigation-by-task)
- [Directory Structure](#directory-structure)
- [Core Scripts](#core-scripts)
- [Training Scripts](#training-scripts)
- [Research & Analysis](#research--analysis)
- [Validation & Testing](#validation--testing)
- [Tools & Utilities](#tools--utilities)
- [Coverage & Performance](#coverage--performance)
- [Legacy & Debugging](#legacy--debugging)

## Quick Navigation by Task

### I want to...

- **Train a robot policy** → [`training_ppo.py`](#training_ppopy) or [`training/`](#training-directory)
- **Run benchmarks** → [`classic_benchmark_full.py`](#classic_benchmark_fullpy) or [`benchmark02.py`](#benchmark02py)
- **Analyze results** → [`research/`](#research-directory) or [`generate_figures.py`](#generate_figurespy)
- **Validate changes** → [`validation/`](#validation-directory)
- **Compare training runs** → [`tools/compare_training_runs.py`](#toolscompare_training_runspy)
- **Preview scenario trajectories** → [`tools/preview_scenario_trajectories.py`](#toolspreview_scenario_trajectoriespy)
- **Work with SNQI metrics** → [`SNQI scripts`](#snqi-weight-scripts)
- **Check performance** → [`validation/performance_smoke_test.py`](#validationperformance_smoke_testpy)
- **Migrate artifacts** → [`tools/migrate_artifacts.py`](#toolsmigrate_artifactspy)

## Directory Structure

```
scripts/
├── README.md                          # This file
├── QUICK_START.md                     # SNQI quick start guide
├── README_SNQI_WEIGHTS.md             # SNQI weight documentation
│
├── training/                          # Training workflows
│   ├── train_expert_ppo.py           # Expert PPO training
│   ├── collect_expert_trajectories.py # Trajectory collection
│   ├── pretrain_from_expert.py       # Behavioral cloning pre-training
│   └── train_ppo_with_pretrained_policy.py # PPO fine-tuning
│
├── research/                          # Research & analysis
│   ├── generate_report.py            # Research report generation
│   └── compare_ablations.py          # Ablation study comparison
│
├── validation/                        # Testing & validation
│   ├── performance_smoke_test.py     # Performance validation
│   ├── run_examples_smoke.py         # Example script smoke tests
│   ├── verify_maps.py                # Map file validation
│   └── test_*.sh                     # Shell-based validation tests
│
├── tools/                             # Utilities & helpers
│   ├── run_tracker_cli.py            # Run tracking CLI
│   ├── compare_training_runs.py      # Training comparison
│   ├── preview_scenario_trajectories.py # Scenario trajectory preview helper
│   ├── migrate_artifacts.py          # Artifact migration
│   ├── check_artifact_root.py        # Artifact policy guard
│   └── validate_report.py            # Report validation
│
├── coverage/                          # Coverage tools
│   ├── open_coverage_report.py       # Open HTML coverage report
│   └── compare_coverage.py           # Coverage comparison
│
├── telemetry/                         # Performance telemetry
│   └── run_perf_tests.py             # Telemetry wrapper for perf tests
│
└── perf/                              # Performance baselines
    └── baseline_factory_creation.py   # Factory creation timing
```

## Core Scripts

### Training Scripts

#### `training_ppo.py`
**Purpose**: Main PPO training entry point  
**Usage**:
```bash
uv run python scripts/training_ppo.py
```
**Details**: Standard PPO training workflow using StableBaselines3

#### `training_a2c.py`
**Purpose**: A2C algorithm training  
**Usage**:
```bash
uv run python scripts/training_a2c.py
```
**Details**: Alternative training algorithm (A2C)

#### `training_ped_ppo.py`
**Purpose**: PPO training for pedestrian environments  
**Usage**:
```bash
uv run python scripts/training_ped_ppo.py
```

#### `wandb_ppo_training.py`
**Purpose**: PPO training with Weights & Biases logging  
**Usage**:
```bash
uv run python scripts/wandb_ppo_training.py
```
**Details**: See `docs/wandb.md` for setup

#### `multi_extractor_training.py`
**Purpose**: Compare multiple feature extractors with reproducible summaries  
**Usage**:
```bash
uv run python scripts/multi_extractor_training.py
```
**Details**: Orchestrates PPO training runs for configured feature extractors, captures hardware metadata

### Benchmark Scripts

#### `classic_benchmark_full.py`
**Purpose**: CLI entry for Full Classic Interaction Benchmark  
**Usage**:
```bash
uv run python scripts/classic_benchmark_full.py
```
**Details**: Expanded parser with full benchmark flags

#### `benchmark02.py`
**Purpose**: Performance benchmarking with metrics collection  
**Usage**:
```bash
DISPLAY= MPLBACKEND=Agg uv run python scripts/benchmark02.py
```
**Expected**: ~22 steps/second, ~45ms per step

#### `benchmark_workers.py`
**Purpose**: Parallel benchmark execution with worker processes  
**Usage**:
```bash
uv run python scripts/benchmark_workers.py
```

#### `run_social_navigation_benchmark.py`
**Purpose**: Complete social navigation benchmark runner  
**Usage**:
```bash
uv run python scripts/run_social_navigation_benchmark.py
```
**Details**: Executes full benchmark suite

### Analysis & Visualization

#### `generate_figures.py`
**Purpose**: Generate benchmark figures from episodes JSONL  
**Usage**:
```bash
uv run python scripts/generate_figures.py
```
**Output**: Pareto plots, distributions, baseline comparison tables, scenario thumbnails

#### `ranking_table.py`
**Purpose**: Generate ranking tables by metric from benchmark episodes  
**Usage**:
```bash
uv run python scripts/ranking_table.py
```
**Details**: Aggregates episode metrics per group, builds ranking tables sorted by chosen metric

#### `analyze_feature_extractors.py`
**Purpose**: Statistical analysis for feature extractor comparison  
**Usage**:
```bash
uv run python scripts/analyze_feature_extractors.py
```

#### `seed_variance.py`
**Purpose**: Compute SNQI seed variance from benchmark episodes  
**Usage**:
```bash
uv run python scripts/seed_variance.py
```
**Details**: Groups episodes and reports variability across seed means

### SNQI Weight Scripts

#### `recompute_snqi_weights.py`
**Purpose**: Recompute SNQI weights using different strategies  
**Usage**:
```bash
# Simple weight recomputation
uv run python scripts/recompute_snqi_weights.py \
    --episodes episodes.jsonl \
    --baseline baseline_stats.json \
    --compare-strategies \
    --output weight_comparison.json
```
**Details**: See `README_SNQI_WEIGHTS.md` for detailed documentation

#### `snqi_weight_optimization.py`
**Purpose**: Advanced SNQI weight optimization with differential evolution  
**Usage**:
```bash
uv run python scripts/snqi_weight_optimization.py \
    --episodes episodes.jsonl \
    --baseline baseline_stats.json \
    --output optimized_weights.json \
    --method evolution \
    --sensitivity
```

#### `snqi_sensitivity_analysis.py`
**Purpose**: Full sensitivity analysis with visualizations  
**Usage**:
```bash
uv run python scripts/snqi_sensitivity_analysis.py \
    --episodes episodes.jsonl \
    --baseline baseline_stats.json \
    --weights optimized_weights.json \
    --output sensitivity_results/
```

#### `validate_snqi_scripts.py`
**Purpose**: Verify SNQI scripts work correctly  
**Usage**:
```bash
uv run python scripts/validate_snqi_scripts.py
```

#### `example_snqi_workflow.py`
**Purpose**: Complete SNQI workflow example with generated data  
**Usage**:
```bash
uv run python scripts/example_snqi_workflow.py
```

## Training Directory

Imitation learning pipeline scripts (feature 001-ppo-imitation-pretrain):

### `training/train_expert_ppo.py`
**Purpose**: Expert PPO training workflow entry point  
**Usage**:
```bash
uv run python scripts/training/train_expert_ppo.py --config configs/training/ppo_imitation/expert_ppo.yaml
```
**Details**: Loads unified config, orchestrates PPO expert training, evaluates policy, persists manifests

### `training/collect_expert_trajectories.py`
**Purpose**: Trajectory collection for imitation learning  
**Usage**:
```bash
uv run python scripts/training/collect_expert_trajectories.py --dataset-id expert_v1 --policy-id ppo_expert_v1 --episodes 200
```
**Details**: Records episodes using expert policy, dumps to NPZ dataset, validates artifact

### `training/pretrain_from_expert.py`
**Purpose**: Behavioral cloning pre-training from expert trajectories  
**Usage**:
```bash
uv run python scripts/training/pretrain_from_expert.py --config configs/training/ppo_imitation/bc_pretrain.yaml
```
**Details**: Trains PPO policy using BC on expert trajectories

### `training/train_ppo_with_pretrained_policy.py`
**Purpose**: PPO fine-tuning from pre-trained checkpoint  
**Usage**:
```bash
uv run python scripts/training/train_ppo_with_pretrained_policy.py --config configs/training/ppo_imitation/ppo_finetune.yaml
```
**Details**: Continues training with PPO from warm-start policy

## Research Directory

### `research/generate_report.py`
**Purpose**: Research report generation from tracked runs  
**Usage**:
```bash
uv run python scripts/research/generate_report.py
```
**Details**: Loads tracker manifests, generates comprehensive reports

### `research/compare_ablations.py`
**Purpose**: CLI for ablation study comparison (User Story 4)  
**Usage**:
```bash
uv run python scripts/research/compare_ablations.py \
    --config configs/research/example_ablation.yaml \
    --experiment-name BC_Ablation \
    --seeds 42 43 44 \
    --threshold 40.0 \
    --output output/research_reports/ablation_bc
```
**Details**: Runs ablation matrix and generates comparison report

## Validation Directory

### `validation/performance_smoke_test.py`
**Purpose**: Performance baseline validation  
**Usage**:
```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/performance_smoke_test.py
```
**Details**: Validates performance against baseline targets

### `validation/run_examples_smoke.py`
**Purpose**: Execute smoke tests for all examples  
**Usage**:
```bash
# Dry run (show what would execute)
uv run python scripts/validation/run_examples_smoke.py --dry-run

# Performance tests only
uv run python scripts/validation/run_examples_smoke.py --perf-tests-only

# Full smoke harness
uv run python scripts/validation/run_examples_smoke.py
```

### `validation/validate_examples_manifest.py`
**Purpose**: Ensure examples manifest is complete and aligned  
**Usage**:
```bash
uv run python scripts/validation/validate_examples_manifest.py
```

### `validation/verify_maps.py`
**Purpose**: Map file validation  
**Usage**:
```bash
# CI mode
uv run python scripts/validation/verify_maps.py --scope ci --mode ci --output output/validation/map_verification.json

# Full validation
uv run python scripts/validation/verify_maps.py
```

### Shell Validation Tests

Quick environment validation scripts:

- `validation/test_basic_environment.sh` - Basic environment sanity check
- `validation/test_model_prediction.sh` - Model loading and inference test
- `validation/test_complete_simulation.sh` - Full simulation run test
- `validation/test_classic_benchmark_full.sh` - Classic benchmark validation
- `validation/test_coverage_collection.sh` - Coverage collection test
- `validation/test_research_report_smoke.sh` - Research report smoke test

### `validation/run_research_quickstart.sh`
**Purpose**: Execute research pipeline quickstart  
**Usage**:
```bash
bash scripts/validation/run_research_quickstart.sh
```

### `validation/render_examples_readme.py`
**Purpose**: Generate/update examples README from manifest  
**Usage**:
```bash
uv run python scripts/validation/render_examples_readme.py
```

### `validation/playback_trajectory.py`
**Purpose**: Trajectory visualization and playback  
**Usage**:
```bash
uv run python scripts/validation/playback_trajectory.py
```

### `validation/performance_research_report.py`
**Purpose**: Research report performance validation  
**Usage**:
```bash
uv run python scripts/validation/performance_research_report.py
```

### `validation/verify_sf_implementation.py`
**Purpose**: Social Force implementation verification  
**Usage**:
```bash
uv run python scripts/validation/verify_sf_implementation.py
```

## Tools Directory

### `tools/run_tracker_cli.py`
**Purpose**: Command-line helper for run-tracking telemetry  
**Usage**:
```bash
# Show run status
uv run python scripts/tools/run_tracker_cli.py status <run_id>

# Watch run progress
uv run python scripts/tools/run_tracker_cli.py watch <run_id> --interval 1.0

# List recent runs
uv run python scripts/tools/run_tracker_cli.py list --status running

# Generate summary
uv run python scripts/tools/run_tracker_cli.py summary <run_id>

# Export report
uv run python scripts/tools/run_tracker_cli.py export <run_id> --format markdown --output output/run-tracker/summaries/<run_id>.md

# Enable TensorBoard
uv run python scripts/tools/run_tracker_cli.py enable-tensorboard <run_id> --logdir output/run-tracker/tb/<run_id>

# Run performance tests
uv run python scripts/tools/run_tracker_cli.py perf-tests \
    --scenario configs/validation/minimal.yaml \
    --output output/run-tracker/perf-tests/latest \
    --num-resets 5
```

### `tools/preview_scenario_trajectories.py`
**Purpose**: Preview single-pedestrian trajectories on top of scenario maps  
**Usage**:
```bash
uv run python scripts/tools/preview_scenario_trajectories.py \
  --scenario configs/scenarios/classic_interactions.yaml \
  --scenario-id classic_head_on_corridor
```
**Details**: Writes a PNG under `output/preview/scenario_trajectories/` by default. Use `MPLBACKEND=Agg` for headless runs.

### `tools/render_scenario_videos.py`
**Purpose**: Render MP4 videos for every scenario in a scenario matrix  
**Usage**:
```bash
uv run python scripts/tools/render_scenario_videos.py \
  --scenario configs/scenarios/francis2023.yaml \
  --all
```
**Details**: Writes videos under a timestamped folder in `output/recordings/` and saves a `manifest.json`. Use `--policy ppo --model-path model/run_023.zip` to drive the robot with the defensive PPO policy.

### `tools/compare_training_runs.py`
**Purpose**: Comparison tool for analyzing training runs  
**Usage**:
```bash
uv run python scripts/tools/compare_training_runs.py
```
**Details**: Computes sample-efficiency metrics, convergence timings

### `tools/migrate_artifacts.py`
**Purpose**: Migration helper to consolidate legacy artifacts  
**Usage**:
```bash
# Run migration
uv run python scripts/tools/migrate_artifacts.py

# Or use console entry point
uv run robot-sf-migrate-artifacts
```
**Details**: Consolidates `results/`, `recordings/`, `htmlcov/`, `coverage.json` into `output/`

### `tools/check_artifact_root.py`
**Purpose**: Guard script ensuring artifacts respect canonical `output/` root  
**Usage**:
```bash
uv run python scripts/tools/check_artifact_root.py
```
**Details**: Fails fast when new top-level artifacts appear

### `tools/validate_report.py`
**Purpose**: Report validation helper  
**Usage**:
```bash
uv run python scripts/tools/validate_report.py
```
**Details**: Checks required files and directories exist

## Coverage & Performance

### `coverage/open_coverage_report.py`
**Purpose**: Open HTML coverage report in browser  
**Usage**:
```bash
uv run python scripts/coverage/open_coverage_report.py
```
**Details**: Automatically opens `output/coverage/htmlcov/index.html`

### `coverage/compare_coverage.py`
**Purpose**: Coverage comparison between runs  
**Usage**:
```bash
uv run python scripts/coverage/compare_coverage.py
```

### `telemetry/run_perf_tests.py`
**Purpose**: Telemetry-aware wrapper for performance smoke test  
**Usage**:
```bash
uv run python scripts/telemetry/run_perf_tests.py
```
**Details**: Invokes `performance_smoke_test.py` and persists structured results

### `perf/baseline_factory_creation.py`
**Purpose**: Baseline environment factory creation timing  
**Usage**:
```bash
uv run python scripts/perf/baseline_factory_creation.py
```
**Details**: Measures factory creation performance

## Legacy & Debugging

### Debugging Scripts

- `debug_random_policy.py` - Test environment with random policy
- `debug_trained_policy.py` - Test with trained policy checkpoint
- `debug_ped_policy.py` - Debug pedestrian policy
- `debug_ped_discrete.py` - Simulate hardcoded deterministic policy with four actions

### Data Conversion

#### `convert_pickle_to_jsonl.py`
**Purpose**: Convert legacy multi-episode pickle files to per-episode JSONL  
**Usage**:
```bash
uv run python scripts/convert_pickle_to_jsonl.py
```

### Recording & Playback

#### `demo_jsonl_recording.py`
**Purpose**: Demonstration of JSONL recording and playback functionality  
**Usage**:
```bash
uv run python scripts/demo_jsonl_recording.py
```

#### `play_recordings.py`
**Purpose**: Playback recorded episodes  
**Usage**:
```bash
uv run python scripts/play_recordings.py
```

### Other Utilities

#### `failure_extractor.py`
**Purpose**: Extract worst episodes by chosen metric from episodes JSONL  
**Usage**:
```bash
uv run python scripts/failure_extractor.py
```
**Details**: Selects top-k worst episodes by dotted metric path

#### `collect_slow_tests.py`
**Purpose**: Parse `pytest --durations=N` output into structured JSON  
**Usage**:
```bash
pytest --durations=20 | uv run python scripts/collect_slow_tests.py > slow_tests.json
```

#### `compare_slow_tests.py`
**Purpose**: Compare before/after slow test JSON captures  
**Usage**:
```bash
uv run python scripts/compare_slow_tests.py --before progress/slow_tests_pre.json --after progress/slow_tests_post.json
```

#### `evaluate.py`
**Purpose**: Policy evaluation script  
**Usage**:
```bash
uv run python scripts/evaluate.py
```

#### `hparam_opt.py`
**Purpose**: Hyperparameter optimization with Optuna  
**Usage**:
```bash
uv run python scripts/hparam_opt.py
```

#### `benchmark_repro_check.py`
**Purpose**: Create minimal test scenario for reproducibility testing  
**Usage**:
```bash
uv run python scripts/benchmark_repro_check.py
```

#### `generate_video_contact_sheet.py`
**Purpose**: Placeholder for generating thumbnail contact sheets from episode videos  
**Usage**:
```bash
uv run python scripts/generate_video_contact_sheet.py
```
**Details**: Implementation deferred until video artifact pipeline stabilizes

#### `update_svg_viewbox.py`
**Purpose**: Update viewBox attribute of SVG files  
**Usage**:
```bash
uv run python scripts/update_svg_viewbox.py
```

#### `scale_svgs_to_50m.py`
**Purpose**: Scale SVG coordinate values  
**Usage**:
```bash
uv run python scripts/scale_svgs_to_50m.py
```

#### `run_classic_interactions.py`
**Purpose**: Run classic interaction scenarios  
**Usage**:
```bash
uv run python scripts/run_classic_interactions.py
```

### PPO Training Subdirectory

Legacy training experiments in `PPO_training/`:

- `train_ppo_punish_action.py` - PPO with action punishment

## Common Patterns

### Environment Variables

Many scripts support headless execution:
```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python scripts/<script>.py
```

### Artifact Override

Override artifact destination:
```bash
export ROBOT_SF_ARTIFACT_ROOT=/path/to/custom/output
uv run python scripts/<script>.py
```

### Common Flags

Most scripts support standard flags:
- `--help` - Show usage information
- `--config <path>` - Specify configuration file
- `--output <path>` - Specify output directory
- `--debug` - Enable debug logging
- `--log-level DEBUG|INFO|WARNING|ERROR` - Set log level

## Quick Start Workflows

### 1. Train a Robot Policy
```bash
# Standard PPO training
uv run python scripts/training_ppo.py

# Or imitation learning pipeline
uv run python examples/advanced/16_imitation_learning_pipeline.py
```

### 2. Run Benchmarks
```bash
# Full classic benchmark
uv run python scripts/classic_benchmark_full.py

# Or quick performance check
DISPLAY= MPLBACKEND=Agg uv run python scripts/benchmark02.py
```

### 3. Analyze Results
```bash
# Generate figures
uv run python scripts/generate_figures.py

# Create ranking table
uv run python scripts/ranking_table.py

# Research report
uv run python scripts/research/generate_report.py
```

### 4. Validate Changes
```bash
# Run validation suite
./scripts/validation/test_basic_environment.sh
./scripts/validation/test_model_prediction.sh
./scripts/validation/test_complete_simulation.sh

# Performance smoke test
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/performance_smoke_test.py

# Artifact guard
uv run python scripts/tools/check_artifact_root.py
```

### 5. SNQI Weight Analysis
```bash
# Quick test
uv run python scripts/validate_snqi_scripts.py

# Demo workflow
uv run python scripts/example_snqi_workflow.py

# Optimization
uv run python scripts/snqi_weight_optimization.py \
    --episodes episodes.jsonl \
    --baseline baseline_stats.json \
    --output optimized_weights.json
```

## Related Documentation

- **Development Guide**: `docs/dev_guide.md` - Primary development reference
- **SNQI Quick Start**: `scripts/QUICK_START.md` - SNQI recomputation guide
- **SNQI Weights**: `scripts/README_SNQI_WEIGHTS.md` - Detailed SNQI documentation
- **Imitation Learning**: `docs/imitation_learning_pipeline.md` - Pipeline guide
- **Examples**: `examples/README.md` - Curated example scripts
- **Artifact Policy**: `specs/243-clean-output-dirs/quickstart.md` - Artifact management

## Contributing

When adding new scripts:

1. **Follow naming conventions**: Use descriptive snake_case names
2. **Add docstrings**: Include purpose, usage, and details at the top of the file
3. **Update this README**: Add entry in appropriate section
4. **Use factory patterns**: Prefer `make_robot_env()` over direct instantiation
5. **Follow artifact policy**: Write outputs to `output/` subdirectories
6. **Add validation**: Include smoke test or validation script if applicable
7. **Document dependencies**: Note any special requirements or setup

## Support

For questions or issues:
- Check the development guide: `docs/dev_guide.md`
- Review examples: `examples/README.md`
- File an issue on GitHub

---

**Last Updated**: 2025-11-24  
**Maintained By**: Robot SF Development Team
