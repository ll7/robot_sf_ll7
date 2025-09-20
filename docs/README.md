# Robot SF Documentation

Welcome to the Robot SF documentation! This directory contains comprehensive guides and references for using and developing with the Robot SF simulation framework.

## 🚀 Social Navigation Benchmark Platform (Complete)

**The Social Navigation Benchmark Platform is now fully operational!** 

### Quick Start
- **[Complete Quickstart Guide](../specs/120-social-navigation-benchmark-plan/quickstart.md)** - Step-by-step experiment execution, visualization, and interpretation
- **[CLI Reference](./dev/issues/social-navigation-benchmark/README.md)** - All 15 CLI subcommands with examples
- **Implementation Status**: All major features complete, 108 tests passing

### Core Capabilities
- **Episode Runner**: Parallel execution with resume functionality and deterministic seeding
- **Metrics Suite**: SNQI composite index with component breakdown and weight recomputation
- **Baseline Interface**: Unified PlannerProtocol for SocialForce, PPO, Random planners  
- **Statistical Analysis**: Bootstrap confidence intervals and robust aggregation
- **Figure Orchestrator**: Distribution plots, Pareto frontiers, force fields, thumbnails, tables
- **CLI Tools**: 15 subcommands covering full experiment workflow

### Ready-to-Use Workflows
1. **Quick Assessment** (~15 min): Compare robot policies against baselines
2. **Research Study** (~2-4 hours): Multi-parameter analysis with publication figures
3. **Weight Sensitivity** (~45 min): Analyze SNQI component importance

**Start Here**: [specs/120-social-navigation-benchmark-plan/quickstart.md](../specs/120-social-navigation-benchmark-plan/quickstart.md)

---

- [🚀 Social Navigation Benchmark Platform (Complete)](#-social-navigation-benchmark-platform-complete)
  - [Quick Start](#quick-start)
  - [Core Capabilities](#core-capabilities)
  - [Ready-to-Use Workflows](#ready-to-use-workflows)
- [📚 Documentation Index](#-documentation-index)
  - [🏗️ Architecture \& Development](#️-architecture--development)
  - [🎮 Simulation \& Environment](#-simulation--environment)
  - [📊 Analysis \& Tools](#-analysis--tools)
    - [Social Navigation Benchmark (Overview)](#social-navigation-benchmark-overview)
    - [Figures naming and outputs](#figures-naming-and-outputs)
    - [LaTeX Table Embedding (SNQI / Benchmark Tables)](#latex-table-embedding-snqi--benchmark-tables)
  - [Per-Test Performance Budget](#per-test-performance-budget)
  - [⚙️ Setup \& Configuration](#️-setup--configuration)
  - [📈 Pedestrian Metrics](#-pedestrian-metrics)
  - [📁 Media Resources](#-media-resources)
- [🚀 Quick Start Guides](#-quick-start-guides)
  - [New Environment Architecture (Recommended)](#new-environment-architecture-recommended)
  - [Legacy Pattern (Still Supported)](#legacy-pattern-still-supported)
- [🎯 Key Features](#-key-features)
  - [Environment System](#environment-system)
  - [Simulation Capabilities](#simulation-capabilities)
  - [Training \& Analysis](#training--analysis)
- [📖 Documentation Highlights](#-documentation-highlights)
  - [🆕 Latest Updates](#-latest-updates)
  - [📋 Migration Status](#-migration-status)
- [🔗 External Links](#-external-links)
- [🤝 Contributing](#-contributing)
- [📞 Support](#-support)

## 📚 Documentation Index

### 🏗️ Architecture & Development
- **[Environment Refactoring](./refactoring/)** - **NEW**: Complete guide to the refactored environment architecture
  - [Deployment Status](./refactoring/DEPLOYMENT_READY.md) - Current implementation status
  - [Refactoring Plan](./refactoring/refactoring_plan.md) - Technical architecture details
  - [Migration Guide](./refactoring/migration_guide.md) - Step-by-step migration instructions
  - [Implementation Summary](./refactoring/refactoring_summary.md) - What was accomplished
  - [Migration Report](./refactoring/migration_report.md) - Automated codebase analysis

### 🎮 Simulation & Environment
- [**Simulation View**](./SIM_VIEW.md) - Visualization and rendering system
- [**Map Editor Usage**](./MAP_EDITOR_USAGE.md) - Creating and editing simulation maps
- [**SVG Map Editor**](./SVG_MAP_EDITOR.md) - SVG-based map creation tools
 - **Classic Interaction Scenario Pack** (configs/scenarios/classic_interactions.yaml) – Canonical crossing, head‑on, overtaking, bottleneck, doorway, merging, T‑intersection, and group crossing archetypes for benchmark coverage.

### 📊 Analysis & Tools  
 - [**SNQI Weight Tooling**](./snqi-weight-tools/README.md) - User guide for recomputing, optimizing, and analyzing SNQI weights
 - [**SNQI Figures (orchestrator usage)**](../examples/README.md) - Generate SNQI-augmented figures from existing episodes
 - [**Full SNQI Flow (episodes → baseline → figures)**](../examples/snqi_full_flow.py) - End-to-end reproducible pipeline script
 - [**Social Navigation Benchmark**](./dev/issues/social-navigation-benchmark/README.md) - Benchmark design, metrics, schema, and how to run episodes/batches
 - **Full Classic Interaction Benchmark** – Implementation complete (episodes, aggregation, effect sizes, adaptive precision, plots, videos, scaling metrics). See detailed guide: [`benchmark_full_classic.md`](./benchmark_full_classic.md) (quickstart & tasks in `specs/122-full-classic-interaction/`).
 - [**Baselines**](./dev/baselines/README.md) — Overview of available baseline planners
   - [Random baseline](./dev/baselines/random.md) — how to use and configure
 - [**Force Field Visualization**](./force_field_visualization.md) — How to generate heatmap + quiver figures (PNG/PDF)
 - [**Scenario Thumbnails & Montage**](./scenario_thumbnails.md) — Generate per-scenario thumbnails and montage grids (PNG/PDF)
 - [**Force Field Heatmap**](./force_field_heatmap.md) — Heatmap + vector overlays figure (PNG/PDF)

#### Social Navigation Benchmark (Overview)

The benchmark layer provides:
 - Deterministic episode JSONL schema (versioned) with per-episode metrics.
 - Batch runner with resume manifest for incremental extensions.
 - Metrics suite + SNQI composite index (with weight recomputation tooling).
 - Aggregation + bootstrap CI utilities for statistical reporting.
 - Figure orchestrator to generate distributions, Pareto frontiers, force-field visualizations, thumbnails, and tables.
See the dedicated design page above for full specification and usage examples.
#### Figures naming and outputs

See `docs/dev/issues/figures-naming/design.md` for the canonical figure folder naming scheme and migration plan. A small tracker lives at `docs/dev/issues/figures-naming/todo.md`.

#### LaTeX Table Embedding (SNQI / Benchmark Tables)

The figures orchestration script writes `baseline_table.md` by default. To obtain a LaTeX version suitable for direct inclusion:

1. Fast path: run the figures orchestrator with `--table-tex` to produce `baseline_table.tex` automatically.
```bash
uv run python scripts/generate_figures.py \
  --episodes results/episodes_sf_long_fix1.jsonl \
  --auto-out-dir --no-pareto --table-tex \
  --dmetrics collisions,comfort_exposure,snqi --table-metrics collisions,comfort_exposure,snqi
```
2. Alternative: use the CLI table command with `--format tex` for custom file naming:
```bash
uv run python -m robot_sf.benchmark.cli table \
  --episodes results/episodes_sf_long_fix1.jsonl \
  --metrics collisions,comfort_exposure,near_misses,snqi \
  --format tex > docs/figures/table_snqi.tex
```
3. Include in LaTeX:
```latex
\input{docs/figures/table_snqi.tex}
```
4. The output uses `booktabs`; ensure your preamble contains:
```latex
\usepackage{booktabs}
```

Optional tuning:
- Reorder metrics via `--metrics` list order.
- Confidence intervals (bootstrap):
  1. Produce an aggregate summary with bootstrap CIs:
     ```bash
     uv run robot_sf_bench aggregate \
       --in results/episodes_sf_long_fix1.jsonl \
       --out results/summary_ci.json \
       --bootstrap-samples 1000 --bootstrap-confidence 0.95 --bootstrap-seed 123
     ```
  2. Generate tables from the summary adding CI columns:
     ```bash
     uv run python scripts/generate_figures.py \
       --episodes results/episodes_sf_long_fix1.jsonl \
       --table-summary results/summary_ci.json \
       --table-metrics collisions,comfort_exposure,snqi \
       --table-stats mean,median,p95 \
       --table-include-ci --table-tex --no-pareto \
       --out-dir docs/figures/ci_example
     ```
  3. Column naming pattern in Markdown: `<metric>_<stat>` plus `<metric>_<stat>_ci_low` / `_ci_high` (or with a custom suffix if `--ci-column-suffix ci95` is used → `_ci95_low/_ci95_high`).
  4. LaTeX version escapes underscores automatically; just `\input{...}` as usual.
  5. Missing CI arrays (e.g., when a stat lacked bootstrap) trigger a consolidated warning and empty cells.

Available CI options:
- `--table-include-ci` add interval columns.
- `--ci-column-suffix ci95` change suffix (default `ci`).

Example (custom suffix for 90% CIs):
```bash
uv run robot_sf_bench aggregate \
  --in results/episodes.jsonl --out results/summary_ci90.json \
  --bootstrap-samples 1000 --bootstrap-confidence 0.90
uv run python scripts/generate_figures.py \
  --episodes results/episodes.jsonl \
  --table-summary results/summary_ci90.json \
  --table-metrics collisions,snqi \
  --table-stats mean,median \
  --table-include-ci --ci-column-suffix ci90 --table-tex \
  --no-pareto --out-dir docs/figures/ci90_example
```

Fast iteration tip:
- Use `--no-pareto` with `scripts/generate_figures.py` to skip Pareto plot during rapid table refinement.
- Restrict distributions via `--dmetrics collisions,snqi` for quick rebuilds.

### Per-Test Performance Budget

A performance budget for tests helps prevent runtime regressions:

- Soft threshold: <20s (advisory)
- Hard timeout: 60s (enforced via `@pytest.mark.timeout(60)` markers)
- Report: Top 10 slowest tests printed with guidance at session end
- Relax: `ROBOT_SF_PERF_RELAX=1` suppresses soft breach failure escalation
- Enforce: `ROBOT_SF_PERF_ENFORCE=1` converts any soft or hard breach into a failure (unless relax set); advanced internal overrides: `ROBOT_SF_PERF_SOFT`, `ROBOT_SF_PERF_HARD`.

Core helpers live in `tests/perf_utils/` (policy, guidance, reporting, minimal_matrix). See the development guide section for authoring guidance and troubleshooting steps: [Dev Guide – Per-Test Performance Budget](./dev_guide.md#per-test-performance-budget).

### ⚙️ Setup & Configuration
- [**GPU Setup**](./GPU_SETUP.md) - GPU configuration for accelerated training
- [**UV Migration**](./UV_MIGRATION.md) - Migration to UV package manager

### 📈 Pedestrian Metrics  
- [**Pedestrian Metrics Overview**](./ped_metrics/PED_METRICS.md) - Summary of implemented metrics and their purpose
- [**Metric Analysis**](./ped_metrics/PED_METRICS_ANALYSIS.md) - Overview of metrics used in research and validation
- [**NPC Pedestrian Design**](./ped_metrics/NPC_PEDESTRIAN.md) - Details on the design and behavior of NPC pedestrians

### 📁 Media Resources
- [`img/`](./img/) - Documentation images and diagrams
- [`video/`](./video/) - Demo videos and animations

## 🚀 Quick Start Guides

### New Environment Architecture (Recommended)
```python
# Modern factory pattern for clean environment creation
from robot_sf.gym_env.environment_factory import (
    make_robot_env,
    make_image_robot_env,
    make_pedestrian_env
)

# Create environments with consistent interface
robot_env = make_robot_env(debug=True)
image_env = make_image_robot_env(debug=True)
ped_env = make_pedestrian_env(robot_model=model, debug=True)
```

### Legacy Pattern (Still Supported)
```python
# Traditional approach - still works for backward compatibility
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.env_config import EnvSettings

env = RobotEnv(env_config=EnvSettings(), debug=True)
```

## 🎯 Key Features

### Environment System
- **Unified Architecture**: Consistent base classes for all environments
- **Factory Pattern**: Clean, intuitive environment creation
- **Configuration Hierarchy**: Structured, extensible configuration system
- **Backward Compatibility**: Existing code continues to work

### Simulation Capabilities
- **Multi-Agent Support**: Robot and pedestrian simulation
- **Advanced Sensors**: LiDAR, image observations, target sensors
- **Map Integration**: Support for SVG maps and OpenStreetMap data
- **Visualization**: Real-time rendering and video recording

### Training & Analysis
- **Gymnasium Integration**: Compatible with modern RL frameworks
- **StableBaselines3 Support**: Ready for SOTA RL algorithms
- **Data Analysis Tools**: Comprehensive analysis utilities
- **Performance Monitoring**: Built-in metrics and logging

## 📖 Documentation Highlights

### 🆕 Latest Updates
**Benchmark Visual Artifacts (Plots + Videos Manifests)**: The Full Classic Interaction Benchmark now emits deterministic plot PDFs and representative (synthetic placeholder, SimulationView-ready) episode video artifacts with machine-readable manifests (`plot_artifacts.json`, `video_artifacts.json`, `performance_visuals.json`). See feature quickstart: `specs/126-title-integrate-plots/quickstart.md`.
- **Environment Refactoring Complete**: New unified architecture deployed
- **Migration Tools Available**: Automated migration script for updating code
- **Factory Pattern**: Clean, consistent environment creation interface
- **Comprehensive Testing**: All patterns validated and working
 - **Benchmark Runner Added**: Single-episode and batch APIs with schema validation and JSONL output. See the Social Navigation Benchmark docs for usage.

### 📋 Migration Status
- **33 files** identified for migration to new pattern
- **Migration script** available for automated updates
- **Backward compatibility** maintained throughout transition
- **Full documentation** provided for smooth migration

## 🔗 External Links

- [**Project Repository**](https://github.com/ll7/robot_sf_ll7) - Main GitHub repository
- [**Gymnasium Documentation**](https://gymnasium.farama.org/) - RL environment framework
- [**StableBaselines3**](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [**PySocialForce**](https://github.com/svenkreiss/PySocialForce) - Pedestrian simulation

## 🤝 Contributing

When contributing to the project:

1. **Use the new factory pattern** for environment creation
2. **Follow the unified configuration system** for settings
3. **Check the migration guide** when updating existing code
4. **Run tests** to ensure compatibility with both old and new patterns

## 📞 Support

- **Environment Issues**: Check the [refactoring documentation](./refactoring/)
- **Migration Help**: Use the [migration guide](./refactoring/migration_guide.md)
- **General Questions**: See individual documentation files
- **Bug Reports**: Use the GitHub issue tracker

---

*Last updated: September 2025 - Benchmark runner and batch API added*
