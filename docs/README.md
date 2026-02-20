# Robot SF Documentation

Welcome to the Robot SF documentation! This directory contains comprehensive guides and references for using and developing with the Robot SF simulation framework.

<!-- This document should mainly serve as a navigation hub and overview for the various components and guides available within the Robot SF project. Refer to individual files for detailed information. -->

**Artifact root**: All generated artifacts (JSONL, figures, videos) must live under the git-ignored `output/` directory. Legacy `results/` paths have been migrated; update commands accordingly when running examples or scripts.

## 🚀 Social Navigation Benchmark Platform (Complete)

**The Social Navigation Benchmark Platform is now fully operational!** 

### Quick Start

* **[Complete Quickstart Guide](../specs/120-social-navigation-benchmark-plan/quickstart.md)** - Step-by-step experiment execution, visualization, and interpretation
* **[CLI Reference](./dev/issues/social-navigation-benchmark/README.md)** - All 15 CLI subcommands with examples
* **Implementation Status**: All major features complete, 108 tests passing

### Core Capabilities

* **Episode Runner**: Parallel execution with resume functionality and deterministic seeding
* **Metrics Suite**: SNQI composite index with component breakdown and weight recomputation
* **Baseline Interface**: Unified PlannerProtocol for SocialForce, PPO, Random planners  
* **Statistical Analysis**: Bootstrap confidence intervals and robust aggregation
* **Figure Orchestrator**: Distribution plots, Pareto frontiers, force fields, thumbnails, tables
* **CLI Tools**: 15 subcommands covering full experiment workflow

### Ready-to-Use Workflows

1. **Quick Assessment** (~15 min): Compare robot policies against baselines
2. **Research Study** (~2-4 hours): Multi-parameter analysis with publication figures
3. **Weight Sensitivity** (~45 min): Analyze SNQI component importance

**Start Here**: [specs/120-social-navigation-benchmark-plan/quickstart.md](../specs/120-social-navigation-benchmark-plan/quickstart.md)

---

* [🚀 Social Navigation Benchmark Platform (Complete)](#-social-navigation-benchmark-platform-complete)
  + [Quick Start](#quick-start)
  + [Core Capabilities](#core-capabilities)
  + [Ready-to-Use Workflows](#ready-to-use-workflows)
* [📚 Documentation Index](#-documentation-index)
  + [Getting Started](#getting-started)
  + [Benchmarking \& Metrics](#benchmarking--metrics)
  + [Tooling](#tooling)
  + [Architecture \& Refactoring](#architecture--refactoring)
  + [Simulation \& UI](#simulation--ui)
  + [Figures \& Visualization](#figures--visualization)
  + [Performance \& CI](#performance--ci)
  + [Hardware \& Environment](#hardware--environment)
  + [Additional Resources (Legacy Structure)](#additional-resources-legacy-structure)
  + [🏗️ Architecture \& Development](#️-architecture--development)
  + [🎮 Simulation \& Environment](#-simulation--environment)
  + [📊 Analysis \& Tools](#-analysis--tools)
    - [Social Navigation Benchmark (Overview)](#social-navigation-benchmark-overview)
    - [Figures naming and outputs](#figures-naming-and-outputs)
    - [LaTeX Table Embedding (SNQI / Benchmark Tables)](#latex-table-embedding-snqi--benchmark-tables)
  + [Per-Test Performance Budget](#per-test-performance-budget)
  + [⚙️ Setup \& Configuration](#️-setup--configuration)
  + [📈 Pedestrian Metrics](#-pedestrian-metrics)
  + [📁 Media Resources](#-media-resources)
* [🚀 Quick Start Guides](#-quick-start-guides)
  + [New Environment Architecture (Recommended)](#new-environment-architecture-recommended)
  + [Legacy Pattern (Still Supported)](#legacy-pattern-still-supported)
    - [Environment Factory Ergonomics Migration (Feature 130)](#environment-factory-ergonomics-migration-feature-130)
* [🎯 Key Features](#-key-features)
  + [Environment System](#environment-system)
  + [Simulation Capabilities](#simulation-capabilities)
  + [Training \& Analysis](#training--analysis)
* [📖 Documentation Highlights](#-documentation-highlights)
  + [🆕 Latest Updates](#-latest-updates)
  + [📋 Migration Status](#-migration-status)
  + [Architecture \& design features](#architecture--design-features)
* [🔗 External Links](#-external-links)
* [🤝 Contributing](#-contributing)
* [📞 Support](#-support)
* [Helper Catalog](#helper-catalog)

## 📚 Documentation Index

### Getting Started
* **[Development Guide](./dev_guide.md)** - Primary reference for development workflows, setup, testing, quality gates, and coding standards
* **[Agent Index](./AGENT_INDEX.md)** - Agent-oriented index of training, benchmarking, observations, and artifacts
* **[Observation Contract](./dev/observation_contract.md)** - Observation schemas, shapes, and normalization conventions
* **[Training Protocol Template](./dev/training_protocol_template.md)** - Fill-in template for documenting training/evaluation runs
* **[Model Registry](../model/registry.md)** - Track trained policies and load them on-demand via `robot_sf.models`
* **[Examples Catalog](../examples/README.md)** - Manifest-backed index of quickstart, advanced, benchmark, and plotting scripts with usage metadata
* **[SocNav structured observation example](../examples/advanced/18_socnav_structured_observation.py)** - Run RobotEnv with SocNavBench-style observations and a simple planner adapter.
* **[SocNav structured observation how-to](./dev/issues/socnav_structured_observation.md)** - Enable `ObservationMode.SOCNAV_STRUCT` and use planner adapters (lightweight + SocNavBench wrapper).
* **[Issue 403 Grid PPO Training Runbook](./training/issue_403_grid_training.md)** - Step-by-step training for the grid+SocNav PPO expert.
* **[Predictive Planner Training Runbook](./training/predictive_planner_training.md)** - Data collection, training, proxy selection, and benchmark evaluation workflow for `prediction_planner`.
* **[DreamerV3 RLlib Runbook (`drive_state` + `rays`)](./training/dreamerv3_rllib_drive_state_rays.md)** - Config-first training flow for RLlib DreamerV3 without image observations.
* **[Global Planner Quickstart (WIP)](../specs/342-svg-global-planner/quickstart.md)** - Placeholder for the upcoming SVG-based global planner documentation and examples.
* **[Artifact Policy Quickstart](../specs/243-clean-output-dirs/quickstart.md)** - Step-by-step migration, guard enforcement, and override instructions for the canonical `output/` tree
* **[Imitation Learning Pipeline](./imitation_learning_pipeline.md)** - Complete guide to PPO pre-training with expert trajectories
* **[Imitation Learning Quickstart](../specs/001-ppo-imitation-pretrain/quickstart.md)** - Step-by-step workflow for BC pre-training and PPO fine-tuning
* **[Optuna Expert PPO Sweep Report (2026-02-11)](./training/optuna_expert_ppo_sweep_2026-02-11.md)** - Sweep findings and PPO hyperparameter guidelines
* **[Waypoint Noise For Route Generalization](./training/waypoint_noise.md)** - Configure Gaussian waypoint perturbation to reduce route memorization during training
* **[Research Reporting](./research_reporting.md)** - Automated research report generation: multi-seed aggregation, statistical analysis, figure generation, Markdown/LaTeX export
* **[Feature Extractors Guide](./feature_extractors/usage_guide.md)** - Configure and compare extractor presets, run multi-extractor training, and generate reports
* **[Run Tracker & History CLI](./dev_guide.md#run-tracker--history-cli)** - Enable the failure-safe tracker on the imitation pipeline, monitor `status`/`watch` output, run telemetry perf-tests, mirror telemetry to TensorBoard, filter historical runs, and export Markdown/JSON summaries via the `scripts/tools/run_tracker_cli.py` commands (`status`, `watch`, `list`, `summary`, `export`, `perf-tests`, `enable-tensorboard`)

### Benchmarking & Metrics

* **[Benchmark Spec (Classic Interactions)](./benchmark_spec.md)** - Scenario split + seeds, baseline categories, reproducible commands, and metric caveats
* **[Benchmark: Camera-ready / Scenario Reports](./benchmark_camera_ready.md)** - Camera-ready campaign workflow, planner report partitions, and publication-grade artifact contract
* **[Benchmark Runner & Metrics](./benchmark.md)** - Episode schema, aggregation, metrics suite (collisions, comfort exposure, SNQI), and validation hooks
* **[Full Classic Interaction Benchmark](./benchmark_full_classic.md)** - Complete guide: episodes, aggregation, effect sizes, adaptive precision, plots, videos, scaling metrics
* **[Benchmark Artifact Publication](./benchmark_artifact_publication.md)** - Public artifact policy, DOI-ready export bundles, release/Zenodo workflow
* **[Benchmark Visual Artifacts](./benchmark_visuals.md)** - SimulationView & synthetic video pipeline, performance metrics
* **[Metrics Specification](./dev/issues/social-navigation-benchmark/metrics_spec.md)** - Formal definitions of benchmark metrics (includes per-pedestrian force quantiles)
* **[Local Navigation Benchmark Gap Analysis (2026-01-14)](./dev/benchmark_plan_2026-01-14.md)** - Current-state inventory, missing pieces, and open questions for local planner benchmarking
* **[Prediction Planner Baseline](./baselines/prediction_planner.md)** - High-level model description, benchmark role, configuration, and citation/provenance notes
* **[Prediction Planner PR Readiness (2026-02-20)](./context/predictive_planner_pr_readiness_2026-02-20.md)** - Completed integration checklist and remaining maintainer decisions before final merge
* **[Issues 485-492 Execution Trace](./context/issues_485_492_execution.md)** - Implementation summary, validation runs, and rollout notes for the benchmark hardening changes
* **[Issues 500-504 Execution Notes](./context/issues_500_501_504_execution.md)** - Metadata enrichment, time-to-goal contract extensions, and adapter-impact probing implementation
* **[Issue 499 Execution Notes](./context/issue_499_execution.md)** - Publication bundle tooling, policy, and size-measurement workflow

### Tooling

* **[SNQI Weight Tools](./snqi-weight-tools/README.md)** - Recompute, optimize, and analyze SNQI weights; command reference and workflow examples
* **[Pyreverse UML Generation](./pyreverse.md)** - Generate class diagrams from code
* **[Data Analysis Utilities](./DATA_ANALYSIS.md)** - Analysis helpers and data processing tools
* **[Imitation Results Analysis](./imitation_results_analysis.md)** - Compare baseline vs pre-trained runs, emit training summaries and figures
* **[SVG Inspection Workflow](./dev/svg_inspection_workflow.md)** - Inspect route/zone consistency, parser-risky path commands, and obstacle crossings with `scripts/validation/svg_inspect.py`

### Architecture & Refactoring

* **[Refactoring Overview](./refactoring/)** - Complete guide to the refactored environment architecture (deployment status, plan, migration guide, summary, automated codebase analysis)
* **[Subtree Migration Guide](./SUBTREE_MIGRATION.md)** - Git subtree integration for fast-pysf (migration from submodule)
* **[UV Migration Notes](./UV_MIGRATION.md)** - Migration to UV package manager
* **[Repository Structure Analysis](./dev/issues/repository-structure-analysis.md)** - Comprehensive assessment of codebase organization and improvement roadmap
* **[Agents & Contributor Onboarding](../AGENTS.md)** - High-level repository structure, coding/testing conventions, workflow tips

### Simulation & UI

* **[Simulation View](./SIM_VIEW.md)** - Visualization and rendering system
* **[SVG Map Editor](./SVG_MAP_EDITOR.md)** - SVG-based map creation tools and usage
* **[OSM Map Generation](./osm_map_workflow.md)** - Programmatic, reproducible maps from OpenStreetMap data (PBF import, zone/route definition, scenario creation)
  + **Quick Start**: 3 approaches (visual editor, programmatic API, hybrid)
  + **API Reference**: 6 helper functions (zones, routes, config management, YAML loading)
  + **Examples**: 4 realistic scenarios (simple navigation, urban intersection, variable density, load/verify)
* **[Map Verification](../specs/001-map-verification/quickstart.md)** - Validate SVG maps for structural integrity and runtime compatibility
* **[Issue 388 Execution Notes](./context/issue_388_execution.md)** - Self-intersecting obstacle-path repair behavior and validation details
* **[Francis 2023 Scenario Pack](../maps/svg_maps/francis2023/readme.md)** - SVG maps +
  scenario matrix for Fig. 7 archetypes; definitions in
  [configs/scenarios/francis2023.yaml](../configs/scenarios/francis2023.yaml)
* **[Occupancy Grid Guide](./dev/occupancy/Update_or_extend_occupancy.md)** - Configure grid observations, spawn queries, and pygame overlays
* **[Circle Rasterization Fix](./dev/issues/circle-rasterization-fix/README.md)** - Clarifies circle overlap handling in occupancy grid rasterization
* **[Telemetry Pane & Headless Artifacts](../specs/343-telemetry-viz/quickstart.md)** - Enable docked charts in Pygame, replay/export telemetry, and run headless smoke tests
* **[Telemetry Pane Display Fix](./telemetry-pane-fix.md)** - Technical analysis and solution for continuous graph rendering, surface caching, and buffer management

### Figures & Visualization

* **[Trajectory Visualization](./trajectory_visualization.md)** - Generate trajectory plots
* **[Force Field Visualization](./force_field_visualization.md)** - Heatmap + quiver figures (PNG/PDF)
* **[Pareto Plotting](./pareto_plotting.md)** - Generate Pareto frontier plots
* **[Force Field Heatmap](./force_field_heatmap.md)** - Heatmap + vector overlays figure (PNG/PDF)

### Performance & CI

* **[Performance Notes](./performance_notes.md)** - Performance targets, benchmarking, and optimization notes
* **[Issue 483 Execution Notes](./context/issue_483_execution.md)** - Cold/warm regression guard implementation details and workflow wiring
* **[Issue 495 Execution Notes](./context/issue_495_execution.md)** - Overall trend benchmark matrix, history comparison, and nightly cache-backed tracking
* **[Warning Hygiene Sweep](./context/warning_hygiene_2026-02-13.md)** - Warning-noise root-cause fixes and dependency mitigation notes
* **[Coverage Guide](./coverage_guide.md)** - Code coverage collection, baseline tracking, CI integration

### Hardware & Environment

* **[Environment Configuration](./ENVIRONMENT.md)** - Detailed environment setup and usage

---

### Additional Resources (Legacy Structure)

<details>
<summary>Click to expand legacy detailed index</summary>

### 🏗️ Architecture & Development

* **[Development Guide](./dev_guide.md)** - Primary reference for development workflows, testing, and quality standards
* **[Configuration Architecture](./architecture/configuration.md)** - Configuration hierarchy, precedence rules, and migration guide
* **[Repository Structure Analysis](./dev/issues/repository-structure-analysis.md)** - Comprehensive assessment of codebase organization and improvement roadmap
* **[Coverage Guide](./coverage_guide.md)** - Comprehensive guide to code coverage collection, baseline tracking, and CI integration
* **[Environment Refactoring](./refactoring/)** - **NEW**: Complete guide to the refactored environment architecture
  + [Deployment Status](./refactoring/DEPLOYMENT_READY.md) - Current implementation status
  + [Refactoring Plan](./refactoring/refactoring_plan.md) - Technical architecture details
  + [Migration Guide](./refactoring/migration_guide.md) - Step-by-step migration instructions
  + [Implementation Summary](./refactoring/refactoring_summary.md) - What was accomplished
  + [Migration Report](./refactoring/migration_report.md) - Automated codebase analysis
  + **Classic interactions refactor (Feature 139)** — Design note: Extract visualization & formatting helpers — `docs/dev/issues/classic-interactions-refactor/design.md`
* **[Architectural Decoupling (Feature 149)](../specs/149-architectural-coupling-and/)** - Backend and sensor registry system for extensible simulation
  + [Quickstart Guide](../specs/149-architectural-coupling-and/quickstart.md) - Usage examples for backend selection and sensor registration
  + [Tasks & Progress](../specs/149-architectural-coupling-and/tasks.md) - Implementation task tracking
* **[Agents & Contributor Onboarding](../AGENTS.md)** – High-level repository structure, coding/testing conventions, and workflow tips for new contributors

### 🎮 Simulation & Environment

* [**Simulation View**](./SIM_VIEW.md) - Visualization and rendering system
* [**Map Editor Usage**](./MAP_EDITOR_USAGE.md) - Creating and editing simulation maps
* [**SVG Map Editor**](./SVG_MAP_EDITOR.md) - SVG-based map creation tools
* [**Single Pedestrians**](./single_pedestrians.md) - Define individual pedestrians with goals or trajectories in SVG/JSON/code
* [**Multi-Pedestrian Example**](../examples/example_multi_pedestrian.py) - Demonstrates multiple single pedestrians (goal, trajectory, static) in one scenario
* [**Scenario Specification Checklist**](./scenario_spec_checklist.md) - Authoring checklist for per-scenario/archetype/manifest files
* **Classic Interaction Scenario Pack** (configs/scenarios/classic_interactions.yaml) – Canonical crossing, head‑on, overtaking, bottleneck, doorway, merging, T‑intersection, and group crossing archetypes for benchmark coverage.
* **[Francis 2023 Scenario Pack](../maps/svg_maps/francis2023/readme.md)** - SVG maps +
  scenario matrix in [configs/scenarios/francis2023.yaml](../configs/scenarios/francis2023.yaml).
* **Classic Interactions PPO Visualization (Feature 128)** – Deterministic PPO policy demo with optional recording (docs: `docs/dev/issues/classic-interactions-ppo/` | spec+plan+tasks under `specs/128-classic-interactions-ppo/`).

### 📊 Analysis & Tools  

* [**SNQI Weight Tooling**](./snqi-weight-tools/README.md) - User guide for recomputing, optimizing, and analyzing SNQI weights
* [**SNQI Figures (orchestrator usage)**](../examples/README.md) - Generate SNQI-augmented figures from existing episodes
* [**Full SNQI Flow (episodes → baseline → figures)**](../examples/benchmarks/snqi_full_flow.py) - End-to-end reproducible pipeline script
 - [**Benchmark Schema & Aggregation Diagnostics**](./benchmark.md) - Episode metadata mirrors, algorithm grouping keys, `_meta` warnings, and validation hooks
 - [Regression Notes – Algorithm Aggregation](./dev/issues/142-aggregation-mixes-algorithms/design.md) - Test matrix, warnings, and smoke workflow for Feature 142
 - [**Social Navigation Benchmark**](./dev/issues/social-navigation-benchmark/README.md) - Benchmark design, metrics, schema, and how to run episodes/batches
 - **Full Classic Interaction Benchmark** – Implementation complete (episodes, aggregation, effect sizes, adaptive precision, plots, videos, scaling metrics). See detailed guide: [ `benchmark_full_classic.md` ](./benchmark_full_classic.md) (quickstart & tasks in `specs/122-full-classic-interaction/` ).
 - **Benchmark Visual Artifacts** – SimulationView & synthetic video pipeline, performance metrics: [ `benchmark_visuals.md` ](./benchmark_visuals.md)
 - **Episode Video Artifacts (MVP)** – Design notes and links: [ `docs/dev/issues/video-artifacts/design.md` ](./dev/issues/video-artifacts/design.md)
 - [**Baselines**](./dev/baselines/README.md) — Overview of available baseline planners
   - [Random baseline](./dev/baselines/random.md) — how to use and configure
 - [**Force Field Visualization**](./force_field_visualization.md) — How to generate heatmap + quiver figures (PNG/PDF)
 - [**Scenario Thumbnails & Montage**](./scenario_thumbnails.md) — Generate per-scenario thumbnails and montage grids (PNG/PDF)
 - [**Force Field Heatmap**](./force_field_heatmap.md) — Heatmap + vector overlays figure (PNG/PDF)

</details>

#### Social Navigation Benchmark (Overview)

The benchmark layer provides:
 - Deterministic episode JSONL schema (versioned) with per-episode metrics.
 - Batch runner with resume manifest for incremental extensions.
 - Metrics suite + SNQI composite index (with weight recomputation tooling).
 - Aggregation + bootstrap CI utilities for statistical reporting.
 - Figure orchestrator to generate distributions, Pareto frontiers, force-field visualizations, thumbnails, and tables.
See the dedicated design page above for full specification and usage examples.

#### Figures naming and outputs

See `docs/dev/issues/figures-naming/design.md` for the canonical figure folder naming scheme and migration plan. A small tracker lives at `docs/dev/issues/figures-naming/todo.md` .

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
* Reorder metrics via `--metrics` list order.
* Confidence intervals (bootstrap):
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
* `--table-include-ci` add interval columns.
* `--ci-column-suffix ci95` change suffix (default `ci`).

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
* Use `--no-pareto` with `scripts/generate_figures.py` to skip Pareto plot during rapid table refinement.
* Restrict distributions via `--dmetrics collisions,snqi` for quick rebuilds.

### Per-Test Performance Budget

A performance budget for tests helps prevent runtime regressions:

* Soft threshold: <20s (advisory)
* Hard timeout: 60s (enforced via `@pytest.mark.timeout(60)` markers)
* Report: Top 10 slowest tests printed with guidance at session end
* Relax: `ROBOT_SF_PERF_RELAX=1` suppresses soft breach failure escalation
* Enforce: `ROBOT_SF_PERF_ENFORCE=1` converts any soft or hard breach into a failure (unless relax set); advanced internal overrides: `ROBOT_SF_PERF_SOFT`,  `ROBOT_SF_PERF_HARD`.

Core helpers live in `tests/perf_utils/` (policy, guidance, reporting, minimal_matrix). See the development guide section for authoring guidance and troubleshooting steps: [Dev Guide – Per-Test Performance Budget](./dev_guide.md#per-test-performance-budget).

### ⚙️ Setup & Configuration

* [**UV Migration**](./UV_MIGRATION.md) - Migration to UV package manager
* [**Subtree Migration**](./SUBTREE_MIGRATION.md) - Git subtree integration for fast-pysf (migration from submodule)

### 📈 Pedestrian Metrics  

* [**Pedestrian Metrics Overview**](./ped_metrics/PED_METRICS.md) - Summary of implemented metrics and their purpose
* [**Metric Analysis**](./ped_metrics/PED_METRICS_ANALYSIS.md) - Overview of metrics used in research and validation
* [**NPC Pedestrian Design**](./ped_metrics/NPC_PEDESTRIAN.md) - Details on the design and behavior of NPC pedestrians
 - [**Pedestrian Density Reference**](./ped_metrics/PEDESTRIAN_DENSITY.md) - Units, canonical triad (0.02/0.05/0.08), advisory range, difficulty mapping & test policy
* [Per-pedestrian force quantiles demo](../examples/benchmarks/per_ped_force_quantiles_demo.py) - Script comparing aggregated vs per-ped force quantiles
* [**Issue 503 Pedestrian-Impact Metrics Notes**](./context/issue_503_pedestrian_impact_metrics.md) - Execution notes for the optional experimental `ped_impact_*` metric group

### 📁 Media Resources

* [`img/`](./img/) - Documentation images and diagrams
* [`video/`](./video/) - Demo videos and animations

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

#### Environment Factory Ergonomics Migration (Feature 130)

See the new migration guide: [Environment Factory Migration](./dev/issues/130-improve-environment-factory/migration.md). Includes before/after examples, seeding, legacy env vars ( `ROBOT_SF_FACTORY_LEGACY` , `ROBOT_SF_FACTORY_STRICT` ), and precedence rules. Quickstart examples: `specs/130-improve-environment-factory/quickstart.md` .

```python
# Traditional approach - still works for backward compatibility
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.env_config import EnvSettings

env = RobotEnv(env_config=EnvSettings(), debug=True)
```

## 🎯 Key Features

### Environment System

* **Unified Architecture**: Consistent base classes for all environments
* **Factory Pattern**: Clean, intuitive environment creation
* **Configuration Hierarchy**: Structured, extensible configuration system
* **Backward Compatibility**: Existing code continues to work

### Simulation Capabilities

* **Multi-Agent Support**: Robot and pedestrian simulation
* **Advanced Sensors**: LiDAR, image observations, target sensors
* **Map Integration**: Support for SVG maps and OpenStreetMap data
* **Visualization**: Real-time rendering and video recording

### Training & Analysis

* **Gymnasium Integration**: Compatible with modern RL frameworks
* **StableBaselines3 Support**: Ready for SOTA RL algorithms
* **Data Analysis Tools**: Comprehensive analysis utilities
* **Performance Monitoring**: Built-in metrics and logging
* **Multi-Extractor Workflow**: `scripts/multi_extractor_training.py` writes timestamped runs under `tmp/multi_extractor_training/`, capturing JSON + Markdown summaries alongside legacy `complete_results.json` for downstream automation.

## 📖 Documentation Highlights

### 🆕 Latest Updates

* **Architecture Decoupling (Feature 149)**: Simulator facade and registries (simulator & sensors) scaffolded behind the factory pattern; backend selection via unified config with a default of "fast-pysf". See design docs and quickstart below.

### 📋 Migration Status

### Architecture & design features

* Architectural decoupling and consistency overhaul (Feature 149):
  + Design: `specs/149-architectural-coupling-and/spec.md`
  + Plan: `specs/149-architectural-coupling-and/plan.md`
  + Quickstart: `specs/149-architectural-coupling-and/quickstart.md`

* **33 files** identified for migration to new pattern
* **Migration script** available for automated updates
* **Full documentation** provided for smooth migration

## 🔗 External Links

* [**Project Repository**](https://github.com/ll7/robot_sf_ll7) - Main GitHub repository
* [**Gymnasium Documentation**](https://gymnasium.farama.org/) - RL environment framework
* [**StableBaselines3**](https://stable-baselines3.readthedocs.io/) - RL algorithms
* [**PySocialForce**](https://github.com/svenkreiss/PySocialForce) - Pedestrian simulation

## 🤝 Contributing

When contributing to the project:

1. **Use the new factory pattern** for environment creation
2. **Follow the unified configuration system** for settings
3. **Check the migration guide** when updating existing code
4. **Run tests** to ensure compatibility with both old and new patterns

## 📞 Support

* **Environment Issues**: Check the [refactoring documentation](./refactoring/)
* **Migration Help**: Use the [migration guide](./refactoring/migration_guide.md)
* **General Questions**: See individual documentation files
* **Bug Reports**: Use the GitHub issue tracker

## Planner Documentation

* **Global Planner**: See `specs/342-svg-global-planner/quickstart.md` for the visibility-graph planner API, POI routing, and integration guidance.
* **Planner selection**: Choose between visibility and classic grid planners in `docs/dev_guide.md#planner-selection-visibility-vs-classic-grid`.

---

*Last updated: February 2026 - Cold/warm performance regression suite added*
