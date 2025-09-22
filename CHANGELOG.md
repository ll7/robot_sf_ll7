# Changelog

All notable changes to the Robot SF project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (Performance Budget Feature 124)
- Per-test performance budget enforcement (soft 20s, hard 60s) with slow test report (top 10) and guidance suggestions.
- Environment variables: `ROBOT_SF_PERF_RELAX` (suppress soft breach enforcement) and `ROBOT_SF_PERF_ENFORCE` (escalate soft breaches to failures).
- Shared performance utilities (`tests/perf_utils/`): policy, reporting, guidance, minimal scenario matrix helper.
- Refactored benchmark integration tests (resume, reproducibility) to use minimal matrix for faster deterministic runs.
- Synthetic slow test and guidance validation tests.

### Added
- Benchmark visual artifact integration (plots + videos manifests) for Full Classic Interaction Benchmark:
  - Post-run single-pass generation of placeholder plots and representative episode videos
  - SimulationView-first architecture with graceful synthetic fallback (current release uses synthetic until replay support added)
  - Deterministic selection (first N episodes) and machine-readable manifests: `plot_artifacts.json`, `video_artifacts.json`, `performance_visuals.json`
  - Renderer attribution field (`renderer`) and budget timing flags
  - Renderer toggle flag (`--renderer=auto|synthetic|sim-view`) with forced mode diagnostics
  - Replay capture adapter enabling SimulationView reconstruction (episode + step validation)
  - Extended skip-note taxonomy: `simulation-view-missing`, `moviepy-missing`, `insufficient-replay`, `render-error:<Type>`, `disabled`, `smoke-mode`
  - Performance split metrics (`first_video_render_time_s`, `first_video_encode_time_s`) plus memory sampling & over‑budget flags
  - Dependency matrix + lifecycle documentation (`docs/benchmark_visuals.md`) covering fallback ladder and required optional deps (pygame, moviepy/ffmpeg, jsonschema, psutil)
- **Social Navigation Benchmark Platform** - Complete benchmark infrastructure for reproducible social navigation research
- **Full Classic Interaction Benchmark (Initial Implementation)**
  - Synthetic placeholder execution pipeline (planning → execution → aggregation → effect sizes → precision loop)
  - Adaptive sampling with CI half-width early stop targets (collision_rate, success_rate, placeholder snqi target)
  - Plot artifacts (distribution, trajectory, KDE/Pareto/force heatmap placeholders) and annotated video generation (graceful fallback)
  - CLI `scripts/classic_benchmark_full.py` exposing comprehensive flags (episodes, precision thresholds, videos)
  - Manifest instrumentation for runtime, episodes_per_second, scaling_efficiency placeholder metrics
  - Resume idempotency and performance smoke tests (T042, T044) plus failure injection test for videos (T043)
  - Dedicated documentation page `docs/benchmark_full_classic.md`

  - **Episode Runner**: Parallel execution with manifest-based resume functionality
  - **CLI Interface**: 15 comprehensive subcommands covering full experiment workflow
    - `run` - Execute episodes with parallel workers
    - `baseline` - Compute baseline statistics  
    - `aggregate` - Generate summaries with bootstrap confidence intervals
    - `validate-config` - Schema validation for scenarios
    - `list-scenarios` - Display scenario configurations
    - `figure-*` - Distribution plots, Pareto frontiers, force field heatmaps, thumbnails
    - `table` - Generate baseline tables (Markdown/LaTeX)
    - `snqi-*` - SNQI weight recomputation and ablation analysis
    - Additional utilities for trajectory extraction and episode validation
  - **SNQI Metrics Suite**: Composite Social Navigation Quality Index with component breakdown
  - **Statistical Analysis**: Bootstrap confidence intervals and robust aggregation
  - **Unified Baseline Interface**: PlannerProtocol for consistent algorithm comparison
  - **Figure Orchestrator**: Publication-quality visualization pipeline
  - **Comprehensive Testing**: 108 tests including 33 new tests for benchmark functionality
- **Complete Documentation**: Step-by-step quickstart guide with example workflows
- **Performance Validation**: 20-25 steps/second with linear parallel scaling

### Changed
- Enhanced baseline planner interface with unified PlannerProtocol
- Improved test coverage with comprehensive benchmark validation
- Updated main documentation with prominent benchmark platform section

### Technical Details
- All baseline planners (SocialForce, PPO, Random) now implement PlannerProtocol interface
- Episode schema includes comprehensive metadata and provenance tracking
- Deterministic execution with seed-based reproducibility
- Manifest-based resume enables incremental experiment extension
- Bootstrap statistical analysis provides uncertainty quantification
- Publication-ready figure generation with LaTeX table support

### Migration Notes
- No breaking changes to existing functionality
- New benchmark CLI commands available via `python -m robot_sf.benchmark.cli`
- Legacy environment interfaces remain fully supported
- Comprehensive backward compatibility maintained

---

## Guidelines for Contributors

When adding entries to this changelog:

1. **Group changes** by `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`
2. **Write for users** - focus on user-visible changes and their benefits
3. **Include migration notes** for breaking changes
4. **Reference related issues/PRs** where applicable
5. **Keep descriptions concise** but informative

## Version Numbering

This project uses semantic versioning:
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions  
- **PATCH** version for backwards-compatible bug fixes