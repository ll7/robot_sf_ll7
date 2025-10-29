# Changelog

All notable changes to the Robot SF project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **fast-pysf Subtree (Feature 146)**: Integrated `fast-pysf` as a git subtree for easier management and updates
- **pytest-cov Integration (Feature 145)**: Comprehensive code coverage monitoring and CI/CD integration
  - **Automatic Collection**: Coverage data collected automatically during test runs via `pytest-cov` without additional commands
  - **Multi-Format Reports**: Terminal summary, interactive HTML (`htmlcov/index.html`), and machine-readable JSON (`coverage.json`)
  - **Baseline Comparison**: CI/CD pipeline compares coverage against baseline with non-blocking warnings on decreases
  - **VS Code Integration**: Tasks for "Run Tests with Coverage" and "Open Coverage Report"
  - **CI/CD Workflow**: GitHub Actions integration with caching, baseline updates on main branch, and artifact uploads
  - **Library Infrastructure**: 
    - `robot_sf/coverage_tools/baseline_comparator.py`: CoverageSnapshot, CoverageBaseline, CoverageDelta entities with comparison logic
    - `robot_sf/coverage_tools/report_formatter.py`: Multi-format report generation (terminal/JSON/markdown)
    - `scripts/coverage/compare_coverage.py`: CLI tool for local and CI baseline comparison
  - **Comprehensive Testing**: 18 unit tests (5 smoke + 13 baseline comparator) with 91.51% coverage of comparison logic
  - **Documentation**: 
    - `docs/coverage_guide.md`: 500+ line comprehensive guide with quickstart, CI integration, troubleshooting
    - `examples/coverage_example.py`: Programmatic usage examples
    - Updated `docs/dev_guide.md` with coverage workflow section
  - **Configuration**: pyproject.toml with [tool.coverage.*] sections, automatic pytest integration, parallel execution support
  - Coverage excludes: tests, examples, scripts, fast-pysf submodule per omit configuration
  - Non-blocking CI design: warnings only, no build failures on coverage decreases
- Paper Metrics Implementation (Feature 144): Comprehensive implementation of 22 social navigation metrics from paper 2306.16740v4 (Table 1):
  - **NHT (Navigation/Hard Task) Metrics (11)**: `success_rate`, `collision_count`, `wall_collisions`, `agent_collisions`, `human_collisions`, `timeout`, `failure_to_progress`, `stalled_time`, `time_to_goal`, `path_length`, `success_path_length` (SPL)
  - **SHT (Social/Human-aware Task) Metrics (14)**: velocity statistics (`velocity_min/avg/max`), acceleration statistics (`acceleration_min/avg/max`), jerk statistics (`jerk_min/avg/max`), clearing distance (`clearing_distance_min/avg`), `space_compliance`, `distance_to_human_min`, `time_to_collision_min`, `aggregated_time`
  - Extended `EpisodeData` dataclass with optional `obstacles` and `other_agents_pos` fields for enhanced collision detection
  - Internal helper functions: `_compute_ped_velocities`, `_compute_jerk`, `_compute_distance_matrix`
  - Comprehensive unit test coverage (30+ tests) with edge case validation
  - All metrics documented with formulas, units, ranges, and paper references
  - Backward compatible integration with existing benchmark infrastructure
- Multi-extractor training flow refactor (Feature 141): `scripts/multi_extractor_training.py` now uses shared helpers in `robot_sf.training`, emits schema-backed `summary.json`/`summary.md` plus legacy `complete_results.json`, captures hardware profiles, honors macOS spawn semantics, and ships default/GPU configs alongside updated SLURM automation and analyzer support.
- Helper Catalog Consolidation (Feature 140): Extracted reusable helper logic from examples and scripts into organized library modules:
  - `robot_sf.benchmark.helper_catalog`: Policy loading (`load_trained_policy`), environment preparation (`prepare_classic_env`), and episode execution (`run_episodes_with_recording`) helpers.
  - `robot_sf.render.helper_catalog`: Directory management (`ensure_output_dir`) and frame capture utilities (`capture_frames`).
  - `robot_sf.docs.helper_catalog`: Documentation index management (`register_helper`) for automated helper catalog updates.
  - Complete refactoring of all maintained examples (`examples/demo_*.py`) and scripts (`scripts/*.py`) to use helper catalog functions instead of duplicate logic.
  - Helper registry data structures with typed interfaces for discoverable, testable helper capabilities.
  - All helper functions include comprehensive docstrings, error handling, and Loguru logging compliance.
- Episode Video Artifacts (MVP):
  - New CLI flags for benchmark runner: `--no-video` and `--video-renderer=synthetic|sim-view|none`.
  - Synthetic lightweight encoder that renders a red-dot path from robot positions and writes per‑episode MP4s under `results/videos/`.
  - Episode JSON schema extension to include optional `video` manifest `{path, format, filesize_bytes, frames, renderer}`.
  - End‑to‑end wiring through CLI → batch runner → worker → episode; deterministic file naming `video_<episode_id>.mp4`.
  - Tests: CLI integration (`tests/test_cli_run_video.py`) and programmatic API (`tests/unit/test_runner_video.py`), both skipped when MoviePy/ffmpeg unavailable.
- Environment Factory Ergonomics (Feature 130): Structured `RenderOptions` / `RecordingOptions`, legacy kw mapping layer (`fps`, `video_output_path`, `record_video`), precedence normalization and logging diagnostics; performance guard (<10% creation mean regression) and new migration guide (`docs/dev/issues/130-improve-environment-factory/migration.md`). New example: `examples/demo_factory_options.py`.
- Governance: Constitution version 1.2.0 introducing Principle XII (Preferred Logging & Observability) establishing Loguru as the canonical logging facade for library code and prohibiting unapproved `print()` usage outside sanctioned CLI/test contexts.
- Documentation: Development guide updated with new Logging & Observability section summarizing usage guidelines (levels, performance constraints, acceptable exceptions).
- SVG Map Validation (Feature 131): Manual bulk SVG validation script (`examples/svg_map_example.py`) supporting strict/lenient modes, summary reporting, environment override (`SVG_VALIDATE_STRICT`), and compliance spec (FR-001–FR-014). Added missing spawn/goal zones and minimal `robot_route_0_0` / `ped_route_0_0` paths to classic interaction SVG maps and large map asset now includes explicit width/height and minimal routes.

### Fixed
- Normalized obstacle vertices to tuples to prevent ambiguous NumPy truth-value error during SVG path obstacle conversion in large map (`map3_1350_buildings_inkscape.svg`).
- Preserved per-algorithm benchmark aggregation (Feature 142): classic orchestrator now mirrors `algo` into `scenario_params`, aggregation raises on missing metadata, and summary outputs emit `_meta` diagnostics plus warnings when expected baselines are absent.

### Migration Notes
- No code changes required; existing Loguru usage already compliant. Any remaining incidental `print()` in library modules should be migrated opportunistically (PATCH) unless tied to user-facing CLI UX.


### Added (Performance Budget Feature 124)
- Per-test performance budget enforcement (soft 20s, hard 60s) with slow test report (top 10) and guidance suggestions.
- Environment variables: `ROBOT_SF_PERF_RELAX` (suppress soft breach enforcement) and `ROBOT_SF_PERF_ENFORCE` (escalate soft breaches to failures).
- Shared performance utilities (`tests/perf_utils/`): policy, reporting, guidance, minimal scenario matrix helper.
- Refactored benchmark integration tests (resume, reproducibility) to use minimal matrix for faster deterministic runs.
- Synthetic slow test and guidance validation tests.

### Added
- Classic Interactions PPO Visualization (Feature 128) – deterministic PPO-driven classic interaction scenario visualization script (`examples/classic_interactions_pygame.py`) with:
  - Constants-based configuration (no CLI) and dry-run validation path
  - Deterministic seed ordering & structured episode summaries (scenario, seed, steps, outcome, success/collision/timeout booleans, recorded)
  - Graceful recording guard (moviepy/ffmpeg optional) with informative skip notes
  - Logging verbosity toggle (`LOGGING_ENABLED`) and performance-friendly frame sampling
  - Improved model load error guidance (actionable download/help message)
  - Headless safety via SDL_VIDEODRIVER=dummy detection
  - Reward fallback integration log (env already falls back to simple_reward)
  - Summary table printer helper for human-readable output
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

- Classic interactions refactor (Feature 139): Extracted small reusable visualization and formatting helpers into `robot_sf.benchmark` (`visualization.py`, `utils.py`) and added contract tests and a dry-run smoke test. See docs/dev/issues/classic-interactions-refactor/design.md.
- **Complete Documentation**: Step-by-step quickstart guide with example workflows
- **Performance Validation**: 20-25 steps/second with linear parallel scaling

### Changed
- Classic interactions pygame demo now respects per-scenario `map_file` entries in the scenario matrix: on each selected scenario it loads the referenced SVG (via converter) or JSON map definition and injects a single-map `MapDefinitionPool` into the environment config. Falls back gracefully (with a warning) to the default pool if loading fails.
- SVG map conversion now validates presence of at least one `robot_route_*_*` path; missing robot routes raises a clear `ValueError` (callers can fallback to default maps) preventing downstream division-by-zero in simulator initialization. Added richer conversion logging (route and zone counts).
- Enhanced baseline planner interface with unified PlannerProtocol
- Improved test coverage with comprehensive benchmark validation
- Updated main documentation with prominent benchmark platform section
- Simplified video encode invocation by replacing signature introspection with ordered fallback attempts (keyword → positional → minimal) for improved maintainability and clearer failure surface.

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

### Fixed
- Video artifact manifest now emits a per-episode `skipped` entry with `moviepy-missing` note instead of silently omitting episodes when SimulationView encoding is unavailable for that episode only.
- Robot / multi-robot environments now gracefully fallback to `simple_reward` when `reward_func=None` is passed via factory functions, preventing a `TypeError: 'NoneType' object is not callable` during `env.step` (affects new classic interactions PPO visualization demo).

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
