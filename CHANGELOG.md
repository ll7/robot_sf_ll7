# Changelog

All notable changes to the Robot SF project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Occupancy grid polish: ego-frame transforms applied consistently, query aggregation returns per-channel means without scaling errors, new quickstart/advanced examples (`examples/quickstart/04_occupancy_grid.py`, `examples/advanced/20_occupancy_grid_workflow.py`), and refreshed guide linked from the docs index.
- Automated research reporting pipeline (feature 270-imitation-report): multi-seed aggregation, statistical hypothesis evaluation (paired t-tests, effect sizes, threshold comparisons), publication-quality figure suite (learning curves, sample efficiency, distributions, effect sizes, sensitivity), ablation matrix orchestration, telemetry section, and programmatic + CLI workflows (`scripts/research/generate_report.py`, `scripts/research/compare_ablations.py`). Includes success criteria tests and demo (`examples/advanced/17_research_report_demo.py`).
- Research reporting polish: metadata manifest aligned with `report_metadata` schema, schema validation tests for metrics/hypotheses, and smoke/performance harnesses (`scripts/validation/test_research_report_smoke.sh`, `scripts/validation/performance_research_report.py`, `tests/research/test_performance_smoke.py`, `tests/research/test_schemas.py`).
- Multi-extractor training now auto-collects convergence/sample-efficiency metrics, baseline comparisons, and learning-curve/reward-distribution figures, emitting schema-compliant summaries (`summary.json`/`summary.md`) plus legacy `complete_results.json`.
- New extractor report generator: `scripts/research/generate_extractor_report.py` converts multi-extractor `summary.json` into research-ready `report.md`/`report.tex` with figures, reproducibility metadata, and baseline comparisons.

### Added
- Map Verification Workflow (Feature 001-map-verification)
  - Single-command map validation tool (`scripts/validation/verify_maps.py`) for SVG map quality checks
  - Rule-based validation engine checking file readability, SVG syntax, file size, and layer organization
  - Scope filtering supporting 'all', 'ci', 'changed', specific filenames, or glob patterns
  - Structured JSON/JSONL manifest output for tooling and dashboard integration
  - CI mode with strict exit codes for automated quality gates
  - Loguru-based diagnostics with human-readable console output
  - Map inventory system with tag-based classification and filtering
  - Verification results include timing, rule violations, and remediation hints
  - Documentation in `docs/SVG_MAP_EDITOR.md` and `specs/001-map-verification/quickstart.md`
  - Sample manifest artifacts under `output/validation/`
- Run tracking & telemetry for the imitation pipeline (Feature 001-performance-tracking)
  - Progress tracker with deterministic step ordinals, ETA smoothing, and manifest-backed step history
  - JSONL manifests enriched with telemetry snapshots, rule-based recommendations, and perf-test results stored under `output/run-tracker/`
  - CLI tooling (`scripts/tools/run_tracker_cli.py`) for status/watch/list/show/export/summary plus optional TensorBoard mirroring
  - Performance smoke CLI (`scripts/telemetry/run_perf_tests.py`) that wraps the existing validation harness and records pass/soft-breach/fail statuses with remediation hints
  - Documentation updates spanning quickstart, dev guide, and docs/README.md so teams can enable the tracker and interpret telemetry in CI or local runs
  - CI guard step invoking `scripts/validation/run_examples_smoke.py --perf-tests-only` so the tracker smoke + telemetry perf wrapper fail fast before pytest
- PPO Imitation Learning Pipeline (Feature 001)
  - Expert PPO training workflow with convergence criteria and evaluation schedules
  - Trajectory dataset collection and validation utilities
  - Behavioral cloning (BC) pre-training from expert demonstrations
  - PPO fine-tuning with warm-start from pre-trained policies
  - Comparative metrics CLI for sample-efficiency analysis
  - Playback and inspection tool for trajectory datasets
  - Bootstrap confidence intervals for metric aggregation
  - Complete artifact lineage tracking (expert → dataset → pre-trained → fine-tuned)
  - Configuration dataclasses for all imitation workflows
  - Integration tests for end-to-end pipeline validation
  - Sample-efficiency target: ≤70% of baseline timesteps to convergence
  - Documentation in `docs/dev_guide.md` and `specs/001-ppo-imitation-pretrain/quickstart.md`
  - Default imitation configs for behavioral cloning and PPO fine-tuning (`configs/training/ppo_imitation/bc_pretrain.yaml`, `configs/training/ppo_imitation/ppo_finetune.yaml`)
- Canonical artifact root enforcement and tooling (Feature 243)
  - Introduced `output/` hierarchy as single destination for coverage, benchmark, recording, wandb, and tmp artifacts
  - Added migration helper (`scripts/tools/migrate_artifacts.py`) and guard (`scripts/tools/check_artifact_root.py`) with console entry point and regression tests
  - Wired guard + migration into CI workflow, publishing artifacts from canonical paths only
  - Refreshed core docs (`docs/dev_guide.md`, `docs/coverage_guide.md`, `docs/README.md`, root `README.md`) with policy overview, quickstart links, and updated coverage instructions
  - Extended quickstart guidance to cover guard execution, artifact overrides, and validation expectations
- Comprehensive configuration architecture documentation (#244)
  - Created `docs/architecture/configuration.md` with configuration precedence hierarchy
  - Documented three-tier precedence system: Code Defaults < YAML < Runtime
  - Added migration guide from legacy config classes to unified config
  - Documented all configuration modules (canonical vs legacy)
  - Linked from `docs/README.md` and `docs/dev_guide.md`
- Automated example smoke harness (`scripts/validation/run_examples_smoke.py`, `tests/examples/test_examples_run.py`) wired into validation workflow (#245)

### Changed
- Expert PPO training and trajectory collection now honor scenario YAML entries, including map files, simulation overrides, and scenario identifiers, while publishing `scenario_coverage` metadata consistent with dataset validators.
- **[BREAKING for internal imports]** Consolidated utility modules into single `robot_sf/common/` directory (#241)
  - Moved `robot_sf/util/types.py` → `robot_sf/common/types.py`
  - Moved `robot_sf/utils/seed_utils.py` → `robot_sf/common/seed.py` (renamed)
  - Moved `robot_sf/util/compatibility.py` → `robot_sf/common/compat.py` (renamed)
  - Removed empty `robot_sf/util/` and `robot_sf/utils/` directories
- Example catalog reorganization and automation improvements (#245)
  - Moved benchmark and plotting scripts into dedicated `examples/benchmarks/` and `examples/plotting/` tiers
  - Regenerated manifest-backed `examples/README.md` and refreshed docs (`README.md`, `docs/README.md`, `docs/benchmark*.md`, `docs/distribution_plots.md`) to reference new paths
  - Updated `examples/examples_manifest.yaml` metadata (tags, CI flags, summaries) and added quick links from docs
  - Imitation pipeline example now auto-selects simulator backends and generates run-specific BC/PPO configs under `output/tmp/imitation_pipeline/` to keep CLI invocations aligned with script requirements
- Visualization stack ownership clarified: the Full Classic pipeline (`robot_sf.benchmark.full_classic.visuals.generate_visual_artifacts`) is now the canonical path that emits manifest-backed plot/video artifacts; the legacy helper API (`robot_sf.benchmark.visualization.*`) is deprecated for benchmark runs and retained only for ad-hoc JSONL plotting.

### Documentation
- Reorganized documentation index with categorized sections (#242)
  - Added clear navigation sections: Getting Started, Benchmarking & Metrics, Tooling, Architecture & Refactoring, Simulation & UI, Figures & Visualization, Performance & CI, Hardware & Environment
  - Added cross-links between core guides for improved discoverability
  - Normalized H1 headings and purpose statements across key documentation files
  - Collapsed legacy detailed index into expandable section for backward compatibility
  
### Migration Guide (Version 2.1.0)

**For robot_sf developers and contributors:**

All utility imports must be updated to reference `robot_sf.common`:

```python
# Before (old paths - no longer valid)
from robot_sf.util.types import Vec2D, RobotPose
from robot_sf.utils.seed_utils import set_global_seed
from robot_sf.util.compatibility import validate_compatibility

# After (new paths - required)
from robot_sf.common.types import Vec2D, RobotPose
from robot_sf.common.seed import set_global_seed
from robot_sf.common.compat import validate_compatibility

# Convenience imports also available:
from robot_sf.common import Vec2D, RobotPose, set_global_seed
```

**Why this change?**
- Eliminates navigation confusion from fragmented utility locations
- Improves IDE autocomplete and discoverability
- Reduces cognitive load for new contributors
- Establishes single canonical location for all shared utilities

**Impact:**
- ~50 import statements updated across codebase
- All 923 tests passing after migration
- No functional changes - pure reorganization

**For external consumers (if any):**
If your project imports from `robot_sf.util` or `robot_sf.utils`, update your imports using the patterns above. The behavior of all utilities remains unchanged.

### Added
- **Architecture Decoupling (Feature 149)**: Introduced simulator facade and backend/sensor registries scaffolding behind the existing factory pattern. Default backend is "fast-pysf" with future backend selection via unified config.
- Backend registry integrated into environment initialization: `BaseEnv` now resolves the simulator via a backend key (`env_config.backend`, default "fast-pysf") using `robot_sf.sim.registry`, with a safe fallback to legacy `init_simulators()` for full backward compatibility.
- **fast-pysf Integration Improvements (Feature 148)**: Enhanced fast-pysf integration with comprehensive testing and quality tooling
  - **Map Verification Enhancements (001-map-verification)**
    - Added informational rule `R005` (layer statistics) emitted when Inkscape-labeled groups (`<g inkscape:label="…">`) are present
    - Enhanced `R004` message and remediation guidance (descriptive labeling for obstacles/spawns/waypoints)
    - Inserted Map Verification section into `docs/benchmark.md` (CI invocation, rule table, manifest structure, extension guidance)
    - Added labeled example SVG (`maps/svg_maps/labeled_example.svg`) for demonstrating layer labeling and future semantic tagging
    - Improved internal type hints (`_load_inventory` return type, expanded docstrings) for verification modules
    - Test coverage extended (`test_layer_stats_info`) validating scope resolution and future R005 visibility

  - **Unified Test Suite**: Single `uv run pytest` command now executes both robot_sf (881 tests) and fast-pysf (12 tests) for total 893 tests
  - **Quality Tooling Extension**: Extended ruff linting, ty type checking, and coverage reporting to include fast-pysf subtree
  - **Type Annotations**: Added comprehensive type hints to fast-pysf public APIs (map_loader, forces, simulator, scene modules)
  - **Configuration**: Per-file ignores in pyproject.toml for gradual quality adoption, ty configuration includes fast-pysf/pysocialforce
  - **Code Quality**: Fixed circular imports, removed dead code, alphabetized imports, replaced wildcard imports
  - **Documentation**: Created annotation_plan.md with numba compatibility guidelines and implementation strategy
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
  - `examples/plotting/coverage_example.py`: Programmatic usage examples
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
- Automated Research Reporting: Multi-seed aggregation & completeness (Feature 270-imitation-report)
  - Seed orchestration (`orchestrate_multi_seed`) combining baseline/pretrained manifests
  - Per-seed manifest parsing (`extract_seed_metrics`) tolerant of JSON/JSONL tracker outputs
  - Completeness scoring (`compute_completeness_score`) with PASS/PARTIAL classification
  - Standardized seed failure logging (`log_seed_failure`) and graceful missing-manifest handling
  - Seed summary table & completeness JSON artifact (`completeness.json`) rendered into `report.md`
  - Tracker manifest parsing (`parse_tracker_manifest`) for run-level status integration
  - Extended aggregation exports (JSON + CSV) and hypothesis evaluation incorporated in multi-seed flow
  - All US2 tasks (T033–T043) implemented with unit & integration test coverage
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
- Full Classic visuals: auto video renderer now falls back to synthetic when all SimulationView encodes fail, recording the downgrade in `performance_visuals`.

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
- **fast-pysf Type Annotations (Feature 148)**: Enhanced type safety across fast-pysf public APIs
  - Added return type annotations to all Force classes (`__call__` methods return `np.ndarray`)
  - Added type hints to Simulator and Simulator_v2 classes (step methods, state management)
  - Improved PedState class with proper array type annotations (`np.ndarray | None` for optional attributes)
  - Added `str | Path` type support to map_loader.load_map() for flexible path handling
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
- **fast-pysf Code Quality Issues (Feature 148 / PR #236)**: Resolved 24 PR review comments
  - Fixed circular import in forces.py (changed `from pysocialforce import logger` to `from pysocialforce.logging import logger`)
  - Removed dead code from scene.py (commented desired_directions method)
  - Alphabetized imports in simulator.py for consistency
  - Removed duplicate simulator assignment in ex09_inkscape_svg_map.py
  - Replaced wildcard import with explicit import in TestObstacleForce.py
  - Fixed file path resolution in test_map_loader.py to work with dynamic paths
  - Added per-file ruff ignores for complexity and print statements in examples
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
