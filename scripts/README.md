# Scripts Directory

Executable entry points for Robot SF, organized by functional area. The root-level
command inventory is **generated** from [`scripts/catalog.yaml`](catalog.yaml); edit the
catalog (or script docstrings/`--help`) rather than this file's generated sections.

## Quick Navigation by Task

* **Train a robot policy** → `scripts/training/train_ppo.py` ([Training](#directory-overview))
* **Train/evaluate predictive planner** → `scripts/training/train_predictive_planner.py` and `scripts/validation/run_predictive_success_campaign.py`
* **Run benchmarks** → `scripts/classic_benchmark_full.py` or `scripts/benchmark_workers.py`
* **Search policy candidates** → `scripts/validation/run_policy_search_candidate.py` plus comparison/failure tools under `scripts/tools/`
* **Analyze results** → `scripts/research/generate_report.py`, `scripts/generate_figures.py`, `scripts/ranking_table.py`
* **Validate changes** → `scripts/validation/`
* **Work with SNQI metrics** → SNQI commands in [Root-Level Entry Points](#root-level-entry-points) and [`README_SNQI_WEIGHTS.md`](README_SNQI_WEIGHTS.md)
* **Check performance** → `scripts/validation/performance_smoke_test.py`
* **Migrate artifacts** → `scripts/tools/migrate_artifacts.py`

## Status Semantics And Migration Policy

Prefer maintained subdirectories (`training/`, `validation/`, `tools/`, `dev/`,
`coverage/`) for new workflows. Root-level scripts are kept only when they are still a
reviewed public command, a compatibility shim, or a bounded debug utility.

Statuses used in the generated table:

* `canonical` — reviewed public command; safe to build on.
* `compatibility` — retained for migration; prefer the listed replacement. Retired
  entry points marked *fails closed* exit with guidance instead of running old behavior.
* `debug-only` — bounded debugging utility; not a supported public workflow.
* `archive-candidate` — candidate for archival; do not extend.

To retire or move a command, update `scripts/catalog.yaml` (status, replacement,
rationale) and re-render; `scripts/dev/check_scripts_catalog.py` fails when a root
script is missing from the catalog, a replacement path does not resolve, or a
compatibility entry has neither replacement nor rationale.

## Root-Level Entry Points

<!-- BEGIN GENERATED:root-entry-point-status -->
| Command | Status | Purpose / canonical path or action |
| --- | --- | --- |
| `__init__.py` | canonical | Import support making `scripts.*` importable for tests and type-checking. |
| `advise-provider-routing.py` | canonical | Forward provider advice to the canonical shared advisor without local route tables. |
| `analyze_feature_extractors.py` | compatibility | Statistical analysis for feature extractor comparison. Prefer `scripts/research/generate_extractor_report.py`. |
| `audit_exemplar_bundles.py` | canonical | Audit exemplar-bundle checksums and provenance (issues #4920/#5005). |
| `benchmark02.py` | compatibility | Retired root performance benchmark. Prefer `scripts/validation/performance_smoke_test.py`. Fails closed. |
| `benchmark_ped_apf_models.py` | debug-only | Narrow APF model comparison helper. |
| `benchmark_ped_policy_collisions.py` | debug-only | Narrow pedestrian-policy collision analysis helper. |
| `benchmark_planner.py` | debug-only | Local planner timing probe. |
| `benchmark_repro_check.py` | compatibility | Create a minimal reproducibility-test scenario. Prefer `Prefer benchmark release/validation tools under scripts/tools/.`. |
| `benchmark_threshold_sensitivity.py` | canonical | Threshold sensitivity analysis for replay-rich benchmark rows. |
| `benchmark_workers.py` | canonical | Maintained worker-throughput benchmark helper. |
| `classic_benchmark_full.py` | canonical | Full Classic Interaction Benchmark CLI. |
| `collect_slow_tests.py` | canonical | Parse `pytest --durations=N` output into structured JSON. |
| `compare_slow_tests.py` | canonical | Compare before/after slow-test JSON captures. |
| `convert_pickle_to_jsonl.py` | compatibility | Convert legacy multi-episode pickle files to per-episode JSONL. Prefer `Retained until legacy pickle artifacts retire.`. |
| `debug_ped_apf.py` | debug-only | Interactive APF debugging. |
| `debug_ped_discrete.py` | debug-only | Pedestrian discrete-action debugging. |
| `debug_ped_forces.py` | debug-only | Pedestrian force debugging. |
| `debug_ped_policy.py` | debug-only | Pedestrian policy debugging. |
| `debug_ped_policy_differential_drive.py` | debug-only | Differential-drive pedestrian policy debugging. |
| `debug_random_policy.py` | debug-only | Manual/random-policy visual debug helper. |
| `debug_trained_policy.py` | archive-candidate | Old trained-policy debug helper; prefer examples and factory-based smoke tests. |
| `demo_jsonl_recording.py` | compatibility | JSONL recording and playback demonstration. Prefer `Prefer examples and render docs for new documentation.`. |
| `evaluate.py` | compatibility | Legacy policy evaluation helper. Prefer `Prefer config-driven benchmark runner tools.`. |
| `example_snqi_workflow.py` | canonical | Complete SNQI workflow example with generated data. |
| `export_issue_4268_trace_episode.py` | canonical | Export the issue #4268 single-episode doorway trace bundle via the map-runner trace path. |
| `export_issue_4848_group_crossing_exemplars.py` | canonical | Export issue #4848 group-crossing exemplar trace bundles. |
| `export_issue_4891_head_on_corridor_exemplars.py` | canonical | Export issue #4891 head-on-corridor exemplar trace bundles. |
| `failure_extractor.py` | canonical | Extract worst episodes by chosen metric from episodes JSONL. |
| `generate_figures.py` | canonical | Benchmark figure generation from episode JSONL. |
| `generate_video_contact_sheet.py` | canonical | Episode-frame thumbnail contact-sheet generation. |
| `hparam_opt.py` | compatibility | Retired root Optuna entrypoint. Prefer `scripts/training/launch_optuna_expert_ppo.py`. Fails closed. |
| `multi_extractor_training.py` | compatibility | Orchestrate PPO runs across configured feature extractors. Prefer `scripts/research/generate_extractor_report.py`. |
| `play_recordings.py` | compatibility | Playback recorded episodes. Prefer `Prefer robot_sf.render playback modules for new playback workflows.`. |
| `ranking_table.py` | canonical | Ranking-table generation from benchmark episode JSONL. |
| `read-active-ledger.py` | canonical | Read a compact snapshot of active common-Git-dir autopilot ledgers. |
| `recompute_snqi_weights.py` | canonical | Recompute SNQI weights using different strategies. |
| `render_multi_planner_trajectory_overlay.py` | canonical | Overlay multi-planner trajectories into one provenance-stamped figure. |
| `replay_episode_figure.py` | canonical | Generate replay-derived figure artifacts from campaign episode rows. |
| `resolve-route.py` | canonical | Forward route resolution to the canonical shared resolver without copying its policy. |
| `review-agent-run.sh` | canonical | Private review wrapper for delegated agent-run artifacts. |
| `run_classic_interactions.py` | canonical | Classic interaction scenario matrix runner. |
| `run_social_navigation_benchmark.py` | compatibility | Older all-in-one social navigation benchmark runner. Prefer `Prefer config-driven tools like classic_benchmark_full.py.`. |
| `save-codex-token-checkpoint.py` | canonical | Build a compact token-saving checkpoint without duplicating route policy. |
| `scale_svgs_to_50m.py` | debug-only | One-off SVG coordinate scaling utility. |
| `seed_variance.py` | canonical | SNQI seed-variance analysis across benchmark episodes. |
| `select_exemplar_episodes.py` | canonical | Select median/best/worst exemplar episodes from campaign JSONL into a manifest. |
| `snqi_sensitivity_analysis.py` | canonical | Full SNQI sensitivity analysis with visualizations. |
| `snqi_weight_optimization.py` | canonical | Advanced SNQI weight optimization with differential evolution. |
| `summarize-agent-runs.py` | canonical | Summarize compact delegated-agent artifacts from the common Git directory. |
| `test_planner_collision.py` | debug-only | Planner collision debug probe; not a pytest module. |
| `training_a2c.py` | compatibility | Retired A2C root entrypoint. Prefer `scripts/training/train_ppo.py`. Fails closed. |
| `training_ped_ppo.py` | compatibility | PPO training for pedestrian environments. Prefer `Retained for tests; prefer config-first training under scripts/training/.`. |
| `training_ped_ppo_differential_drive.py` | compatibility | Pedestrian PPO variant for differential-drive robots. Prefer `Retained for tests; prefer config-first training under scripts/training/.`. |
| `training_ppo.py` | compatibility | Retired root PPO entrypoint. Prefer `scripts/training/train_ppo.py`. Fails closed. |
| `update_deps.sh` | canonical | Refresh Python dependencies with uv sync. |
| `update_svg_viewbox.py` | debug-only | One-off SVG viewBox utility. |
| `validate_snqi_scripts.py` | canonical | SNQI script smoke validator. |
| `wandb_ppo_training.py` | compatibility | Retired root W&B PPO entrypoint. Prefer `scripts/training/train_ppo.py`. Fails closed. |
<!-- END GENERATED:root-entry-point-status -->

## Directory Overview

Nested directories keep their own guides; see each package for per-script detail.
Only `scripts/dev/` currently ships a README. Note: the retired root profiling
benchmark remains a fail-closed module command (`uv run python -m scripts.benchmark`);
prefer `validation/performance_smoke_test.py`.

<!-- BEGIN GENERATED:directory-overview -->
| Directory | Role |
| --- | --- |
| `PPO_training/` | Legacy PPO training variants |
| `adversarial/` | Adversarial scenario tools |
| `analysis/` | Trace analyzers and crossing-failure packs |
| `benchmark/` | Benchmark reporting and campaign helpers |
| `carla_bridge/` | CARLA bridge diagnostics |
| `ci/` | CI helper scripts |
| `coverage/` | Coverage reporting tools |
| `data/` | Data files |
| `demo/` | Demo and smoke scripts |
| [`scripts/dev/`]( dev/ ) | Development helpers and automation |
| `diagnostics/` | Diagnostic tools |
| `manual_control/` | Manual control interface |
| `models/` | Model-related helpers |
| `perf/` | Performance baselines |
| `quality/` | Quality tooling |
| `reporting/` | Report generation |
| `repro/` | Reproducibility helpers |
| `research/` | Research report and ablation tools |
| `telemetry/` | Telemetry wrappers for performance tests |
| `tools/` | Utilities: tracker CLI, comparisons, artifact guards |
| `training/` | Config-first training workflows |
| `validation/` | Validation suites and shell checks |
<!-- END GENERATED:directory-overview -->

## Common Patterns

Many visualization/benchmark scripts support headless execution:

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python scripts/<script>.py
```

Override the artifact destination:

```bash
export ROBOT_SF_ARTIFACT_ROOT=/path/to/custom/output
```

Most scripts support `--help`, `--config <path>`, `--output <path>`, `--debug`,
and `--log-level DEBUG|INFO|WARNING|ERROR`.

Behavioral-cloning pre-training requires the imitation group:
`uv sync --group imitation` then `uv run --group imitation ...`.

## Quick Start Workflows

```bash
# Train (canonical PPO)
uv run python scripts/training/train_ppo.py --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml
# Benchmark + analyze
uv run python scripts/classic_benchmark_full.py && uv run python scripts/generate_figures.py
# Validate
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python scripts/validation/performance_smoke_test.py
# SNQI optimization
uv run python scripts/snqi_weight_optimization.py --episodes episodes.jsonl --baseline baseline_stats.json --output optimized_weights.json
```

## Contributing

1. Follow naming conventions: descriptive snake_case filenames.
2. Add docstrings: purpose, usage, and details at the top of the file.
3. Register every new root-level command in `scripts/catalog.yaml` — the checker
   fails on undocumented root scripts. Nested-directory scripts follow their own guide.
4. Use factory patterns (`make_robot_env()`) and write outputs under `output/`.
5. Add validation (smoke test) and document special dependency extras via
   `required_extras` in the catalog.

## Related Documentation

* Development guide: `docs/dev_guide.md`
* SNQI quick start / weights: [`QUICK_START.md`](QUICK_START.md), [`README_SNQI_WEIGHTS.md`](README_SNQI_WEIGHTS.md)
* Dev helpers: [`dev/README.md`](dev/README.md)
* Examples: `examples/README.md`; artifact policy: `specs/243-clean-output-dirs/quickstart.md`

---

Maintained By: Robot SF Development Team
