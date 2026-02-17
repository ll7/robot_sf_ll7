# Camera-Ready Benchmark Plan (2026-02-17)

## 1. Goal

Deliver a **single, reproducible, config-driven benchmark campaign pipeline** that:

1. Runs current scenario suites with multiple planners (baseline-safe + experimental profiles).
2. Produces camera-ready summary artifacts (machine-readable and publication-ready tables).
3. Captures provenance and reproducibility metadata for every planner run and campaign-level bundle.
4. Exports a publication bundle suitable for GitHub Release + Zenodo DOI archiving.

This plan starts from existing functionality in:
- `robot_sf_bench run` (map-runner benchmark execution)
- `robot_sf.benchmark.aggregate` (aggregations/CI)
- `robot_sf.benchmark.artifact_publication` (bundle export)

and closes missing orchestration/reporting gaps.

## 2. Current Capability Baseline

### 2.1 What already works

- Scenario manifests are stable and load correctly:
  - `configs/scenarios/classic_interactions.yaml` (20)
  - `configs/scenarios/francis2023.yaml` (25)
  - `configs/scenarios/classic_interactions_francis2023.yaml` (45)
- Planner readiness gates exist (`baseline-safe`, `paper-baseline`, `experimental`).
- `robot_sf_bench run` executes map-based planners and writes schema-validated `episodes.jsonl`.
- Publication helper exists for **single run directory** bundle export.

### 2.2 Missing pieces for camera-ready workflow

1. No canonical **campaign orchestrator** to execute a planner matrix in one run.
2. No canonical **campaign-level artifact contract** (what exact files are required).
3. No consolidated **cross-planner result tables** (CSV/Markdown) at campaign scope.
4. No campaign-level provenance manifest linking scenario matrix, seed policy, planner configs, and output hashes.
5. No standard handoff between campaign outputs and publication bundle export.

## 3. Target Architecture

## 3.1 New core concept: Benchmark Campaign

A campaign is a set of planner runs over one scenario matrix + seed policy.

- Input:
  - campaign config (YAML)
  - scenario matrix path
  - planner entries (`algo`, profile, optional algo config, prereq policy)
  - global runtime options (`workers`, `horizon`, `dt`, `record_forces`, `adapter_impact_eval`)
- Execution:
  - run each planner using `run_map_batch` via existing `run_batch` path
- Output:
  - campaign root under `output/benchmarks/camera_ready/<campaign_id>/...`

## 3.2 Artifact contract (campaign)

Required artifacts:

1. `campaign_manifest.json`
   - campaign id, timestamp, git hash, host/python metadata
   - scenario matrix path/hash
   - seed policy
   - planner matrix with per-planner config hash
   - command/config provenance

2. `runs/<planner_key>/`
   - `episodes.jsonl` (per planner run output)
   - `summary.json` (runner summary)

3. `reports/campaign_summary.json`
   - per-planner aggregate metrics
   - cross-planner comparison blocks
   - warnings/missing-data notes

4. `reports/campaign_table.csv`
   - one row per planner
   - camera-ready key metrics

5. `reports/campaign_table.md`
   - markdown version for papers/docs

6. `reports/campaign_report.md`
   - narrative summary, methods context, and caveats

7. `publication/`
   - campaign export bundle (folder + tar.gz + checksums)

## 3.3 Planner sets

- `baseline-safe` set: `goal`, `social_force`, `orca`
- `experimental` set: `ppo`, `socnav_sampling`, `sacadrl`, `socnav_bench`

Note: placeholders (`rvo`, `dwa`, `teb`) remain blocked by readiness policy.

## 4. Result Representation (Camera-Ready)

## 4.1 Primary metrics table (per planner)

Columns:
- `planner`
- `episodes`
- `success_mean`
- `collision_mean`
- `time_to_goal_norm_mean`
- `path_efficiency_mean`
- `near_misses_mean`
- `comfort_exposure_mean`
- `jerk_mean`
- `snqi_mean`

If CI available from aggregation, include:
- `success_mean_ci_low`, `success_mean_ci_high`
- `collision_mean_ci_low`, `collision_mean_ci_high`
- `snqi_mean_ci_low`, `snqi_mean_ci_high`

## 4.2 Narrative report content

`campaign_report.md` sections:
1. Campaign metadata and reproducibility
2. Planner matrix and readiness profile notes
3. Key ranking and trade-off summary
4. Safety/comfort caveats (force missing, fallback modes)
5. Archival references (bundle path, checksums, DOI placeholder)

## 5. Archiving Strategy

1. Use campaign output as publication root.
2. Export bundle via `artifact_publication.export_publication_bundle`.
3. Include publication metadata placeholders:
   - repository URL
   - release tag placeholder
   - DOI placeholder
4. Keep all generated outputs under canonical `output/` root.

## 6. Implementation Plan

1. Add camera-ready campaign orchestrator module under `robot_sf/benchmark/`.
2. Add CLI entry script under `scripts/tools/`.
3. Add config templates in `configs/benchmarks/`.
4. Add campaign report/table generators.
5. Add tests for config parsing/report output contracts.
6. Execute multi-planner smoke campaign and validate artifacts.
7. Document usage and artifact tree in docs.

## 7. Acceptance Criteria

A campaign is accepted when:

1. Pipeline runs from one command with a campaign config.
2. All required artifacts listed in Section 3.2 are produced.
3. Baseline-safe campaign runs cleanly on current environment.
4. Experimental campaign executes and records any planner fallback/degradation explicitly.
5. Publication bundle exports with checksums and manifest.
6. Tests for config/report contracts pass.

## 8. Non-Goals (this iteration)

1. Rewriting core planner algorithms.
2. Enforcing strict paper-baseline PPO provenance gates by default.
3. CI-wide full campaign execution on every PR (costly).

## 9. Risks and Mitigations

- Risk: experimental planners fail due optional deps/models.
  - Mitigation: prereq policy in config + explicit run status reporting.
- Risk: high runtime for full 45-scenario x multi-planner matrix.
  - Mitigation: provide smoke and full presets; document expected runtime.
- Risk: mixed fallback behavior reduces comparability.
  - Mitigation: include fallback/status fields in campaign summary.

## 10. Implementation Status (This Branch)

Implemented artifacts and entry points:

- campaign engine:
  - `robot_sf/benchmark/camera_ready_campaign.py`
- CLI:
  - `scripts/tools/run_camera_ready_benchmark.py`
- campaign presets:
  - `configs/benchmarks/camera_ready_smoke_all_planners.yaml`
  - `configs/benchmarks/camera_ready_baseline_safe.yaml`
  - `configs/benchmarks/camera_ready_all_planners.yaml`
- tests:
  - `tests/benchmark/test_camera_ready_campaign.py`
- docs:
  - `docs/benchmark_camera_ready.md`

Compatibility fix included:

- `socnav_bench` now forwards `allow_fallback` in `_build_policy`:
  - `robot_sf/benchmark/map_runner.py`
  - regression test:
    - `tests/benchmark/test_map_runner_utils.py`

Validation runs executed:

- smoke campaign (`7` planners, `1` scenario): success (`7/7`)
- full campaign (`7` planners, `45` scenarios, `1` seed): success (`7/7`, `315` episodes)
- publication bundle export: success
- baseline-safe calibration campaign (`3` planners, `45` scenarios, `3` eval seeds):
  success (`3/3`, `405` episodes)
- baseline-safe verification campaign with SNQI enabled (`3` planners, `45` scenarios, `3` eval seeds):
  success (`3/3`, `405` episodes)
- full all-planners verification with SNQI + multi-seed (`7` planners, `45` scenarios, `3` eval seeds):
  success (`7/7`, `945` episodes)

## 11. Publication Freeze Follow-Ups

Resolved in this branch:

1. SNQI calibration for table completeness
   - Added canonical camera-ready SNQI config assets:
     - `configs/benchmarks/snqi_weights_camera_ready_v1.json`
     - `configs/benchmarks/snqi_baseline_camera_ready_v1.json`
   - Wired into campaign presets:
     - `configs/benchmarks/camera_ready_smoke_all_planners.yaml`
     - `configs/benchmarks/camera_ready_baseline_safe.yaml`
     - `configs/benchmarks/camera_ready_all_planners.yaml`
   - Verified smoke campaign table now emits numeric `snqi_mean` values.

2. Multi-seed evaluation policy
   - Switched canonical full presets to seed sets:
     - `configs/benchmarks/camera_ready_baseline_safe.yaml`
     - `configs/benchmarks/camera_ready_all_planners.yaml`
   - Seed policy now uses:
     - `mode: seed-set`
     - `seed_set: eval`
     - `seed_sets_path: configs/benchmarks/seed_sets_v1.yaml`

Still pending:

3. Release metadata finalization
   - Replace placeholder release tag and DOI values in camera-ready config
     before public archival.

## 12. Documentation + Visualization Helper Roadmap

To make benchmark outputs easier to review and publish, implement the
following helper layer on top of existing campaign artifacts.

1. Report builder helper (`scripts/tools/benchmark_report_build.py`)
   - Input: `reports/campaign_summary.json`
   - Output: normalized markdown brief with:
     - best/worst planner per key metric
     - runtime leaderboard
     - fallback/degraded planner flags
   - Why: reduces manual interpretation work before PR/release.

2. Artifact explorer helper (`scripts/tools/benchmark_artifact_inspect.py`)
   - Input: campaign root path
   - Output: structural validation + human summary:
     - required files present/missing
     - schema version, git hash, command provenance
     - publication bundle completeness
   - Why: faster publication freeze checks.

3. Visualization helper (`scripts/tools/benchmark_plot.py`)
   - Input: `campaign_table.csv` or `campaign_summary.json`
   - Output:
     - planner metric bar charts (success/collision/SNQI)
     - runtime vs quality scatter plot
     - optional SVG/PNG exports under `reports/figures/`
   - Why: camera-ready figure generation from canonical artifacts.

4. Notebook template for deep dives (`docs/notebooks/benchmark_analysis.ipynb`)
   - Reusable notebook loading one campaign and comparing two runs.
   - Includes significance checks for multi-seed summaries.
   - Why: supports paper iteration and regression analysis.

5. PR-facing summary integration
   - Add optional step in `scripts/dev/pr_ready_check.sh` to print:
     - latest campaign id
     - planner runtime table
     - any warning/fallback counts
   - Why: puts benchmark signal directly into review workflow.

6. Potential tooling stack
   - Dataframes: `pandas`
   - Plots: `matplotlib` (default) with optional `seaborn`
   - Rich terminal summaries: `tabulate` or `rich`
   - Why: low-friction, already common in research/dev environments.
