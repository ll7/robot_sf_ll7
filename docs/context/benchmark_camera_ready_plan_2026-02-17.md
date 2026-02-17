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
