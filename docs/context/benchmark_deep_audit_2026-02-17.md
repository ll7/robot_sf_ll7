# Benchmark Deep Audit (2026-02-17)

## Scope

This audit reviews camera-ready benchmark readiness across:

- algorithm execution correctness (especially PPO)
- benchmark reproducibility/provenance
- result interpretability and comparability
- remaining gaps that should become benchmark-milestone issues

Primary evidence run:

- `output/benchmarks/camera_ready/camera_ready_all_planners_deep_audit_20260217_150448`

Supporting reference runs:

- `output/benchmarks/camera_ready/camera_ready_all_planners_full_run_20260217_131510`
- `output/benchmarks/camera_ready/camera_ready_all_planners_full_run_ppo_native_fix_20260217_140732`
- `output/benchmarks/camera_ready/camera_ready_ppo_only_full_calib_vmax2_20260217_145908`
- `output/benchmarks/camera_ready/camera_ready_ppo_only_full_vmax3_calib_vmax3_20260217_150029`

## Big Picture: What The Benchmark Contributes

The current camera-ready benchmark already provides a reproducible, cross-planner evaluation pipeline with:

- shared scenario matrix (`classic_interactions_francis2023`)
- fixed evaluation seed set (`eval`)
- shared occupancy-grid observation substrate in the environment
- unified artifact output (campaign summary/table/report + publication bundle)
- run-time provenance (`invoked_command`, per-planner timing)
- automated consistency analysis via `scripts/tools/analyze_camera_ready_campaign.py`

In practical terms, this is a complete end-to-end benchmarking pipeline for baseline + experimental planners.

## Verified Findings

### 1. PPO is no longer silently degraded to goal-like behavior

Confirmed by comparing old vs new full runs:

- old (`...131510`): PPO success `0.0444` (goal-like)
- current (`...150448`): PPO success `0.9111`

Root cause fixed in code path:

- dict-observation passthrough in `robot_sf/benchmark/map_runner.py`
- PPO dict mode in `configs/baselines/ppo_15m_grid_socnav.yaml`

### 2. PPO adapter-impact accounting is now coherent in current code path

Earlier stale runs showed `summary pending` + `episodes complete` mismatch.
Current deep-audit run analyzer shows no adapter-status mismatch finding.

### 3. PPO `v_max` (2.0 vs 3.0) is not the dominant failure/quality driver

A/B runs (`ppo-only full`) show effectively unchanged outcomes:

- same success (`0.9111`)
- same collision (`0.1037`)
- nearly identical SNQI

Conclusion: primary PPO quality limits are not explained by simple action upper-bound tuning.

### 4. The benchmark currently emits non-portable absolute map paths

Analyzer flagged all planners:

- every planner had `135` episodes with absolute `scenario_params.map_file`.

This hurts artifact portability across machines and weakens publication bundle reproducibility guarantees.

### 5. Experimental SocNav planners are still degraded by fallback mode

Deep-audit run still reports:

- `socnav_sampling` preflight `fallback`
- `socnav_bench` preflight `fallback`

This is explicit in outputs, but means these rows are not native-model benchmark-grade comparisons yet.

## Current Deep-Audit Result Snapshot

From `...deep_audit_20260217_150448` planner rows:

- `goal`: success `0.0444`, collision `0.0000`, snqi `-0.1083`
- `social_force`: success `0.9111`, collision `0.0370`, snqi `-1.3779`
- `orca`: success `0.8963`, collision `0.1185`, snqi `-1.8459`
- `ppo`: success `0.9111`, collision `0.1037`, snqi `-2.7278`
- `sacadrl`: success `0.9333`, collision `0.0519`, snqi `-3.1321`
- `socnav_sampling`: success `0.9111`, collision `0.0741`, snqi `-1.1118` (fallback)
- `socnav_bench`: success `0.9111`, collision `0.0741`, snqi `-1.1205` (fallback)

Interpretation:

- PPO is functionally active and competitive on success.
- PPO remains relatively collision-heavy and comfort-heavy (SNQI penalty), so it is not yet dominant under current metric weighting.
- SocNav experimental rows should be treated as degraded-fallback diagnostics until prereqs are native-ready.

## What Is Missing For Publication-Grade Freeze

1. Path portability in episode provenance (`map_file` relative paths).
2. Explicit readiness tiering in final tables/reports to distinguish native vs fallback experimental rows.
3. Per-scenario diagnostic slicing to explain where planners win/lose (current outputs are mostly aggregate).
4. Statistical comparison helpers (effect sizes / paired tests) for stronger claims.
5. Policy-env contract checks for learned planners (observation/action/kinematics compatibility guardrails).

## Suggested Execution Plan

1. Fix portable map path provenance in writer path and add regression tests.
2. Add report-level readiness badges and fallback impact warnings in campaign report/table.
3. Add scenario-family and scenario-level breakdown report generation.
4. Add paired-comparison analysis helper (`vs baseline` significance/effect summary).
5. Add learned-policy compatibility contract checks in preflight (hard fail or warn based on profile).

## Follow-Up Issues Created

- #517 Benchmark artifact portability: use repo-relative map_file paths in episodes
- #518 Camera-ready reporting: make fallback/degraded planner status explicit
- #519 Camera-ready diagnostics: add per-scenario and scenario-family breakdown reports
- #520 Benchmark preflight: add learned-policy compatibility contract checks

## Commands Used During Audit

```bash
# Full campaign (fresh evidence run)
LOGURU_LEVEL=INFO uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/camera_ready_all_planners.yaml \
  --label deep_audit

# Campaign consistency analysis
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/camera_ready/camera_ready_all_planners_deep_audit_20260217_150448

# PPO v_max A/B calibration checks
LOGURU_LEVEL=INFO uv run python scripts/tools/run_camera_ready_benchmark.py --config /tmp/ppo_only_full.yaml --label calib_vmax2
LOGURU_LEVEL=INFO uv run python scripts/tools/run_camera_ready_benchmark.py --config /tmp/ppo_only_full_vmax3.yaml --label calib_vmax3
```
