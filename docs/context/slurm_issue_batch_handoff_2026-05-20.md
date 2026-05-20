# SLURM Issue Batch Handoff 2026-05-20

Worktree: `/home/luttkule/git/robot_sf_ll7.worktrees/1391-1392-slurm-workflow`

Branch: `issue-1391-1392-slurm-workflow`

Current implementation commit for submitted jobs: `0620ea20`

## Scope

This pass handled the issues requested in the 2026-05-20 batch:

- #1358 bounded ORCA-residual learned local policy
- #1167 predictive planner obstacle-feature same-seed comparison
- #1344 paired nominal/stress AMV protocol
- #1354 fixed-spec cross-kinematics AMV evidence campaign
- #1391 feature-extractor SLURM output layout
- #1392 generic camera-ready benchmark SLURM launcher

## Implemented Locally

#1391:

- Canonicalized feature-extractor SLURM stdout/stderr to `output/slurm/`.
- Canonicalized feature-extractor training artifacts to `output/feature_extractor_comparison/`.
- Updated `docs/feature_extractors/README.md`.
- Validation:
  - `bash -n SLURM/feature_extractor_comparison/run_comparison.slurm SLURM/feature_extractor_comparison/run_array.slurm SLURM/feature_extractor_comparison/analyze_results.slurm SLURM/feature_extractor_comparison/submit_parallel.sh`
  - `ROBOT_SF_MULTI_EXTRACTOR_TEST_MODE=1 uv run python scripts/multi_extractor_training.py --config configs/scenarios/multi_extractor_default.yaml --run-id issue1391-smoke --output-root output/feature_extractor_comparison_smoke`

#1392:

- Added `SLURM/Auxme/camera_ready_benchmark.sl`.
- Supports `CAMERA_READY_BENCHMARK_MODE=preflight|run`, explicit config, label/campaign id, output root, and publication-bundle skip.
- The launcher falls back to direct execution inside the batch allocation for observed Auxme `srun` infrastructure failures:
  - `srun: symbol lookup error: srun: undefined symbol: slurm_msg_set_r_uid`
  - `Unable to confirm allocation ... Zero Bytes were transmitted or received`
- Updated `SLURM/Auxme/README.md` and `docs/dev/slurm_submission.md`.
- Validation:
  - `bash -n SLURM/Auxme/camera_ready_benchmark.sl`
  - `CAMERA_READY_BENCHMARK_CONFIG=configs/benchmarks/sanity_v1_smoke.yaml CAMERA_READY_BENCHMARK_MODE=preflight CAMERA_READY_BENCHMARK_LABEL=sanity-v1-preflight scripts/dev/sbatch_use_max_time.sh --dry-run SLURM/Auxme/camera_ready_benchmark.sl`

#1344:

- Added paired primary-row configs:
  - `configs/benchmarks/amv_paired_nominal_primary_v1.yaml`
  - `configs/benchmarks/amv_paired_stress_primary_v1.yaml`
- Added protocol note:
  - `docs/context/issue_1344_amv_paired_primary_protocol.md`
- Local preflight passed for both configs:
  - `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/amv_paired_nominal_primary_v1.yaml --mode preflight --output-root output/benchmarks/issue_1344_local_preflight --campaign-id nominal_preflight_check --log-level WARNING`
  - `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/amv_paired_stress_primary_v1.yaml --mode preflight --output-root output/benchmarks/issue_1344_local_preflight --campaign-id stress_preflight_check --log-level WARNING`

#1354:

- Reused the existing bounded first-pass config `configs/benchmarks/cross_kinematics_v1.yaml`.
- Local preflight passed:
  - `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/cross_kinematics_v1.yaml --mode preflight --output-root output/benchmarks/issue_1354_local_preflight --campaign-id cross_kinematics_preflight_check --log-level WARNING`
- This is a compatibility/evidence first pass, not the full later paper-facing AMV campaign. #1354 still depends on interpreting #1344 first.

## Submitted Jobs

Initial #1344 submissions:

- `12566`: stress, intended a30, failed in 1 second before benchmark execution because the original launcher used `srun` and hit `srun: symbol lookup error: srun: undefined symbol: slurm_msg_set_r_uid`.
- `12567`: nominal, intended a30, failed in 1 second before benchmark execution because `srun` could not confirm allocation.
- `12568`: cross-kinematics l40s, canceled before start because it had the pre-fallback launcher script captured by `sbatch`.

Current live jobs:

| Job | Issue | Config | Partition | State at handoff | Output root |
| --- | --- | --- | --- | --- | --- |
| `12569` | #1344 nominal | `configs/benchmarks/amv_paired_nominal_primary_v1.yaml` | `a30/a30-gpu` | completed benchmark execution, Slurm `FAILED` exit `2:0` because `benchmark_success=false` | `output/benchmarks/issue_1344/amv_paired_nominal_primary_v1_issue1344-nominal-primary_20260520_154231/` |
| `12570` | #1344 stress | `configs/benchmarks/amv_paired_stress_primary_v1.yaml` | `a30/a30-gpu` | running on `auxme-imech172` | `output/benchmarks/issue_1344/amv_paired_stress_primary_v1_issue1344-stress-primary_20260520_154229/` |
| `12571` | #1354 cross-kinematics | `configs/benchmarks/cross_kinematics_v1.yaml` | `l40s/l40s-gpu` | pending, expected start was `2026-05-21T08:10:02` in the first `squeue` snapshot | `output/benchmarks/issue_1354/` |

Nominal #1344 interpretation snapshot:

- `scripts/tools/analyze_camera_ready_campaign.py` completed for the nominal campaign root.
- Analyzer output:
  - `output/benchmarks/issue_1344/amv_paired_nominal_primary_v1_issue1344-nominal-primary_20260520_154231/reports/campaign_analysis.json`
  - `output/benchmarks/issue_1344/amv_paired_nominal_primary_v1_issue1344-nominal-primary_20260520_154231/reports/campaign_analysis.md`
- Campaign summary reported `total_runs=3`, `successful_runs=1`, `total_episodes=31`, and `benchmark_success=false`.
- Failure reasons:
  - `goal`: partial failure from an episode-integrity contradiction on `classic_doorway_low`, seed `111`.
  - `orca`: partial failure from an episode-integrity contradiction on `classic_doorway_low`, seed `112`.
  - `social_force`: ran without runner failure but had `success_mean=0.0000`.
- This should be interpreted as useful #1344 evidence that the proposed nominal primary surface is not a clean routine-competence gate yet; it is not an infrastructure failure.

Monitor:

```bash
cd /home/luttkule/git/robot_sf_ll7.worktrees/1391-1392-slurm-workflow
squeue -j 12569,12570,12571 --format='%i %j %T %P %Q %y %b %M %l %S %R'
sacct -j 12569,12570,12571 --format=JobID,JobName%28,State,ExitCode,Partition,Elapsed,Start,End -P
tail -n 120 output/slurm/12569-camera-ready-benchmark.out
tail -n 120 output/slurm/12570-camera-ready-benchmark.out
```

After completion, analyze remaining campaign roots:

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1344/amv_paired_stress_primary_v1_issue1344-stress-primary_20260520_154229
```

For #1354, wait for the exact campaign directory under `output/benchmarks/issue_1354/`, then run the same analyzer.

## Deliberately Not Submitted Yet

#1167:

- Issue #1218 is closed and `docs/context/issue_1165_predictive_obstacle_lifecycle.md` records a real map-derived obstacle-feature collector smoke.
- I did not submit the training/eval comparison because the current config-first pipeline is not yet a clean obstacle-feature comparison path:
  - `scripts/training/run_predictive_training_pipeline.py` does not expose/pass `--model-family predictive_obstacle_features_v1` to the base collector.
  - `scripts/training/collect_predictive_hardcase_data.py` still only supports legacy/ego state dimensions and does not append obstacle features.
- A suitable next implementation is to add a tracked #1167 pipeline config plus schema-aware collection for both base and hardcase data, then submit the same-seed comparison.

#1358:

- No training job was submitted because this is not currently a config-only run.
- The issue needs implementation first: ORCA command/risk observation extension, bounded residual policy action adapter, hard guard diagnostics, tests for clipping/diagnostics, and a registered policy-search candidate.
- Submitting a generic PPO or policy-search job now would not answer the issue and would risk producing misleading artifacts.

## Interpretation Notes

- The #1344 runs are primary-row first-pass evidence only: compare nominal competence and stress robustness separately.
- Do not treat nominal success as safety evidence.
- For #1354, unsupported/degraded/fallback rows must remain caveats; the current submitted run is the existing bounded compatibility surface.
- Generated artifacts under `output/` are local and ignored. Promote only compact summaries/manifests under `docs/context/evidence/` after interpreting the jobs.
