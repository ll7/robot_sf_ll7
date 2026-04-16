# Issue 832 Paper-Matrix Extended Seed Schedule

Date: 2026-04-16

Related:
- Upstream issue: `ll7/robot_sf_ll7#832`
- Frozen paper matrix: `docs/context/issue_535_paper_matrix_freeze.md`
- Fixed-scenario seed pilot: `docs/context/issue_751_seed_variability_pilot_execution.md`
- Fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`

## Goal

Extend the paper-facing full benchmark matrix beyond the frozen S3 eval schedule while preserving
the v1 matrix contract: same scenario matrix, planner set, planner grouping, differential-drive
kinematics, SNQI assets, bootstrap settings, and publication/export layout.

This note is benchmark-side evidence guidance only. It does not retroactively replace the
manuscript's frozen S3 reported numbers.

## Selected Seed Policy

The baseline remains:

- S3 / `eval`: `[111, 112, 113]`

The selected staged extension policy is:

- S5 / `paper_eval_s5`: `[111, 112, 113, 114, 115]`
- S10 / `paper_eval_s10`: `[111, 112, 113, 114, 115, 116, 117, 118, 119, 120]`
- S20 / `paper_eval_s20`: `[111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130]`

Rationale:

- Contiguous deterministic seeds avoid hidden seed-selection bias.
- S5 is the first full-matrix extension and keeps runtime manageable on local hardware.
- S10 is the next escalation target if S5 changes planner ranking, scenario winners, or aggregate
  means enough to alter the paper interpretation.
- S20 is a later high-cost reference schedule, not the default first execution.

Versioned surfaces:

- `configs/benchmarks/seed_sets_v1.yaml`
- `configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s5.yaml`
- `configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s10.yaml`
- `scripts/tools/compare_seed_schedule_campaigns.py`

## Decision Criteria

The comparison artifact uses these defaults:

- CI-width reduction target: at least `20%` relative reduction in mean CI width per metric.
- Aggregate mean drift flag: absolute drift greater than `max(0.02, 5% of the S3 mean)`.
- Ranking stability flag: planner-level Kendall tau on `snqi_mean` below `0.8`.
- Scenario-winner flag: more than `10%` of comparable scenarios change winner under scenario-level
  `snqi`.

CI-width target misses are precision caveats. The conservative headline interpretation changes only
when ranking, scenario-winner, or aggregate mean-drift triggers fire.

## Runtime Estimate

Reference S3 run:

- Campaign: `paper_experiment_matrix_v1_release_rehearsal_latest_20260409_20260409_120154`
- Episodes: `987`
- Runtime: `728.7301911249997` seconds
- Throughput: `1.354410743537726` episodes/second

Linear seed-count estimates under the same worker count:

| Schedule | Seeds | Estimated episodes | Estimated runtime |
|---|---:|---:|---:|
| S3 | 3 | 987 | 12.1 min observed |
| S5 | 5 | 1645 | 20.2 min |
| S10 | 10 | 3290 | 40.5 min |
| S20 | 20 | 6580 | 81.0 min |

Local machine context on `LeLuMBP24` says to keep CPU workloads conservative and use capped
concurrency. The paper matrix configs keep `workers: 1`, and long/staged executions should run in
tmux so they can be reattached.

## Canonical Commands

S5 preflight:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s5.yaml \
  --mode preflight \
  --label issue832_s5_preflight
```

S5 run in tmux:

```bash
tmux new -d -s issue832_s5 -- \
  zsh -lc 'cd /Users/lennart/git/robot_sf_ll7 && source .venv/bin/activate && caffeinate -dimsu uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s5.yaml --campaign-id issue832_s5_stage 2>&1 | tee output/benchmarks/camera_ready/issue832_s5_stage.log'
```

If the session stops early, re-run the same tmux command. The fixed `--campaign-id` targets the
same campaign root and the config's `resume: true` skips already-written episode ids.

Analyzer:

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/camera_ready/<s5_campaign_id>
```

Comparison:

```bash
uv run python scripts/tools/compare_seed_schedule_campaigns.py \
  --base-campaign-root output/benchmarks/camera_ready/<s3_campaign_id> \
  --candidate-campaign-root output/benchmarks/camera_ready/<s5_campaign_id> \
  --output-json output/benchmarks/camera_ready/<s5_campaign_id>/reports/seed_schedule_comparison.json \
  --output-md output/benchmarks/camera_ready/<s5_campaign_id>/reports/seed_schedule_comparison.md
```

## Interpretation Guidance

Use the S5 comparison as a full-matrix staged extension, not as a replacement for the frozen S3
publication bundle. If the comparison verdict is `stable`, downstream paper text can keep the S3
numbers while saying a higher-seed full-matrix check did not change the ranking or scenario-level
interpretation under the configured thresholds.

If the verdict is `review`, paper consumers should inspect the flagged planner/metric rows and
changed scenarios before strengthening or narrowing manuscript claims. Do not describe fallback or
degraded planner execution as successful benchmark evidence.

## Validation Log

Code/config validation:

- `uv run ruff check scripts/tools/compare_seed_schedule_campaigns.py tests/tools/test_compare_seed_schedule_campaigns.py tests/benchmark/test_camera_ready_campaign.py`
- `uv run pytest -o addopts='' tests/tools/test_compare_seed_schedule_campaigns.py tests/benchmark/test_camera_ready_campaign.py -k 'seed_schedule or extended_seed'`
- `uv run pytest -o addopts='' tests/tools/test_compare_seed_schedule_campaigns.py tests/tools/test_run_camera_ready_benchmark.py tests/benchmark/test_camera_ready_campaign.py`
- `uv run ruff check scripts/tools/run_camera_ready_benchmark.py robot_sf/benchmark/camera_ready_campaign.py tests/tools/test_run_camera_ready_benchmark.py tests/benchmark/test_camera_ready_campaign.py`
- `uv run pytest -o addopts='' tests/tools/test_run_camera_ready_benchmark.py tests/benchmark/test_camera_ready_campaign.py -k 'campaign_id or run_mode or extended_seed'`
- `uv run pytest -o addopts='' tests/benchmark/test_camera_ready_campaign.py -k 'planner_report_row_counts_existing_records_after_resume or campaign_id or extended_seed'`
- `git diff --check`

Preflight:

- S5:
  `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s5.yaml --mode preflight --label issue832_s5_preflight_committed`
- S10:
  `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s10.yaml --mode preflight --label issue832_s10_preflight_committed`

Executed campaigns:

- S3 reference root:
  `output/benchmarks/camera_ready/issue832_s3_reference_bd60bae`
  - episode execution commit: `bd60bae4401d075fc67436b42da6f5f4deb95aa7`
  - final report-generation commit: `b993fd11cbeb91c44f7329b0f853ab76cc27a488`
  - episodes: `987`
  - runs: `7/7` successful
  - SNQI contract: `pass`
  - tmux session: `issue832_s3_bd60bae`
  - execution wall clock: about `10.7` min (`11:52:14` to `12:02:56`)
- S5 extended root:
  `output/benchmarks/camera_ready/issue832_s5_stage_bd60bae`
  - episode execution commit: `bd60bae4401d075fc67436b42da6f5f4deb95aa7`
  - final report-generation commit: `b993fd11cbeb91c44f7329b0f853ab76cc27a488`
  - episodes: `1645`
  - runs: `7/7` successful
  - SNQI contract: `warn` with `snqi_contract.enforcement=warn`
  - tmux sessions: `issue832_s5_bd60bae`, then `issue832_s5_bd60bae_resume1`
  - execution wall clock including manual restart gap: about `22.9` min (`11:52:14` to `12:15:12`)
  - active execution time estimate from logs excluding the restart gap: about `18.1` min

The final reports were regenerated after adding fixed campaign ids and correcting resumed planner
episode counts. The regeneration pass used `resume: true` and did not re-execute completed episode
ids; this is why `campaign.runtime_sec` in the final JSON files reflects the short report
regeneration invocation rather than total benchmark wall time.

Analysis/export artifacts:

- S3:
  - `output/benchmarks/camera_ready/issue832_s3_reference_bd60bae/reports/campaign_analysis.json`
  - `output/benchmarks/camera_ready/issue832_s3_reference_bd60bae/reports/campaign_analysis.md`
  - `output/benchmarks/camera_ready/issue832_s3_reference_bd60bae/reports/scenario_difficulty_analysis.json`
  - `output/benchmarks/camera_ready/issue832_s3_reference_bd60bae/reports/scenario_difficulty_analysis.md`
- S5:
  - `output/benchmarks/camera_ready/issue832_s5_stage_bd60bae/reports/campaign_analysis.json`
  - `output/benchmarks/camera_ready/issue832_s5_stage_bd60bae/reports/campaign_analysis.md`
  - `output/benchmarks/camera_ready/issue832_s5_stage_bd60bae/reports/scenario_difficulty_analysis.json`
  - `output/benchmarks/camera_ready/issue832_s5_stage_bd60bae/reports/scenario_difficulty_analysis.md`
  - `output/benchmarks/camera_ready/issue832_s5_stage_bd60bae/reports/seed_schedule_comparison.json`
  - `output/benchmarks/camera_ready/issue832_s5_stage_bd60bae/reports/seed_schedule_comparison.md`

Comparison result:

- Verdict: `review`
- Ranking stability: stable (`Kendall tau = 1.0`, `Spearman rho = 1.0`)
- Scenario winner changes: `7 / 47` scenarios (`14.9%`), above the `10%` threshold
- Aggregate mean drift flags: `9 / 35` planner-metric rows
- CI-width target misses: `success`, `near_misses`, `time_to_goal_norm`, `snqi`

Interpretation:

- The S5 extension does not change the aggregate SNQI planner ordering.
- The S5 extension does change enough scenario-level winners and aggregate means that downstream
  paper text should avoid strengthening seed-stability claims from the frozen S3 bundle.
- The current paper-facing S3 numbers can remain a bounded initial full-matrix protocol, but this
  benchmark-side follow-up should be cited as a caution that broader seeding preserves ranking while
  exposing scenario-level and mean-drift sensitivity.
