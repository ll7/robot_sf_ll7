# Camera-Ready Benchmark Campaign Workflow

This document describes the config-driven campaign workflow for generating
camera-ready benchmark outputs across multiple planners.

Canonical benchmark fallback policy:

* [Issue #691 Benchmark Fallback Policy](./context/issue_691_benchmark_fallback_policy.md)
* [Experimental Planner Guardrails](./benchmark_experimental_planners.md)
* [Planner-Family Coverage Matrix](./benchmark_planner_family_coverage.md)
* [Francis Guideline Mapping For Robot SF](./context/issue_759_francis_guideline_mapping.md)

## Entry Point

Run the campaign CLI:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/camera_ready_all_planners.yaml
```

Preflight-only (validate + preview + matrix summary, no episode execution):

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1.yaml \
  --mode preflight \
  --label preflight
```

Optional:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/camera_ready_all_planners.yaml \
  --label draft \
  --log-level INFO
```

For long/full campaigns, force worker log noise down:

```bash
LOGURU_LEVEL=INFO uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/camera_ready_all_planners.yaml \
  --label full_run
```

## Current Baseline (2026-02-20)

Current promoted all-planners baseline run:

* campaign id:
  + `camera_ready_all_planners_prediction_first_prediction_first_stop_verify_20260220_201848`
* root:
  + `output/benchmarks/camera_ready/camera_ready_all_planners_prediction_first_prediction_first_stop_verify_20260220_201848`
* status:
  + `8/8 planners ok`,   `1080 episodes`, runtime `386.54s`
* predictive planner:
  + `status=ok`,   `episodes=135`,   `failed_jobs=0`
  + `success_mean=0.9778`,   `collisions_mean=0.0000`,   `near_misses_mean=0.0296`

## Config Presets

* `configs/benchmarks/camera_ready_smoke_all_planners.yaml`
  + single scenario smoke for fast validation
* `configs/benchmarks/camera_ready_baseline_safe.yaml`
  + baseline-ready planners on full scenario suite
* `configs/benchmarks/camera_ready_all_planners.yaml`
  + baseline + experimental planners on full scenario suite
  + prediction planner runs first for early fail-fast signal
  + `stop_on_failure: true` (aborts on `failed` and `partial-failure`)
* `configs/benchmarks/paper_experiment_matrix_all_planners_v1.yaml`
  + all-planners paper-facing contract for release rehearsal and analysis
  + `stop_on_failure: false` so campaign execution continues and failed planners are surfaced
    explicitly with reason fields in artifacts

* `configs/benchmarks/camera_ready_all_planners_holonomic.yaml`
  + co-existing holonomic sibling profile for issue 690 feasibility work
  + keeps the same scenario matrix, seed policy, publication bundle, and report layout
  + strict fail-closed planner policy: no fallback-to-success behavior in benchmark mode
* `configs/benchmarks/camera_ready_all_planners_strict_socnav.yaml`
  + full suite with strict SocNav prereq policy (`fail-fast`, no fallback)
* `configs/benchmarks/paper_experiment_matrix_v1.yaml`
  + frozen paper-facing execution contract (`paper_profile_version=paper-matrix-v1`)
  + mixed planner matrix with explicit `planner_group` tags (`core|experimental`)
  + differential-drive-only kinematics for v1 paper freeze
  + `paper_interpretation_profile=baseline-ready-core` means the matrix is paper-facing and
    anchored to the core baseline set, while still allowing experimental challenger rows for
    comparison
* `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml`
  + current long-horizon benchmark collection that replaces the fixed campaign `horizon` with
    `scenario_horizons: configs/policy_search/scenario_horizons_h500.yaml`
  + patches each scenario's `simulation_config.max_episode_steps` before the map runner executes
    and records `metadata.scenario_horizon` for provenance and `planner_blocked` accounting
  + includes the baseline-ready core rows plus experimental challengers, including the h500
    policy-search candidates `scenario_adaptive_hybrid_orca_v1` and
    `hybrid_rule_v3_fast_progress_static_escape`
  + disables publication bundle export because the candidate-augmented local full run completed but
    has SNQI contract status `fail` under warn enforcement
* `configs/benchmarks/sanity_v1_smoke.yaml`
  + non-paper-facing nominal calibration smoke for issue #1083
  + runs `configs/scenarios/sanity_v1.yaml` with fixed seed `111`, differential-drive kinematics,
    and the baseline-safe `goal` plus `orca` planners
  + use it to confirm baseline competence on easier deployment-like scenes before interpreting
    hard-matrix failures; it is not a replacement for the paper or h500 stress surfaces
* `configs/benchmarks/paper_cross_kinematics_v1.yaml`
  + issue #1082 cross-kinematics parity smoke surface
  + uses `paper_profile_version=paper-cross-kinematics-v1` and exactly
    `differential_drive`, `bicycle_drive`, and `holonomic`
  + runs only the first supported core rows (`goal`, `social_force`, `orca`) on one
    `classic_cross_trap_low` seed, with deferred planner coverage recorded in
    `configs/benchmarks/paper_cross_kinematics_v1_compatibility.yaml`
* `configs/benchmarks/cross_kinematics_v1.yaml`
  + issue #1274 non-paper cross-kinematics parity smoke surface
  + uses the same three kinematics modes and core planner rows as the paper profile but keeps
    `paper_facing: false`
  + records supported, degraded, and unsupported rows in
    `configs/benchmarks/cross_kinematics_v1_compatibility.yaml`
* `configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s5.yaml`
  + stage-1 paper-matrix seed extension using `paper_eval_s5=[111..115]`
  + preserves the v1 scenario matrix, planner grouping, differential-drive kinematics, SNQI assets,
    and publication/export contract; only the named seed set changes
* `configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s10.yaml`
  + escalation target using `paper_eval_s10=[111..120]`
  + use when the S5 comparison still shows material interval width, ranking, or scenario-winner
    instability
* `configs/benchmarks/socnav_bench_reentry_probe.yaml`
  + focused `socnav_bench` re-entry probe on one Francis blind-corner scenario and seeds
    `[111, 112, 113]`
  + includes only `goal` and `socnav_bench` so runtime and outcome comparison stays cheap and
    attributable
  + keeps `socnav_bench` fail-fast; fallback/degraded execution is not valid re-entry evidence

* `configs/algos/prediction_planner_camera_ready.yaml`
  + explicit `prediction_planner` camera-ready profile used by all-planners presets
  + resolves checkpoint via `predictive_model_id` from `model/registry.yaml`

`prediction_planner` is now part of all-planners campaign presets as an experimental planner.
For reproducible runs, verify that the configured model id exists and resolves to a valid local checkpoint before launching the campaign.

### Forecast Transferability Stress Matrix

Use the diagnostic transferability matrix builder to summarize existing `ForecastMetrics.v1`
reports across explicit observation tier, noise, latency, dropout, occlusion, map family, density,
pedestrian-model family, and actor-type dimensions:

```bash
uv run python scripts/benchmark/build_forecast_transferability_matrix.py \
  path/to/forecast_metrics.json \
  --report-id prediction_transferability_diagnostic \
  --out-json output/forecast_transferability_matrix.json \
  --out-md output/forecast_transferability_matrix.md
```

The builder does not run a benchmark campaign by itself. Missing transfer dimensions are emitted as
limitation rows, oracle-only rows remain diagnostic-only, and the recommendation field is the claim
boundary for continue/revise/stop decisions.

Use the calibration report builder to summarize existing `ForecastMetrics.v1` reliability rows by
scenario family, horizon, observation tier, and predictor family:

```bash
uv run python scripts/benchmark/build_forecast_calibration_report.py \
  path/to/forecast_metrics.json \
  --report-id prediction_calibration_diagnostic \
  --out-json output/forecast_calibration_report.json \
  --out-md output/forecast_calibration_report.md
```

Deterministic or uncertainty-incomplete rows are reported as calibration-unavailable limitations.
The recommendation field is analysis-only guidance for continue/revise/wait decisions before any
planner-risk coupling.

Use the conformal pilot builder to fit split-conformal deterministic forecast tubes on calibration
`ForecastBatch.v1` artifacts and evaluate coverage on separate held-out forecast artifacts:

```bash
uv run python scripts/benchmark/build_forecast_conformal_pilot.py \
  --calibration-batch path/to/calibration_forecast_batch.json \
  --calibration-ground-truth path/to/calibration_ground_truth.json \
  --evaluation-batch path/to/heldout_forecast_batch.json \
  --evaluation-ground-truth path/to/heldout_ground_truth.json \
  --report-id prediction_conformal_smoke \
  --out-json output/forecast_conformal_pilot.json \
  --out-md output/forecast_conformal_pilot.md
```

The pilot records split provenance, empirical held-out coverage, radius/set-size diagnostics, and
missing-denominator limitations. Its recommendation is smoke-only guidance and must not be treated
as planner safety or real-world coverage evidence.

Testing-only planners remain opt-in only under the guardrail policy in
`docs/benchmark_experimental_planners.md` . Issue 596 adds a verified-simple stage gate for any
future reconsideration, but does not remove the opt-in requirement by itself.

`socnav_bench` remains outside the canonical paper matrix. Re-entry requires the issue-562 focused
probe to pass without fallback, complete all three probe episodes, show non-zero success, and keep
runtime within `3x` the `goal` row on the same slice before any broader paper-compatible subset is
run. See `docs/context/issue_562_socnav_bench_reentry.md`.

### Paper-Matrix Planner Readiness

The frozen `paper_experiment_matrix_v1` profile has seven planner rows. Interpret them with the
fallback policy in `docs/context/issue_691_benchmark_fallback_policy.md`: `native` and declared
`adapter` rows can be benchmark evidence when the run reports `availability_status=available`;
`fallback`, `degraded`, `failed`, `partial-failure`, and `not_available` rows are exclusions or
caveats, not successful planner results.

| Planner | Group | Expected mode | Readiness interpretation |
| --- | --- | --- | --- |
| `goal` | core | native | Baseline-ready control row. |
| `social_force` | core | adapter | Baseline-ready force-based comparator through the declared adapter path. |
| `orca` | core | adapter | Baseline-ready reciprocal-avoidance comparator when `rvo2` is available. |
| `ppo` | learned baseline | native | Paper-facing PPO row when model provenance and benchmark-set claim caveats are satisfied. |
| `prediction_planner` | experimental | adapter | Checkpoint-dependent experimental challenger row. |
| `socnav_sampling` | experimental | adapter | In-repo sampling challenger; not upstream SocNavBench support and not a SocNavBench bridge result. |
| `sacadrl` | experimental | adapter | Legacy adapter-sensitive challenger row; implementation evidence only. |

`socnav_bench` is not part of this frozen paper matrix. The May 4 all-planners run recorded
`socnav_bench` as `execution_mode=unknown`, `readiness_status=degraded`, and
`availability_status=failed` because required SocNavBench assets were absent; that row must remain
excluded until a focused fail-fast re-entry probe succeeds.

### Scenario-Horizon Change

The long-horizon collection uses the h500-derived scenario schedule instead of a single fixed
`horizon: 100`. The schedule in `configs/policy_search/scenario_horizons_h500.yaml` was derived
from safe h500 policy-search incumbents using the documented selection parameters:
`p95_multiplier=1.2`, `buffer_steps=20`, `floor_steps=80`, and `cap_steps=600`. Three scenarios
remain marked `planner_blocked` and keep the h600 cap rather than being silently excluded.

This change exists because the fixed 100-step budget mixed route-completion quality with route
length and scenario geometry. Longer or bottlenecked routes could appear as planner failures even
when the planner was still making valid progress. The scenario-specific schedule reduces that
budget artifact, but it also changes the safety exposure: longer episodes give planners more time
to collide or enter near-miss states. Treat success gains under the long horizons as a
route-budget sensitivity result unless the updated campaign also preserves collision, near-miss,
SNQI, fallback, and runtime evidence.

The 2026-05-06 candidate-augmented local full run completed 9 planner rows and 1296 episodes, with
no failed, unavailable, fallback, or degraded planners. It is still not release-tag evidence:
SNQI contract status is `fail`, and the two added candidates remain experimental challenger rows.

SNQI calibration assets used by camera-ready presets:

* `configs/benchmarks/snqi_weights_camera_ready_v1.json`
* `configs/benchmarks/snqi_baseline_camera_ready_v1.json`
* `configs/benchmarks/snqi_weights_camera_ready_v2.json`
* `configs/benchmarks/snqi_baseline_camera_ready_v2.json`
* `configs/benchmarks/snqi_weights_camera_ready_v3.json` (current paper-facing default)
* `configs/benchmarks/snqi_baseline_camera_ready_v3.json` (current paper-facing default)

## Produced Artifacts

Campaign outputs are written under:

 `output/benchmarks/camera_ready/<campaign_id>/`

Expected tree:

```text
<campaign_id>/
  campaign_manifest.json
  manifest.json
  run_meta.json
  preflight/
    validate_config.json
    preview_scenarios.json
  runs/
    <planner_key>/
      episodes.jsonl
      summary.json
  reports/
    matrix_summary.json
    matrix_summary.csv
    amv_coverage_summary.json
    amv_coverage_summary.md
    comparability_matrix.json
    comparability_matrix.md
    seed_variability_by_scenario.json
    seed_variability_by_scenario.csv
    seed_episode_rows.csv
    statistical_sufficiency.json
    seed_schedule_comparison.json       # optional comparison-script output
    seed_schedule_comparison.md         # optional comparison-script output
    snqi_diagnostics.json
    snqi_diagnostics.md
    snqi_sensitivity.csv
    campaign_summary.json
    campaign_table.csv
    campaign_table.md
    campaign_table_core.csv
    campaign_table_core.md
    campaign_table_experimental.csv
    campaign_table_experimental.md
    campaign_report.md
```

  Optional post-run analyzer outputs after calling
`scripts/tools/analyze_camera_ready_campaign.py` :

  + `reports/campaign_analysis.json`
  + `reports/campaign_analysis.md`
  + `reports/scenario_difficulty_analysis.json`
  + `reports/scenario_difficulty_analysis.md`

Publication bundle export is written under:

 `output/benchmarks/publication/`

with files generated by `export_publication_bundle` .

Release publication runbook:

* `docs/benchmark_camera_ready_release.md`
* `docs/benchmark_artifact_publication.md`
* `docs/benchmark_release_protocol.md`
* `docs/benchmark_release_reproducibility.md`
* `docs/context/issue_2040_artifact_publication_workflow.md`

Benchmark release entrypoint:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml
```

## Post-Run Difficulty Analysis

Use the analyzer after a full campaign when you need to distinguish scenario-driven difficulty from
planner-specific mismatch:

```bash
source .venv/bin/activate
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id>
```

The analyzer keeps benchmark semantics unchanged and only consumes existing artifacts. It adds:

* planner integrity and runtime hotspot checks in `reports/campaign_analysis.{json,md}`
* scenario difficulty ranking, family rollups, planner residual summaries, and verified-simple
  subset assessment in `reports/scenario_difficulty_analysis.{json,md}`

Interpretation rules:

* Treat the difficulty score as a reproducible proxy, not as ground truth.
* Use residuals to distinguish globally hard scenarios from planner-specific mismatch.
* Treat the verified-simple subset as a calibration aid only when a bounded pilot preserves planner
  ordering and does not inflate seed noise materially.

For the pilot method and current recommendation gate, see
`docs/context/issue_692_scenario_difficulty_analysis.md` .

## Campaign Summary Semantics

Benchmark mode is fail-closed:

* fallback-only or skipped planners are reported as `not_available`
* partial-failure planners are reported as non-success
* campaign CLI exit status is non-zero when any planner row is not benchmark-success
* diagnostic fallback remains valid only for explicit probe workflows, not for benchmark claims

`reports/campaign_summary.json` contains:

* campaign metadata/provenance (scenario hash, git hash, runtime)
* per-planner run summary from benchmark runner
* per-planner aggregate statistics (mean and CI when available)
* flattened planner comparison rows
* matrix definition summary rows (`reports/matrix_summary.{json,csv}`)
* AMV scope coverage summary (`reports/amv_coverage_summary.{json,md}`)
* Alyassi comparability summary (`reports/comparability_matrix.{json,md}`)
* seed variability by scenario/planner (`reports/seed_variability_by_scenario.{json,csv}`)
* planner-aware per-episode seed rows (`reports/seed_episode_rows.csv`)
* statistical sufficiency summary for seed variability (`reports/statistical_sufficiency.json`)
* SNQI contract diagnostics (`reports/snqi_diagnostics.{json,md}` + `reports/snqi_sensitivity.csv`)
  + includes contract health, planner ordering, component correlations, and ablation-based weight sensitivity
* warning list
* explicit per-planner failure reason field (`most_likely_failure_reason`) in planner rows/tables
* publication bundle paths (if export enabled)
* interpretation profile metadata (`paper_interpretation_profile`)

## Reproducibility Metadata

Each campaign now stores the exact invocation and timing provenance.

Captured fields include:

* exact command used to launch the campaign (`invoked_command`)
* campaign wallclock start/end (`started_at_utc`,   `finished_at_utc`)
* campaign runtime and throughput (`runtime_sec`,   `episodes_per_second`)
* per-planner start/end/runtime/throughput in run entries and planner summaries
* seed-policy provenance (`mode`, configured seeds, resolved seed list)
* preflight artifact paths (`validate_config`,   `preview_scenarios`)

Primary locations:

* `output/benchmarks/camera_ready/<campaign_id>/reports/campaign_summary.json`
  + `campaign.invoked_command`
  + `campaign.started_at_utc`
  + `campaign.finished_at_utc`
  + `campaign.runtime_sec`
  + `campaign.episodes_per_second`
  + `runs[].started_at_utc`
  + `runs[].finished_at_utc`
  + `runs[].runtime_sec`
  + `runs[].summary.episodes_per_second`
* `output/benchmarks/camera_ready/<campaign_id>/campaign_manifest.json`
  + `invoked_command`
  + `started_at_utc`
  + `runtime_sec`
* `output/benchmarks/camera_ready/<campaign_id>/run_meta.json`
  + `invoked_command`
  + `started_at_utc`
  + `finished_at_utc`
  + `runtime_sec`
  + `episodes_per_second`
  + `seed_policy.*`
  + `preflight_artifacts.*`
* `output/benchmarks/camera_ready/<campaign_id>/preflight/validate_config.json`
* `output/benchmarks/camera_ready/<campaign_id>/preflight/preview_scenarios.json`
* `output/benchmarks/camera_ready/<campaign_id>/reports/matrix_summary.json`
* `output/benchmarks/camera_ready/<campaign_id>/reports/matrix_summary.csv`
* `output/benchmarks/camera_ready/<campaign_id>/reports/seed_variability_by_scenario.json`
* `output/benchmarks/camera_ready/<campaign_id>/reports/seed_variability_by_scenario.csv`
* `output/benchmarks/camera_ready/<campaign_id>/reports/seed_episode_rows.csv`
* `output/benchmarks/camera_ready/<campaign_id>/reports/statistical_sufficiency.json`
* `output/benchmarks/camera_ready/<campaign_id>/reports/campaign_report.md`
  + command in header
  + per-planner timing columns in the summary table

## Fixed-Scenario Multi-Seed Variability

Use the camera-ready campaign stack as the canonical upstream contract for paper-side seed
variability analysis.

Required config shape:

* one fixed scenario manifest
* one explicit planner set
* explicit `seed_policy`
* bootstrap settings recorded in config

Recommended seed policy for paper-facing variability work:

```yaml
seed_policy:
  mode: fixed-list
  seeds: [111, 112, 113, 114, 115, 116, 117, 118]
bootstrap_samples: 1000
bootstrap_confidence: 0.95
bootstrap_seed: 123
```

For full paper-matrix seed-count extensions, use the named seed sets in
`configs/benchmarks/seed_sets_v1.yaml`:

* `eval` / S3: `[111,112,113]`, the frozen paper-facing baseline.
* `paper_eval_s5` / S5: `[111,112,113,114,115]`, the first staged full-matrix extension.
* `paper_eval_s10` / S10: `[111..120]`, the escalation target when S5 is still unstable.
* `paper_eval_s20` / S20: `[111..130]`, a high-cost reference schedule for later variance
  power checks.

Interpret extended-seed comparisons with the following default decision criteria:

* CI-width reduction target: at least 20% relative reduction in mean CI width per metric, while
  accepting unchanged zero-width intervals as already tight.
* Aggregate mean drift: flag planner/metric rows whose absolute drift exceeds
  `max(0.02, 5% of the S3 mean)`.
* Ranking stability: flag if Kendall tau on planner-level `snqi_mean` falls below `0.8`.
* Scenario-winner stability: flag if more than 10% of comparable scenarios change winner under
  scenario-level `snqi`.

Staged stopping rule:

* Run S5 first under the same matrix, planner, kinematics, SNQI, bootstrap, and export settings as
  S3.
* Stop at S5 when planner ranking is stable, scenario-winner changes are within threshold, and no
  aggregate drift trigger changes the interpretation. CI-width target misses are reported as
  precision caveats, not automatic headline changes.
* Escalate to S10 when S5 crosses the ranking, scenario-winner, or aggregate-drift thresholds.
* Reserve S20 for a dedicated high-cost reference run when S10 still leaves manuscript-relevant
  uncertainty.

Canonical S5 preflight:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s5.yaml \
  --mode preflight \
  --label issue832_s5_preflight
```

Canonical resumable S5 run in tmux:

```bash
tmux new -d -s issue832_s5 -- \
  zsh -lc 'cd /path/to/robot_sf_ll7 && source .venv/bin/activate && caffeinate -dimsu uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s5.yaml --campaign-id issue832_s5_stage 2>&1 | tee output/benchmarks/camera_ready/issue832_s5_stage.log'
```

Re-run the same command to resume an interrupted root; `--campaign-id` keeps the output directory
stable and `resume: true` skips completed episode ids.

Grouping semantics:

* raw per-planner `runs/<planner>/episodes.jsonl` remain the execution records
* `reports/seed_episode_rows.csv` is the paper-facing flat export for grouping by
`scenario_id` , `planner_key` , `seed` , and deterministic `repeat_index`

* `reports/seed_variability_by_scenario.json` is the aggregate export grouped by
`(scenario_id, planner_key)` across seeds

* `reports/seed_episode_rows.csv` is a per-episode traceability export retained for legacy
  compatibility with manuscript-side table tooling. In this release artifact family, the
  `collision` column is the legacy per-episode **metric alias** sourced from
  `metrics.collisions`; canonical event-level collision state remains
  `outcome.collision_event` in the source `runs/<planner>/episodes.jsonl`, and the two
  can diverge only if a planner/metric compatibility edge case is present in historical
  releases.

Confidence semantics:

* the seed-variability export uses bootstrap over per-seed means
* the JSON payload records:
  + method
  + confidence level
  + bootstrap sample count
  + bootstrap seed
* each metric summary records:
  + `mean`
  + `std`
  + `cv`
  + `count`
  + `ci_low`
  + `ci_high`
  + `ci_half_width`

Artifact bundle expected by paper consumers:

* `campaign_manifest.json`
* `run_meta.json`
* `reports/seed_variability_by_scenario.json`
* `reports/seed_variability_by_scenario.csv`
* `reports/seed_episode_rows.csv`
* `reports/statistical_sufficiency.json`

Recommended pilot for downstream manuscript issue `amv_benchmark_paper#74` :

* benchmark config: `configs/benchmarks/paper_seed_variability_pilot_v1.yaml`
* scenarios:
  + `classic_cross_trap_low`
  + `classic_head_on_corridor_low`
  + `classic_overtaking_low`
  + `classic_t_intersection_low`
* planners:
  + `orca`
  + `ppo`

Quick inspection example:

```bash
jq '.campaign | {invoked_command, started_at_utc, finished_at_utc, runtime_sec, episodes_per_second}' \
  output/benchmarks/camera_ready/<campaign_id>/reports/campaign_summary.json
```

Inspect frozen matrix summary:

```bash
jq '.rows[0] | {planner_key, planner_group, kinematics, repeats, paper_profile_version}' \
  output/benchmarks/camera_ready/<campaign_id>/reports/matrix_summary.json
```

Analyzer helper:

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id>
```

This emits:

* `reports/campaign_analysis.json`
* `reports/campaign_analysis.md`
* `reports/scenario_difficulty_analysis.json`
* `reports/scenario_difficulty_analysis.md`

SNQI contract analyzer helper:

```bash
# For v3 paper-facing campaigns only.  For other campaigns, replace --weights and
# --baseline with the asset paths recorded in that campaign's manifest.json.
uv run python scripts/tools/analyze_snqi_contract.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --weights configs/benchmarks/snqi_weights_camera_ready_v3.json \
  --baseline configs/benchmarks/snqi_baseline_camera_ready_v3.json
```

This emits:

* `reports/snqi_diagnostics.json`
* `reports/snqi_diagnostics.md`
* `reports/snqi_sensitivity.csv`

The diagnostics now include:

* a positioning recommendation for SNQI under the current contract
* planner ordering by mean SNQI
* per-component Spearman correlations against episode-level SNQI
* per-weight ablation sensitivity rows in `snqi_sensitivity.csv`

The analyzer now also emits runtime hotspot diagnostics:

* top slow planners by campaign runtime
* per-planner `wall_time_sec` mean/p95
* top slow scenarios per hotspot planner

Campaign-to-campaign comparison helper:

```bash
uv run python scripts/tools/compare_camera_ready_campaigns.py \
  --base-campaign-root output/benchmarks/camera_ready/<base_campaign_id> \
  --candidate-campaign-root output/benchmarks/camera_ready/<candidate_campaign_id> \
  --output-json output/benchmarks/camera_ready/<candidate_campaign_id>/reports/campaign_comparison.json \
  --output-md output/benchmarks/camera_ready/<candidate_campaign_id>/reports/campaign_comparison.md
```

Use this to validate quality changes (for example predictive planner success/collision deltas)
after compatibility or config fixes. When `scenario_breakdown.csv` and
`scenario_family_breakdown.csv` are present in both campaigns, the JSON comparison also includes
complete scenario-level and scenario-family deltas. The helper reports `unfinished_mean` as
`1 - success_mean`; treat that as a route-incomplete comparison metric, not raw timeout
attribution.

Seed-schedule comparison helper:

```bash
uv run python scripts/tools/compare_seed_schedule_campaigns.py \
  --base-campaign-root output/benchmarks/camera_ready/<s3_campaign_id> \
  --candidate-campaign-root output/benchmarks/camera_ready/<s5_or_s10_campaign_id> \
  --output-json output/benchmarks/camera_ready/<s5_or_s10_campaign_id>/reports/seed_schedule_comparison.json \
  --output-md output/benchmarks/camera_ready/<s5_or_s10_campaign_id>/reports/seed_schedule_comparison.md
```

Use the seed-schedule comparison for issue-832-style stability checks. It consumes the versioned
campaign outputs and reports CI-width changes, aggregate mean drift, planner ranking correlation,
scenario-level winner changes, and a conservative `stable` / `review` interpretation.

Analyzer findings now also include portability checks, including detection of
absolute `scenario_params.map_file` paths in episodes (these should be
repository-relative for publication-grade portability).

## Camera-Ready Table Fields

`campaign_table.csv` and `campaign_table.md` include at least:

* planner key and algorithm
* execution mode and readiness status (`native` / `adapter` / `fallback` / `degraded`)
* readiness tier and preflight status
* episode count and failure count
* success/collision/near-miss means
* time-to-goal normalization mean
* path efficiency mean
* comfort exposure mean
* jerk mean
* SNQI mean and CI fields (if available)

Core vs experimental partitions:

* paper-facing profile (`paper_profile_version=paper-matrix-v1`):
  partitioning follows explicit planner tags from config ( `planner_group=core|experimental` )
  as part of the frozen execution contract.
* non-paper runs:
  partitioning remains readiness-tier based. When the scenario-difficulty analyzer
  cannot find an eligible core benchmark-success set, the report falls back to the
  full scenario breakdown and says so explicitly instead of implying a core consensus.
* `campaign_table_core.{csv,md}`:
  core partition rows ( `planner_group=core` for paper-facing runs;
`readiness_tier=baseline-ready` for non-paper runs).
* `campaign_table_experimental.{csv,md}`:
  non-core rows ( `planner_group!=core` for paper-facing runs; non-baseline-ready otherwise).

Interpretation note:

- `campaign_table_core` is an implementation-maturity or policy-family slice, not
  the manuscript main comparison set. For `orca`/`ppo`-style headline comparisons in release
  or dissertation workflows, read from `campaign_table.csv` (or explicitly documented
  alternative subset).
- Release `0.0.2` used a scoped planner set where `ppo` was outside the core partition, so
  its secondary-core table is expected not to contain the ppo row.

Portability guarantee:

* Episode `scenario_params.map_file` is normalized to repository-relative paths
  when the map resides in the repository tree (for example
`maps/svg_maps/classic_crossing.svg` ).

Additional diagnostics generated per campaign:

* `reports/scenario_breakdown.csv` and `reports/scenario_breakdown.md`
  + per-planner, per-scenario metric means
  + AMV taxonomy columns (`use_case`, `context`, `speed_regime`, `maneuver_type`)
    carry the direct source scenario metadata when present
* `reports/scenario_family_breakdown.csv` and `reports/scenario_family_breakdown.md`
  + per-planner, per-family (archetype) metric means
  + AMV taxonomy columns aggregate the distinct non-empty values from contributing
    scenarios per dimension, sorted and joined with semicolons

Empty AMV taxonomy cells mean the source scenario, or all contributing scenarios
for a family, had no AMV metadata for that dimension. They are descriptors only:
they do not encode planner success,
failure, fallback, degraded execution, or availability. Continue to interpret
benchmark evidence through `availability_status`, `benchmark_success`, and the
fail-closed fallback policy above.

Canonical table exporter:

```bash
export CAMPAIGN_ROOT=output/benchmarks/camera_ready/<campaign_id>

uv run python - <<'PY'
import json
import os
from pathlib import Path

reports = Path(os.environ["CAMPAIGN_ROOT"]) / "reports"
summary = json.loads((reports / "campaign_summary.json").read_text(encoding="utf-8"))
rows = summary["planner_rows"]
(reports / "planner_rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
PY

uv run robot_sf_bench export-canonical-table \
  --table-id planner_outcome_summary \
  --rows "$CAMPAIGN_ROOT/reports/planner_rows.json" \
  --out-dir "$CAMPAIGN_ROOT/reports/canonical_tables" \
  --source "$CAMPAIGN_ROOT/reports/campaign_summary.json"
```

The exporter writes `csv`, `md`, and `tex` fragments plus
`<table-id>.metadata.json` with source checksums, command, commit, row count, selected columns, and
generated output paths. Named table contracts currently cover planner outcome, scenario-family,
seed-variability, execution-mode, and artifact-source summaries. Fallback, degraded, failed, and
not-available rows remain explicit table rows; the exporter only formats rows and records
provenance, and does not recompute benchmark metrics or reinterpret row status.

## Notes on Experimental Planners

Experimental planners are executed with explicit profile and prereq policy from
the campaign config. For dependency-sensitive planners (for example SocNav
adapters), set `socnav_missing_prereq_policy: fallback` when you want campaign
execution to continue with degraded behavior instead of hard-fail.

Current planner mapping in map-runner:

* `socnav_sampling`: in-repo `SamplingPlannerAdapter` baseline (no upstream SocNavBench dependency).
* `socnav_bench`: upstream `SocNavBenchSamplingAdapter` wrapper (requires SocNav prereqs).

Decision rule for publication:

* Use strict profile/config when you need publication claims without degraded
  fallback behavior.
* Use fallback profile/config for diagnostics/exploration only, and always cite
`preflight_status` plus the report disclosure section
`SocNav Strict-vs-Fallback Disclosure` .

## Current Validation Snapshot

Validated on branch `codex/benchmark-camera-ready-pipeline` :

* baseline-safe calibration campaign (multi-seed `eval` set):
  + `camera_ready_baseline_safe_snqi_calib_base_20260217_122711`
  + `total_runs=3`,   `successful_runs=3`,   `total_episodes=405`
  + output used to derive:
    - `configs/benchmarks/snqi_baseline_camera_ready_v1.json`
* baseline-safe verification with SNQI enabled:
  + `camera_ready_baseline_safe_snqi_verify_20260217_123159`
  + `total_runs=3`,   `successful_runs=3`,   `total_episodes=405`
  + `snqi_mean` is numeric in campaign table (no `nan`)
* smoke all-planners with SNQI calibration enabled:
  + `camera_ready_smoke_all_planners_snqi_check_20260217_123000`
  + `total_runs=7`,   `successful_runs=7`,   `total_episodes=7`
  + `snqi_mean` is numeric in campaign table (no `nan`)
* full all-planners with multi-seed + SNQI enabled:
  + `camera_ready_all_planners_snqi_multiseed_verify_20260217_123437`
  + `total_runs=7`,   `successful_runs=7`,   `total_episodes=945`
  + `snqi_mean` is numeric for all planners in campaign table
* smoke all-planners:
  + `camera_ready_smoke_all_planners_smoke3_20260217_112307`
  + `total_runs=7`,   `successful_runs=7`,   `total_episodes=7`
* full all-planners:
  + `camera_ready_all_planners_full2_20260217_112600`
  + `total_runs=7`,   `successful_runs=7`,   `total_episodes=315`
  + campaign runtime: `130.03s`
  + publication bundle created

Artifact locations:

* campaign root:
  + `output/benchmarks/camera_ready/camera_ready_all_planners_snqi_multiseed_verify_20260217_123437`
* publication bundle:
  + `output/benchmarks/publication/camera_ready_all_planners_snqi_multiseed_verify_20260217_123437_publication_bundle`

## Remaining Camera-Ready Gaps

The pipeline is complete and reproducible, but final publication-grade reporting
still requires:

* release metadata finalization:
  + replace `release_tag`/DOI placeholders in campaign config before archival
