# Issue 1023 Scenario-Horizon Benchmark Surface

Date: 2026-05-06

Related:

- Upstream issue: `ll7/robot_sf_ll7#1023`
- Scenario horizon schedule: `configs/policy_search/scenario_horizons_h500.yaml`
- New benchmark config: `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml`
- Experimental candidate note:
  `docs/context/issue_1023_experimental_benchmark_candidates.md`
- Preflight evidence:
  `docs/context/evidence/issue_1023_scenario_horizons_preflight_2026-05-06/`
- Local full-campaign evidence:
  `docs/context/evidence/issue_1023_scenario_horizons_local_full_2026-05-06/`
- Fixed-horizon reference evidence:
  `docs/context/evidence/camera_ready_all_planners_2026-05-04/`
- H500 policy-search source evidence:
  `docs/context/evidence/policy_search_h500_2026-05-06/`
- Candidate-augmented preflight evidence:
  `docs/context/evidence/issue_1023_candidate_augmented_preflight_2026-05-06/`
- Candidate-augmented local full evidence:
  `docs/context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/`
- SNQI calibration follow-up: `ll7/robot_sf_ll7#1038`
- Fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`

## Goal

Issue 1023 asks whether the h500-derived scenario-specific horizon schedule should become a
paper-facing benchmark surface. This branch makes that schedule runnable through the camera-ready
benchmark path, preserves preflight proof, records a full local non-Slurm campaign requested by the
maintainer, and compares it against the May 4 fixed-horizon reference.

## What Changed

- `robot_sf.benchmark.camera_ready_campaign` now accepts an optional top-level
  `scenario_horizons` YAML path.
- `robot_sf.benchmark.map_runner` now resolves policy-search candidate manifests in benchmark
  runs. It merges `base_config_path` and `params`, applies scenario-specific parameter overrides,
  and honors `scenario_algo_overrides` such as the ORCA switch used by
  `scenario_adaptive_hybrid_orca_v1`.
- The schedule is joined by scenario name/id and patches each loaded scenario's
  `simulation_config.max_episode_steps` before the map runner sees the matrix.
- Each patched scenario receives `metadata.scenario_horizon` with source, horizon, status, and
  bucket. This keeps the three `planner_blocked` scenarios visible instead of excluding them.
- Preflight and matrix artifacts now report `horizon_mode=scenario_horizons`, the schedule path,
  horizon range, and status counts.
- The issue-1023 benchmark config is now the current long-horizon collection. It remains
  paper-facing for validation/comparability/SNQI contracts but disables publication bundle export
  because the candidate-augmented local full run has SNQI contract status `fail`.
- The collection now includes `scenario_adaptive_hybrid_orca_v1` and
  `hybrid_rule_v3_fast_progress_static_escape` as experimental candidates, not as
  baseline-ready headline planners.

## Why The Horizons Changed

The previous fixed `horizon: 100` budget was too blunt for the mixed classic and Francis scenario
matrix. Some long routes and constrained-geometry interactions were counted as route-incomplete
because the episode budget ended, not necessarily because the planner had stopped making useful
progress. The h500-derived schedule instead uses successful safe incumbent traces to estimate a
per-scenario route-completion budget, with `p95_multiplier=1.2`, a `20` step buffer, an `80` step
floor, and a `600` step cap.

The underlying reason is likely route-budget confounding: scenario geometry, route length, social
interaction timing, and local bottlenecks were all being compressed into a single 100-step cutoff.
The longer per-scenario horizons reduce that confound and make route completion less dependent on
the shortest routes in the suite.

The change is not a free improvement. Longer horizons increase exposure time, which can raise
collision and near-miss opportunities. The local seven-planner campaign already showed exactly that
tradeoff: success improved for matched planners, while collision and near-miss means also rose.
For release claims, the long-horizon surface must therefore be interpreted with safety metrics,
fallback/degraded status, runtime, and candidate provenance together.

## Canonical Commands

Local preflight, already run:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --mode preflight \
  --campaign-id issue1023_scenario_horizons_preflight_2026-05-06 \
  --log-level INFO
```

Local full campaign, run from this non-Slurm worktree per maintainer direction:

```bash
.venv/bin/python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --output-root output/benchmarks/issue_1023 \
  --campaign-id issue1023_scenario_horizons_h500_local_2026-05-06 \
  --mode run \
  --log-level INFO
```

Post-run diagnostics:

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06
```

Fixed-vs-scenario comparison:

```bash
uv run python scripts/tools/compare_camera_ready_campaigns.py \
  --base-campaign-root docs/context/evidence/camera_ready_all_planners_2026-05-04 \
  --candidate-campaign-root output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06 \
  --output-json output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/fixed_vs_scenario_horizon_comparison.json \
  --output-md output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/fixed_vs_scenario_horizon_comparison.md
```

Canonical Auxme Slurm command retained for reproducibility if maintainers later require a
cluster-side rerun:

```bash
ISSUE791_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
ISSUE791_BENCHMARK_LABEL=issue1023-scenario-horizons-h500 \
ISSUE791_BENCHMARK_OUTPUT_ROOT=output/benchmarks/issue_1023 \
sbatch SLURM/Auxme/issue_791_benchmark.sl
```

## Preflight Evidence

The tracked preflight bundle from before candidate augmentation records:

- 48 scenarios.
- 7 planners.
- Seed set `eval` = `[111, 112, 113]`.
- Horizon mode `scenario_horizons`.
- Horizon range 102-600 steps.
- Status counts: 45 `recommended`, 3 `planner_blocked`.

The current config has 9 planners after adding the two experimental candidates. The tracked
candidate-augmented preflight records 48 scenarios, 9 planners, `scenario_horizons` mode, horizon
range 102-600, and status counts of 45 `recommended` plus 3 `planner_blocked`.

The three `planner_blocked` scenarios remain in the matrix with the h600 cap and must be reported
separately in the final comparison. They should not be excluded from aggregate accounting without a
separate paper-table decision.

## Comparison Boundary

The current fixed-horizon reference remains the May 4 camera-ready all-planners evidence. The local
scenario-horizon campaign is valid issue-1023 evidence for this branch, but it is not a Slurm
reproduction and should be labeled as local non-Slurm evidence in PR and paper-facing handoff text.

The tracked fixed-vs-scenario comparison reports:

- planner,
- scenario,
- scenario family,
- success,
- collision,
- near-miss,
- `unfinished_mean = 1 - success_mean` as the route-incomplete comparison metric,
- fallback/degraded/not-available status.

Raw per-episode timeout attribution is not preserved in the compact May 4 fixed-horizon evidence,
so this note does not claim a raw timeout-vs-collision decomposition for that reference. Use
`unfinished_mean` for the comparable route-incomplete signal and the raw local JSONL only for
local-only debugging.

Fallback/degraded statuses must be interpreted through the fail-closed benchmark fallback policy.
`orca` and `sacadrl` keep their existing fallback prereq policies in the config, while
`socnav_sampling` remains fail-fast.

## Local Full Campaign Evidence

The tracked local full-campaign bundle records:

- 7 successful planner runs.
- 1008 episodes.
- Runtime: 819.7857 seconds.
- Campaign warning: SNQI contract status `warn`.
- Analyzer finding: no automated campaign inconsistencies.
- Runtime hotspots: `prediction_planner`, `ppo`, and `social_force`.
- Hardest scenarios by analyzer: `classic_cross_trap_high`,
  `francis2023_narrow_hallway`, `francis2023_robot_crowding`, and
  `classic_station_platform_medium`.
- Verified-simple assessment: `rerun_required` because this campaign does not include the
  verified-simple candidate scenarios.

Planner-level local scenario-horizon outcomes:

| planner | success | collisions | SNQI |
|---|---:|---:|---:|
| goal | 0.0556 | 0.6181 | -0.1904 |
| orca | 0.7569 | 0.1667 | -0.2513 |
| ppo | 0.8056 | 0.1667 | -0.2074 |
| prediction_planner | 0.4931 | 0.4514 | -0.1408 |
| sacadrl | 0.0833 | 0.6667 | -0.2726 |
| social_force | 0.0139 | 0.3819 | -0.9537 |
| socnav_sampling | 0.4028 | 0.5972 | -0.0848 |

Fixed-vs-scenario planner-level deltas for the seven matched planners show success increased under
scenario-specific horizons, but every matched planner also saw higher collision and near-miss means.
The comparison is therefore a horizon-confounding analysis, not a clean benchmark improvement.
`socnav_bench` is a coverage gap because it is present in the May 4 fixed-horizon evidence but not
in the issue-1023 scenario-horizon config.

## Candidate-Augmented Local Full Campaign

The tracked 9-planner local full-campaign bundle records:

- 9 successful planner runs.
- 1296 episodes.
- Runtime: 1966.5880 seconds.
- Benchmark success: true.
- No failed, partial, not-available, fallback, or degraded planners.
- Campaign warning: SNQI contract status `fail` with warn enforcement.
- Analyzer finding after report-table regeneration: no automated campaign inconsistencies.
- Runtime hotspots: `hybrid_rule_v3_fast_progress_static_escape`,
  `scenario_adaptive_hybrid_orca_v1`, and `prediction_planner`.

Planner-level candidate-augmented outcomes:

| planner | group | success | collisions | near misses | SNQI |
|---|---|---:|---:|---:|---:|
| goal | core | 0.0556 | 0.6181 | 7.0556 | -0.1904 |
| orca | core | 0.7569 | 0.1667 | 13.6806 | -0.2513 |
| social_force | core | 0.0139 | 0.3819 | 5.2917 | -0.9537 |
| ppo | experimental | 0.8056 | 0.1667 | 5.6181 | -0.2074 |
| prediction_planner | experimental | 0.4931 | 0.4514 | 24.9375 | -0.1408 |
| sacadrl | experimental | 0.0833 | 0.6667 | 5.7778 | -0.2726 |
| socnav_sampling | experimental | 0.4028 | 0.5972 | 1.6458 | -0.0848 |
| scenario_adaptive_hybrid_orca_v1 | experimental | 0.9097 | 0.0278 | 19.4583 | -0.0835 |
| hybrid_rule_v3_fast_progress_static_escape | experimental | 0.9028 | 0.0278 | 20.7778 | -0.0874 |

The two new candidates are the best route-completion rows in this matrix, but they are slow and
remain experimental. Their high near-miss counts under longer horizons are not compatible with a
clean headline claim without deeper scenario-level safety review.

## Recommendation

Do not promote the scenario-horizon surface as the paper headline benchmark. The local full run
shows the longer scenario-specific budgets reduce route-incompletion and raise success, but the same
surface also increases collisions and near-misses.

Use this evidence as a sensitivity/confounding analysis for the paper-facing benchmark. If the paper
mentions scenario-specific horizons, present them beside the fixed-horizon surface with explicit
safety caveats, the SNQI `fail` status on the candidate-augmented run, the local non-Slurm
provenance, and the `socnav_bench` coverage gap. Do not publish a release tag from this evidence
until the SNQI contract failure and candidate safety interpretation are resolved. Follow-up issue
`ll7/robot_sf_ll7#1038` tracks the SNQI calibration or claim-scope decision. The current decision
is recorded in `docs/context/issue_1038_h500_snqi_contract.md`: keep h500 scenario horizons as an
experimental sensitivity surface and do not overwrite the camera-ready v3 SNQI assets from this
local evidence.

## Validation Run

Commands run in this branch:

```bash
uv run pytest tests/benchmark/test_camera_ready_campaign.py::test_scenario_horizon_schedule_applies_to_loaded_campaign_scenarios \
  tests/benchmark/test_camera_ready_campaign.py::test_preflight_reports_scenario_horizon_schedule_summary \
  tests/benchmark/test_camera_ready_campaign.py::test_issue_1023_scenario_horizon_config_uses_h500_schedule -q
```

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --mode preflight \
  --campaign-id issue1023_scenario_horizons_preflight_2026-05-06 \
  --log-level INFO
```

```bash
.venv/bin/python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --output-root output/benchmarks/issue_1023 \
  --campaign-id issue1023_scenario_horizons_h500_local_2026-05-06 \
  --mode run \
  --log-level INFO
```

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06
```

```bash
uv run python scripts/tools/compare_camera_ready_campaigns.py \
  --base-campaign-root docs/context/evidence/camera_ready_all_planners_2026-05-04 \
  --candidate-campaign-root output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06 \
  --output-json output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/fixed_vs_scenario_horizon_comparison.json \
  --output-md output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/fixed_vs_scenario_horizon_comparison.md
```

```bash
uv run pytest tests/tools/test_compare_camera_ready_campaigns.py -q
```

The Slurm command remains documented above for optional cluster reproduction, but the issue-1023
full campaign evidence preserved here is the local non-Slurm run requested for this worktree.

Additional targeted validation for candidate-manifest support and config membership:

```bash
uv run pytest \
  tests/benchmark/test_map_runner_utils.py::test_resolve_policy_search_candidate_runtime_merges_base_and_scenario_override \
  tests/benchmark/test_map_runner_utils.py::test_resolve_policy_search_candidate_runtime_switches_algo_for_scenario \
  tests/benchmark/test_camera_ready_campaign.py::test_issue_1023_scenario_horizon_config_uses_h500_schedule -q
```

Candidate-augmented preflight:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --mode preflight \
  --campaign-id issue1023_scenario_horizons_candidates_preflight_2026-05-06 \
  --log-level INFO
```

Candidate-augmented local full campaign:

```bash
.venv/bin/python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --output-root output/benchmarks/issue_1023 \
  --campaign-id issue1023_scenario_horizons_candidates_local_2026-05-06 \
  --mode run \
  --log-level INFO
```

Candidate-augmented analyzer:

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06
```
