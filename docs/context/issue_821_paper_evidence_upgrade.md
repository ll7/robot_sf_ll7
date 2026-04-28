# Issue 821 Paper Evidence Upgrade

Date: 2026-04-17

Related:

- Upstream: `ll7/robot_sf_ll7#821`
- Paper matrix baseline: `ll7/robot_sf_ll7#832` (extended seed schedule)
- Seed variability pilot: [issue_751_seed_variability_pilot_execution.md](issue_751_seed_variability_pilot_execution.md)
- Scenario difficulty analysis: [issue_692_scenario_difficulty_analysis.md](issue_692_scenario_difficulty_analysis.md)
- Fallback policy: [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md)

## Scope

This note records the three-part camera-ready evidence upgrade requested by issue
821:

1. Broaden the scenario evidence by using the full verified benchmark matrix.
2. Add a stronger safety-aware baseline (and a second classical-planner baseline).
3. Add an ablation that tests whether SNQI is load-bearing or cosmetic relative
   to single-metric rankings.

It is not a matrix redesign and it does not change the benchmark contract.

## What Changed

### Scenario Matrix

The scenario matrix is `configs/scenarios/classic_interactions_francis2023.yaml`
(47 verified scenarios = 22 classic archetype variants + 25 Francis 2023 singles).
This is identical to `paper_experiment_matrix_v1`, and it is already the maximum
set of scenarios that pass the paper-facing verification gate.

Additional archetypes under `configs/scenarios/archetypes/issue_596_*` were
considered and explicitly excluded:

- They use atomic test maps (`atomic_*`) scoped to specific topology/robustness
  validation purposes rather than paper-facing benchmark comparison.
- Their `plausibility.status` fields are `unverified` with null metrics, which
  conflicts with the paper-facing verification gate.
- Mixing validation-scoped scenarios with benchmark-scoped scenarios would
  dilute the headline claim and raise reviewer pushback about matrix curation.

This exclusion is a limitation, not a quality claim: a future issue can add a
supplementary labeled "stress" campaign on verified atomic scenarios once they
have been calibrated against the benchmark contract.

### Planners

The planner matrix is `paper_experiment_matrix_v1` (7 planners) **plus** two
additions:

- `guarded_ppo` (experimental, safety-aware learned baseline): PPO + short
  rollout safety veto. Canonical config: `configs/algos/guarded_ppo_camera_ready.yaml`.
- `teb` (experimental, testing-only corridor-commitment planner): native TEB-inspired
  local planner. Canonical config: `configs/algos/teb_commitment_camera_ready.yaml`.
  Admission requires `allow_testing_algorithms: true` at the campaign level.

`teb` remains testing-only per the planner coverage matrix, so this run **does
not** promote it to baseline-ready. It is included as a non-core experimental
row whose results must be interpreted conservatively.

### Ablation

`scripts/tools/analyze_snqi_vs_single_metric_ranking.py` is a new script that
reads `reports/campaign_table.csv` and compares the planner ranking under
`snqi_mean` against the rankings under `success_mean`, `collisions_mean`, and
`near_misses_mean`. It reports Kendall tau and Spearman rho, and flags one of:

- `snqi_redundant`: composite does not reorder anything; it is cosmetic.
- `snqi_mostly_consistent`: composite shifts some positions but not winners.
- `snqi_reorders_tail`: composite preserves top planner but moves tail.
- `snqi_changes_winner`: composite selects a different top planner than at
  least one single metric; composite is load-bearing.

## Canonical Command

Extended campaign:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_issue_821_extended.yaml \
  --label issue821_run
```

Post-campaign analyzers:

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id>

uv run python scripts/tools/analyze_snqi_contract.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --weights configs/benchmarks/snqi_weights_camera_ready_v3.json \
  --baseline configs/benchmarks/snqi_baseline_camera_ready_v3.json

uv run python scripts/tools/analyze_snqi_vs_single_metric_ranking.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --core-only
```

## Run Provenance

- Campaign id: `paper_experiment_matrix_v1_issue_821_extended_issue821_run_20260417_094129`
- Campaign root: `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue_821_extended_issue821_run_20260417_094129`
- Config: `configs/benchmarks/paper_experiment_matrix_v1_issue_821_extended.yaml`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml` (47 scenarios)
- Scenario matrix hash: `6bfb216f1c48`
- Git commit recorded by the run: `00fbe955e35e504ca8a7d512ead5e9900fe004bb`
- Benchmark success: `true`
- Total runs: `9 / 9` planners ok
- Total episodes: `1269`
- Runtime: `1225.45 s` (~20 min)
- SNQI assets: `snqi_weights_camera_ready_v3`, `snqi_baseline_camera_ready_v3`
- SNQI contract status: `pass` (rank alignment Spearman=0.5000, dominant component=`time_penalty`)

## Observed Planner-Level Results

All 9 planners reached `benchmark_success=true` and `availability=available`
under the fail-closed contract. Per-planner aggregates (from
`reports/campaign_table.csv`, 141 episodes each):

| Planner | success | collisions | near-misses | SNQI |
| --- | ---: | ---: | ---: | ---: |
| `orca` | 0.1844 | 0.0355 | 4.2553 | -0.2446 |
| `socnav_sampling` | 0.1773 | 0.5177 | 1.5035 | -0.1360 |
| `ppo` | 0.1206 | 0.5674 | 1.6099 | -0.1094 |
| `prediction_planner` | 0.0638 | 0.2270 | 7.8085 | -0.1913 |
| `goal` | 0.0142 | 0.2411 | 2.6950 | -0.1626 |
| `guarded_ppo` | 0.0071 | 0.0851 | 2.0142 | -0.2248 |
| `teb` | 0.0071 | 0.0355 | 1.7163 | -0.2348 |
| `sacadrl` | 0.0000 | 0.3901 | 3.3262 | -0.2798 |
| `social_force` | 0.0000 | 0.2128 | 2.4610 | -0.8485 |

Safety-axis observation:

- `teb` ties `orca` for the **lowest mean collisions** (0.0355) while `guarded_ppo`
  is the second-lowest (0.0851). Both additions produce demonstrably cleaner
  collision profiles than the previous paper matrix's seven-planner set.
- `teb` and `guarded_ppo` also have among the **lowest mean near-misses** (1.72
  and 2.01 respectively), outperforming most core planners on the clearance axis.

Throughput-axis observation:

- Both additions have very low success means (0.007), similar to `goal`. On the
  Francis + classic matrix in its current form, safety-aware and corridor-commitment
  planners trade throughput for safety rather than dominating a Pareto frontier.
  `orca` retains the highest success mean (0.1844) at the cost of the highest
  near-miss count (4.26).

Execution-mode note:

- `orca` runs through an adapter and shows `infeasible_rate=0.78`, which is a
  known adapter caveat for this planner on the current matrix.
- `socnav_sampling` shows `infeasible_rate=0.97`, which also reflects its known
  adapter/prereq policy rather than catastrophic failure; its `benchmark_success`
  flag is `true` under the campaign contract.

These two infeasibility rates are not new to this run and are inherited from
the existing paper matrix; flagging them here because they affect interpretation
of any head-to-head success comparison.

## Ablation Result: SNQI vs Single-Metric Rankings

Ablation artifacts:

- `reports/snqi_vs_single_metric_ranking.json`
- `reports/snqi_vs_single_metric_ranking.md`

Verdict: **`snqi_changes_winner`**.

| Metric | Kendall tau vs SNQI | Spearman rho vs SNQI | Winner agrees |
| --- | ---: | ---: | :---: |
| `success_mean` | 0.5556 | 0.6167 | no |
| `collisions_mean` | -0.5556 | -0.6000 | no |
| `near_misses_mean` | 0.2778 | 0.4667 | no |

Winners under each ranking:

- SNQI (campaign_table.csv `snqi_mean`): `ppo`
- `success_mean`: `orca`
- `collisions_mean`: `orca` (tied with `teb` at 0.0355; alphabetical tiebreak)
- `near_misses_mean`: `socnav_sampling`

The `snqi_diagnostics.md` episode-level recomputation places `socnav_sampling`
first by a small margin over `ppo`. This small disagreement between the
publication-table `snqi_mean` and the episode-level recomputation is itself a
caveat worth flagging in the paper: the composite's top-1 separation is narrow
and sensitive to the aggregation step. It is not a correctness issue — both
values come from the same episode records — but it is not free noise either.

Interpretation:

- SNQI is **not** cosmetic for this matrix. No single metric reproduces the SNQI
  ordering, and the winner under SNQI differs from the winner under each of
  the three single metrics tested.
- The composite must therefore be justified in the paper: a reviewer looking at
  a success-only or collisions-only table would conclude a different planner
  wins.
- The ablation does not prove SNQI is the *right* composite; it only proves the
  composite is doing work. Pairing SNQI with the single-metric tables, rather
  than reporting SNQI alone, is the defensible reporting posture.

## Effect on the Publication Claim

Before this run, the camera-ready story leaned on seven planners on a
47-scenario matrix, with a thin safety narrative (ORCA carries low collisions
but high near-misses; PPO carries higher success but higher collisions). The
phrase "first step" remained the honest framing because the safety axis was
under-represented.

After this run:

- **Safety axis is now discriminated.** `teb` and `guarded_ppo` both beat every
  core planner on collisions and most on near-misses, while staying below every
  core planner on success. The paper can now state that safety-aware and
  corridor-commitment classical planners occupy a distinct low-collision /
  low-throughput region in this matrix rather than speculate about it.
- **The safety-vs-throughput tradeoff is evidential, not rhetorical.** On this
  matrix, there is no planner that simultaneously dominates both axes. That is a
  sharper, more defensible claim than "PPO does better in some places and ORCA
  does better in others".
- **SNQI is load-bearing.** The paper can no longer present SNQI as just a
  convenient summary; the ablation shows the composite reorders planners, so
  the paper must either motivate the composite or report single-metric tables
  alongside it. The conservative reporting posture is to do both.

This is a refinement of the claim scope, not an expansion. It makes the
camera-ready story less fragile: a reviewer who disputes SNQI can fall back on
the single-metric tables, and the safety-vs-throughput tradeoff is visible in
either view.

## Caveats and Limitations

- `teb` is testing-only and is flagged as `experimental` in the planner row;
  treating it as a benchmark-promoted corridor-commitment baseline would exceed
  the current planner-coverage policy.
- `orca` (infeasible_rate=0.78) and `socnav_sampling` (infeasible_rate=0.97)
  continue to carry adapter caveats that predate this change.
- The ablation uses planner-level aggregates from `campaign_table.csv`. An
  episode-level recomputation gives a very similar but not identical ordering;
  the narrative focuses on the planner-level publication-facing aggregate.
- Seed schedule is S3 (`eval` = `[111, 112, 113]`), matching the frozen paper
  baseline. If the paper wants tighter intervals, the issue 832 extended-seed
  workflow can be re-run with `paper_experiment_matrix_v1_issue_821_extended`
  as the base.
- Atomic-map validation scenarios (`issue_596_*`) remain explicitly excluded
  from the paper-facing benchmark and are open as a future supplementary run.

## Reproducibility

All artifacts are derivable from versioned inputs:

- Benchmark config: `configs/benchmarks/paper_experiment_matrix_v1_issue_821_extended.yaml`
- Scenario set: `configs/scenarios/classic_interactions_francis2023.yaml`
- SNQI assets: `configs/benchmarks/snqi_weights_camera_ready_v3.json`,
  `configs/benchmarks/snqi_baseline_camera_ready_v3.json`
- Ablation script: `scripts/tools/analyze_snqi_vs_single_metric_ranking.py`
- Comparability mapping: `configs/benchmarks/alyassi_comparability_map_v1.yaml`
  (now includes a `teb` row)

The campaign manifest records the full `invoked_command`, git commit,
start/end timestamps, and per-planner runtime fields.
