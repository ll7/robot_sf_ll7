# Issue #1554 S20/S30 Seed-Budget Bundle

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1554>

## Scope

This note started as a local archival/uncertainty tooling SLURM launch packet for an
S20/S30 (20/30-seed) seed-budget comparison bundle on the h500 social-navigation surface.
It now also records the completed job `13198` S20 constraints-first decision below. The
job `13198` result replaces the old `blocked_until_run` state for the S20 H500
planner-family run, but it still does **not** assert a blanket S20/S30 ranking,
significance, safety, or paper-grade strict-ordering claim.

## Job 13198 Constraints-First Decision

Related evidence:
[evidence/issue_1554_job_13198_constraints_first_analysis/](evidence/issue_1554_job_13198_constraints_first_analysis/).

Job `13198` completed the S20 H500 split planner-family campaign with nine successful
planner rows, 8,640 episode rows, S20 per planner, and 960 episodes per planner. All nine
planner rows count as `successful_evidence` under the fail-closed benchmark-row policy
(`status=ok`, `benchmark_success=true`).

The launch-packet scope above remains historical context. The current evidence
does **not** support a blanket paper-grade strict planner ordering. It supports only the
adjacent statements named in the job `13198` packet, with explicit confidence-interval
downgrades for budget-limited adjacent pairs. SNQI remains explanatory-only for this packet
because the SNQI contract failed under warn enforcement.

Constraints-first `ci_separable` adjacent statements:

- `ppo` over `orca`
- `orca` over `prediction_planner`
- `prediction_planner` over `socnav_sampling`
- `socnav_sampling` over `sacadrl`
- `goal` over `social_force`

Constraints-first `not_statistically_distinguishable_budget` adjacent statements:

- `hybrid_rule_v3_fast_progress_static_escape` over `scenario_adaptive_hybrid_orca_v1`
- `scenario_adaptive_hybrid_orca_v1` over `ppo`
- `sacadrl` over `goal`

`diagnostic_only` statements:

- All SNQI adjacent-rank statements in
  [adjacent_rank_claims.csv](evidence/issue_1554_job_13198_constraints_first_analysis/adjacent_rank_claims.csv).
- Any statement that uses SNQI to reorder planners. SNQI changes the nominal order but
  does not change the constraints-first decision because `snqi_diagnostics.json` reports
  `contract_status=fail` with enforcement `warn`.

More seed-budget compute is conditionally justified only if a manuscript claim needs
strict adjacent ordering for the three `not_statistically_distinguishable_budget` pairs
above. It is not justified for already `ci_separable` adjacent statements, and more SNQI
compute alone is not justified until the SNQI contract issue is resolved.

## Deliverables

- Tool: `scripts/benchmark/build_s20_s30_seed_budget_bundle_issue_1554.py`
- Archive-readiness checker: `scripts/validation/check_s20_s30_archive_readiness.py`
- Launch packet: `configs/benchmarks/s20_s30_seed_budget_issue_1554_launch_packet.yaml`
- Tests: `tests/benchmark/test_s20_s30_seed_budget_bundle_issue_1554.py`
- Readiness tests: `tests/validation/test_check_s20_s30_archive_readiness.py`
- Seed set added: `configs/benchmarks/seed_sets_v1.yaml` (`paper_eval_s30`, 111-140)

## Reused canonical statistics (no reinvention)

The tool reuses existing canonical helpers rather than hand-rolling statistics:

- `robot_sf.benchmark.seed_variance.build_seed_variability_rows` /
  `compute_seed_variance` — per-planner-by-seed summaries and per-metric variance with bootstrap
  per-seed CIs (`bootstrap_mean_over_seed_means`, 1000 samples, seed 123).
- `robot_sf.benchmark.snqi.bootstrap.bootstrap_stability` — deterministic stratified bootstrap SNQI
  ranking stability (`rng=np.random.default_rng(seed)`).
- `robot_sf.benchmark.rank_metrics.rank_order` / `kendall_tau` — seed-resampling rank-flip analysis
  on the direct outcome metrics.

## Claim-map gate inputs (recorded conservatively)

All inputs below are **to be confirmed by maintainer**; this note asserts no paper claim.

- **Target claim**: the per-planner safety/quality ordering observed at S10 on the h500
  social-navigation surface is stable under a paper-facing S20 (then S30 on rank flip) seed budget,
  with bootstrap per-seed uncertainty small enough for the declared primary deltas.
- **Required metric surface**: `success`, `collisions`, `near_misses`, `time_to_goal_norm`.
  Descriptive-only (no uncertainty claim by default): min/mean clearance, min distance,
  timeout/unfinished rate, low-progress rate — consistent with issue #1545.
- **Planner rows**: `goal`, `social_force`, `orca`, `ppo`, `prediction_planner`,
  `socnav_sampling`, `sacadrl`.
- **Seed tier**: `paper_eval_s20`, escalating to `paper_eval_s30` if rankings flip under seed
  resampling or a primary delta is below the issue #1545 effect-size threshold.
- **Why S10 is insufficient** (to be confirmed by maintainer): issue #1545 records material
  aggregate seed instability across the 10 fixed S10 seeds (e.g. success range across planner-level
  seed means median `0.1875`, max `0.2917`) and `snqi_contract_status="fail"` on the S10 h500
  surface, so close paper-facing comparisons need a larger seed budget.

## Methodology and fail-closed policy

- Seed-budget methodology: [issue_1545_power_aware_seed_budget_planning.md](issue_1545_power_aware_seed_budget_planning.md).
- Fail-closed (issue #691): only `native`/`adapter` rows count as benchmark-success evidence;
  `fallback`, `degraded`, `unavailable`, `failed`, `partial`, `not_available`, and
  `diagnostic_only` rows are classified fail-closed with exact per-planner reasons and never
  counted as success. Missing/absent planner rows are classified fail-closed.

## Status

`job_13198_constraints_first_analyzed` for the completed S20 H500 planner-family run. The
historical S20/S30 bundle builder remains useful for archive/readiness checks and any future S30
escalation. For a fresh bundle/checker pass, run:

```bash
uv run python scripts/validation/check_s20_s30_archive_readiness.py --json
```

The checker validates target claim metadata, planner rows, seed tier, required metrics, output
locations, and missing-artifact diagnostics.

After readiness passes, build the reviewable bundle:

```bash
uv run python scripts/benchmark/build_s20_s30_seed_budget_bundle_issue_1554.py \
  --rows output/campaign_result_store/s20_s30_h500_social_navigation \
  --output-dir output/issue_1554_s20_s30
```

The tool emits a real bundle only when valid S20+ rows exist for at least two planners; otherwise it
honestly reports `blocked_until_run` naming the missing seed tier.
