# Issue #1554 S20/S30 Seed-Budget Bundle

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1554>

## Scope

This note records the **local archival/uncertainty tooling and SLURM launch packet** for an
S20/S30 (20/30-seed) seed-budget comparison bundle on the h500 social-navigation surface. It does
**not** run the S20/S30 campaign (that is the SLURM half), and it does **not** assert any S20/S30
result, ranking, significance, or safety claim. The repository currently has durable **S10** h500
comparison evidence (issue #1454), not durable S20 or S30 comparison evidence, so the bundle status
is **`blocked_until_run`** until the SLURM S20/S30 campaign produces valid rows.

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

`blocked_until_run` pending the SLURM S20/S30 h500 social-navigation campaign. Run the campaign per
the launch packet, write a canonical campaign result store, then run the fail-closed
archive-readiness check:

```bash
uv run python scripts/validation/check_s20_s30_archive_readiness.py --json
```

The checker validates target claim metadata, planner rows, seed tier, required metrics, output
locations, and missing-artifact diagnostics. A blocked result is expected until the canonical S20/S30
result store exists.

After readiness passes, build the reviewable bundle:

```bash
uv run python scripts/benchmark/build_s20_s30_seed_budget_bundle_issue_1554.py \
  --rows output/campaign_result_store/s20_s30_h500_social_navigation \
  --output-dir output/issue_1554_s20_s30
```

The tool emits a real bundle only when valid S20+ rows exist for at least two planners; otherwise it
honestly reports `blocked_until_run` naming the missing seed tier.
