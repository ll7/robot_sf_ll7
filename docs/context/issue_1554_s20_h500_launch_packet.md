# Issue #1554 S20 H500 Launch Packet 2026-06-23

This launch packet makes issue #1554 concrete enough for Slurm routing without
turning it into an open-ended benchmark rerun. It targets the paper-facing
scenario-horizon h500 matrix and extends only the seed schedule from the frozen
S3 `eval` seeds to the predeclared S20 schedule in
`configs/benchmarks/seed_sets_v1.yaml`.

## Claim Gate

- Target claim: the paper-facing h500 planner comparison is stable enough to
  support planner-ranking or safety-delta language after confidence intervals
  and rank-stability analysis.
- Metric surface: success, collision, near-miss/clearance where available,
  low-progress/timeout, `time_to_goal_norm`, and SNQI for rank stability.
- Planner rows: the baseline-ready core rows in
  `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml`.
- Seed tier: S20 via `paper_eval_s20` (`111` through `130`).
- Why S10/S3 is insufficient: issue #3216 needs confidence intervals and
  bootstrap rank stability before manuscript-relied planner rankings are
  promoted beyond diagnostic or nominal status.

## Runnable Config

- Config:
  `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml`
- Seed set: `paper_eval_s20`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Horizon schedule: `configs/policy_search/scenario_horizons_h500.yaml`
- Kinematics: `differential_drive`
- Publication bundle export: disabled until final analysis and artifact
  promotion classify the result.

Canonical command shape:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml \
  --campaign-id issue1554_s20_h500
```

The Slurm wrapper should submit this command from a dedicated public worktree,
record the exact commit, and keep raw episode logs/checkpoints/videos out of
git. Compact outputs and manifest pointers belong in `docs/context/evidence/`
or an external durable artifact pointer after finalization.

## Required Closeout

After the campaign finishes, run seed sufficiency and headline-rank stability
against the S20 root and any retained S3/S10 roots:

```bash
uv run python scripts/tools/analyze_seed_sufficiency.py \
  --campaign-root <s3-or-s10-root> \
  --campaign-root <issue1554_s20_h500_root> \
  --headline-required-durable-root <issue1554_s20_h500_root> \
  --output-dir docs/context/evidence/issue_1554_s20_h500_seed_sufficiency
```

Rows with fallback, degraded, failed, partial-failure, or unavailable execution
must remain excluded from promoted claims and must carry explicit row status.

## S30 Boundary

No S30 seed set is tracked in `configs/benchmarks/seed_sets_v1.yaml` as of this
packet. S30 is therefore blocked until a seed schedule is predeclared and tested.
Do not synthesize ad hoc seeds at submit time.

## Decision Outcomes

Classify the final bundle as exactly one of:

- `paper_grade_for_named_claims`
- `not_distinguishable_at_s20`
- `diagnostic_only_missing_rows`
- `blocked_missing_durable_artifacts`
- `failed_closed`
