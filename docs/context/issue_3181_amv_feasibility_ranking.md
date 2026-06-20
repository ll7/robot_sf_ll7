# Issue #3181 AMV Feasibility Ranking Slice (2026-06-20)

Issue: [#3181](https://github.com/ll7/robot_sf_ll7/issues/3181)
Predecessor: [issue_3170_amv_feasibility_ranking_stress.md](issue_3170_amv_feasibility_ranking_stress.md)
Status: bounded diagnostic feasibility direction, not benchmark-strength or paper-facing evidence.

## Question

Does the #2446 actuation-feasibility ordering generalize beyond one scenario and one seed when
`hybrid_rule_v3_fast_progress` is paired against `actuation_aware_hybrid_rule_v0` under the same
synthetic AMV actuation profile?

## Slice

- Config: `configs/benchmarks/issue_3181_amv_feasibility_ranking_slice_v0.yaml`
- Scenarios: `classic_bottleneck_high`, `classic_cross_trap_high`
- Seeds: `101`, `102`
- Variants: `hybrid_rule_v3_fast_progress` vs `actuation_aware_hybrid_rule_v0`
- Synthetic actuation profile: `amv-actuation-stress-v0`, diagnostic-only and not
  hardware-calibrated.

The local campaign completed 8 episodes with two successful planner rows. Both rows were
`adapter`/`available`; no fallback or degraded rows were included.

## Result

`actuation_aware_hybrid_rule_v0` was weakly better on command clipping in this small paired slice:

- mean paired command-clip delta, intervention minus baseline: `-0.159375`;
- improved pairs: 3;
- tied pairs: 1;
- worse pairs: 0.

This is only a bounded diagnostic feasibility direction. It does not show planner improvement:
success stayed zero for every paired row, one `classic_cross_trap_high` seed-102 pair collided in
both variants, and `classic_cross_trap_high` seed 101 improved clipping while final route progress
worsened.

## Claim Boundary

Supported:

- Within this 2-scenario x 2-seed synthetic diagnostic slice, the actuation-aware variant reduced
  or tied command clipping relative to the baseline.

Not supported:

- benchmark-strength AMV ranking;
- paper-facing result;
- hardware-calibrated AMV actuation claim;
- success, collision, or planner-quality superiority.

## Evidence

Compact artifact:
[issue_3181_amv_feasibility_ranking_2026-06-20/summary.json](evidence/issue_3181_amv_feasibility_ranking_2026-06-20/summary.json)

The evidence directory README records the generation route and parent validation commands. Raw
campaign outputs under `output/` are worktree-local and disposable.
