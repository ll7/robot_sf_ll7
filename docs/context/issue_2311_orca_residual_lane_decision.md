# Issue #2311 ORCA-Residual Lane Decision

Issue: [#2311](https://github.com/ll7/robot_sf_ll7/issues/2311)
Parent issue: [#1475](https://github.com/ll7/robot_sf_ll7/issues/1475)
Date: 2026-06-05
Status: decision recorded; diagnostic smoke evidence only.

## Decision

Choose `revise_residual_objective`.

Do not rerun the same smoke packet unchanged, do not submit `nominal_sanity`, and do not stop the
ORCA-residual BC lane yet. The next useful issue should revise the learned-residual objective or
candidate training/evaluation contract so the residual policy is optimized for measurable route
progress under the guarded runtime contract, then rerun the same bounded smoke gate.

Uncertainty: 0.30. The evidence clearly rejects blind rerun and nominal escalation, but it does not
fully separate a weak residual target from a too-small BC dataset without another bounded revision.

## Follow-Up Status

Issue [#2390](https://github.com/ll7/robot_sf_ll7/issues/2390) implements the selected
`revise_residual_objective` route as `orca_residual_guarded_ppo_progress_v1`: a smoke-only
progress probe that keeps the hard guard authoritative, increases the forward residual envelope
from `0.25` to `0.35`, and uses the existing ORCA-prior v2 progress-comparison margin value
`0.08`. This is only a launch-packet revision until the bounded #1475 smoke rerun produces durable
evidence.

## Evidence Table

| Mechanism | Source issue | Evidence tier | Config | Seeds | Artifacts | Metrics | Verdict | Caveats |
|---|---:|---|---|---|---|---|---|---|
| ORCA-residual BC smoke | #1475 / #1967 / #2265 / #2298 / #2311 | diagnostic smoke | `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml`; `configs/training/orca_residual/orca_residual_bc_issue_1475_smoke_pretrain.yaml` | submitted as `"111:112:113"`; tracked smoke row records 1 usable episode | `docs/context/evidence/issue_1967_orca_residual_bc_smoke_adapter_summary.json`; `docs/context/evidence/issue_1475_orca_residual_bc_smoke_12749_summary.json`; `docs/context/policy_search/reports/2026-06-05_orca_residual_guarded_ppo_v0_smoke.md`; `docs/context/issue_2272_orca_residual_launch_packet_status.md` | `success_rate=0.0`; `collision_rate=0.0`; `near_miss_rate=0.0`; `timeout_low_progress=1`; `mean_avg_speed=0.8038`; `shield_decision_count=80`; `shield_override_rate=0.0`; `shield_hard_constraint_violation_rate=0.0` | `revise_residual_objective` before rerun or nominal escalation | One smoke row is diagnostic-only, not benchmark evidence. Raw SLURM outputs remain local/non-durable; compact summary/report are tracked. |

## Observed Evidence

- Issue #1967 repaired the runtime observation adapter enough to produce a usable smoke row after
  earlier flat-Box mismatch failures, but that repaired row still had `success_rate=0.0` with
  `termination_reason=max_steps`.
- Job `12749` then reached smoke execution after lineage validation, dataset collection, BC
  pretraining, candidate materialization, and the post-adapter runtime contract fixes.
- The smoke runner produced one usable JSONL row, so the runner-level result was `pass`, but the
  Issue #1475 gate correctly failed closed because success is required before nominal escalation.
- The tracked compact summary reports `success_rate=0.0`, `collision_rate=0.0`,
  `near_miss_rate=0.0`, and `failure_mode_counts.timeout_low_progress=1`.
- The row did not show guard saturation: `shield_decision_count=80`, `shield_override_rate=0.0`,
  and `shield_hard_constraint_violation_rate=0.0`.
- The smoke report records negative success deltas against `goal`, `orca`, and `ppo` baselines on
  the same smoke surface, so `planner_sanity_simple` is not a useful target to weaken before the
  residual candidate changes.

## Decision Rationale

`rerun_with_corrected_dataset` is not selected because the tracked evidence already contains a
usable smoke row after the adapter and missing-JSONL blockers were fixed. Another unchanged rerun
would not test a different hypothesis.

`revise_smoke_target` is not selected as the primary decision because a candidate that cannot make
progress on `planner_sanity_simple` has not earned a broader nominal target. The smoke target may
need additional diagnostics later, but weakening or bypassing it would blur the gate.

`stop_ORCA_residual_BC_lane` is premature. The smoke result rejects the current candidate, not the
whole ORCA-residual idea, because the job exercised only a bounded v0 smoke row.

`revise_residual_objective` best matches the evidence. The residual policy was allowed through the
guarded path without hard-constraint violations or override saturation, yet it timed out with low
progress. The next revision should make progress on the route an explicit objective or diagnostic
target under the same guarded command contract before spending SLURM on another smoke.

## Next Small Issue Shape

Open a follow-up implementation or analysis issue only after naming the revision. A useful child
would include:

- target hypothesis: current residual BC target does not optimize enough route progress under the
  guarded ORCA runtime contract;
- comparator: job `12749` current v0 smoke row;
- minimum evidence: one bounded smoke row with success, collision, near-miss, residual clipping,
  guard override, fallback/degraded status, and durable compact summary;
- stop rule: if revised smoke still has `success_rate=0.0` with `timeout_low_progress`, either
  revise the dataset/objective once more with a named reason or stop the BC lane;
- no nominal escalation unless the revised smoke passes the #1475 gate.

## Claim Boundary

This decision is research-routing evidence. It is not learned-residual success evidence, benchmark
strength evidence, or paper-facing hybrid-learning evidence.

## Validation

Checked:

```bash
python -m json.tool docs/context/evidence/issue_1475_orca_residual_bc_smoke_12749_summary.json
```

Also inspected:

- `docs/context/policy_search/reports/2026-06-05_orca_residual_guarded_ppo_v0_smoke.md`
- `docs/context/evidence/issue_1967_orca_residual_bc_smoke_adapter_summary.json`
- `docs/context/issue_2272_orca_residual_launch_packet_status.md`
- `docs/context/slurm_issue_batch_status_2026-05-21.md`
