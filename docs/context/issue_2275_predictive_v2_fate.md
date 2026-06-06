# Issue #2275 Predictive-v2 Fate Decision

Date: 2026-06-05
Issue: <https://github.com/ll7/robot_sf_ll7/issues/2275>
Parent: <https://github.com/ll7/robot_sf_ll7/issues/1490>

## Scope

This note decides whether the predictive-v2 lane should continue, revise, or stop after the
coupling preflight evidence. It does not run the old four-way expansion, train new predictive rows,
or make a paper-facing planner claim.

Compact decision evidence lives at
`docs/context/evidence/issue_2275_predictive_v2_fate_2026-06-05/decision_matrix.csv`.

## Evidence Sources

- `docs/context/issue_1543_predictive_v2_negative_audit.md`: durable same-seed negative audit for
  the #1427 obstacle-feature prerequisite.
- `docs/context/issue_1856_predictive_coupling_objective.md`: local planner-side coupling gate
  proposal and routing rules for the blocked expansion issues.
- `docs/context/issue_1897_predictive_coupling_gate_preflight.md`: local closed-loop gate result
  for the revised phase-coupled planner row.
- `docs/context/evidence/issue_1897_predictive_coupling_gate_2026-05-31/README.md`: compact
  preflight evidence and gate thresholds.
- Live issue state for Issues #1490, #1505, #1506, and #1507 checked on 2026-06-05.

## Decision

Decision: `stop_old_predictive_v2_expansion`; revise only through a new planner-coupling or
planner-aligned objective hypothesis.

Confidence: `0.9`.

Rationale:

- Issue #1543 already showed that obstacle features did not transfer to better closed-loop
  outcomes: baseline predictive success was `0.1304`, obstacle-feature success was `0.1014`, and
  hard-seed success was `0.0000` for both variants.
- Issue #1856 defined the smallest reasonable rescue gate: keep the same checkpoint and require a
  planner-side coupling row to improve closed-loop success over `baseline_like` before spending on
  ego/obstacle expansion rows.
- Issue #1897 failed that gate. Both `baseline_like` and `phase_coupled_sequence_gate` recorded
  global success `0.0000` and hard success `0.0000`; the revised row only improved global mean
  min-distance by `0.0108`.
- A clearance-only gain with zero success movement is exactly the fail condition for the #1856
  gate, not a reason to reopen #1505/#1506/#1507 as execution work.

## Gate Status

| Issue | Current recommendation | Reason |
| --- | --- | --- |
| Issue #1505 | Keep blocked. | Data-row preflight should not run until there is a new coupling/objective hypothesis that names the closed-loop mechanism it expects to improve. |
| Issue #1506 | Keep blocked. | The old four-way Slurm matrix has no passing local preflight and would spend on rows that the #1543/#1897 evidence does not justify. |
| Issue #1507 | Keep blocked or rescope to close/downgrade synthesis. | Forecast-to-control analysis is useful only after a new hypothesis exists, or as a bounded closeout that explains why predictive-v2 is being stopped. |

## Follow-Up Boundary

Do not create another predictive-v2 execution issue unless it names:

- the revised planner-coupling or planner-aligned objective mechanism,
- the checkpoint/provenance boundary,
- the closed-loop success threshold that would reopen expansion work,
- the exact artifact surface that would make the result durable.

Without that, the most useful research action is a closeout/downgrade of the predictive-v2 parent
Issue #1490 rather than another local or Slurm run.

## Claim Boundary

This is routing and analysis evidence only. It does not prove predictive forecasting is useless in
Robot SF. It says the current predictive-v2 route should not continue in its old form because both
the obstacle-feature prerequisite and the planner-side coupling preflight failed to produce
closed-loop success movement.

## Validation

```bash
for path in \
  docs/context/issue_1543_predictive_v2_negative_audit.md \
  docs/context/issue_1856_predictive_coupling_objective.md \
  docs/context/issue_1897_predictive_coupling_gate_preflight.md \
  docs/context/evidence/issue_1897_predictive_coupling_gate_2026-05-31/README.md; do
  test -f "$path"
done
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
