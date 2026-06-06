# Issue #2259 AMV Clipping Versus Success Boundary

Date: 2026-06-05
Issue: <https://github.com/ll7/robot_sf_ll7/issues/2259>
Source evidence: Issues #2224 and #2268.

## Scope

This note closes the parent analysis question for why AMV-aware scoring reduced synthetic command
clipping in the Issue #2224 / PR #2249 smoke without improving task success. It synthesizes tracked
compact evidence only; it does not rerun the smoke, add a planner variant, or make calibrated AMV
hardware claims.

Compact evidence summary:
`docs/context/evidence/issue_2259_amv_clipping_success_boundary_2026-06-05/summary.json`.

## Observed Result

The matched `amv_actuation_smoke` slice compared `hybrid_rule_v3_fast_progress` against
`actuation_aware_hybrid_rule_v0` on `classic_cross_trap_high` with the synthetic
`amv-actuation-stress-v0` profile.

| Metric | `hybrid_rule_v3_fast_progress` | `actuation_aware_hybrid_rule_v0` | Delta |
| --- | ---: | ---: | ---: |
| Success rate | 0.0000 | 0.0000 | 0.0000 |
| Collision rate | 0.0000 | 0.0000 | 0.0000 |
| Near-miss rate | 0.0000 | 0.0000 | 0.0000 |
| Failure mode | `timeout_low_progress: 1` | `timeout_low_progress: 1` | unchanged |
| Mean command clip fraction | 0.2750 | 0.1875 | -0.0875 |
| Mean yaw-rate saturation fraction | 0.0000 | 0.0000 | 0.0000 |
| Mean signed braking peak | -2.5000 | -2.5000 | 0.0000 |
| Mean average speed | 1.6623 | 1.6670 | +0.0047 |
| Mean minimum distance | 2.1571 | 2.3627 | +0.2056 |

The actuation-aware row improved command feasibility as measured by clipping, but it did not change
the terminal outcome or the failure mechanism class.

## Explanation

The most plausible explanation is that the smoke was success-blocked by route progress rather than
by raw command feasibility:

- `actuation_aware_hybrid_rule_v0` reduced clipping by `0.0875` absolute.
- Both rows still had zero success and one `timeout_low_progress` failure.
- Average speed changed by only `+0.0047`.
- Yaw-rate saturation stayed `0.0000`.
- Signed braking peak stayed `-2.5000`.

Classification: `diagnostic_success_blocked_by_route_progress`.

Confidence: `0.74` that reduced clipping was orthogonal to success on this one-row smoke. The
confidence is not higher because per-step route-progress and command-feasibility traces are still
missing.

## Recommendation

Recommendation: `keep_diagnostic_only`.

Keep `actuation_aware_hybrid_rule_v0` as a useful synthetic feasibility diagnostic, not a
navigation-success result. Do not propose a broad AMV actuation benchmark or paper-facing AMV claim
from this smoke. The next useful proof is a trace-level rerun or extractor for the same matched
slice that records route-progress, selected command, clipping/projection, and deadlock/oscillation
signals over time.

If that trace shows route progress remains blocked while command feasibility improves, the planner
work should move toward route-progress or local-deadlock interventions rather than more actuation
scoring variants. If the trace shows feasibility still intermittently blocks progress, then a
narrow follow-up actuation run is justified.

## Claim Boundary

This is synthetic diagnostic evidence under the Issue #2230 AMV actuation evidence ladder. It does
not support calibrated AMV actuation, AMV hardware truth, safety certification, benchmark-strength
planner superiority, or paper-facing AMV performance.

The `research-v1.amv.calibrated_actuation` claim remains `blocked`: synthetic command-clipping
diagnostics can guide tooling and failure analysis, but they do not supply the missing calibrated
runtime/provenance fields.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_2259_amv_clipping_success_boundary_2026-06-05/summary.json
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
