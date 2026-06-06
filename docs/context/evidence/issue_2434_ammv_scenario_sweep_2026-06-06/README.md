# Issue #2434 AMMV Scenario Sweep Evidence

Date: 2026-06-06

This bundle records a diagnostic-only local AMMV/default Social Force screen for Issue #2434. It
extends the Issue #2432 trace-selection negative result from one head-on slice to a compact set of
five classic close-interaction scenario families.

## Files

- `summary.json`: machine-readable result, commands, raw-output checksums, metric-delta maxima,
  and candidate pair records.
- `candidate_pair_comparison.csv`: flat scenario/seed comparison table.

Raw regenerated JSONL outputs are not tracked. Their checksums are preserved in `summary.json`.

## Result

The sweep compared 15 matched default/AMMV episode pairs across:

- `classic_bottleneck_medium`
- `classic_cross_trap_low`
- `classic_doorway_high`
- `classic_head_on_corridor_medium`
- `classic_overtaking_medium`

Every pair had 120 recorded frames and matched on robot state, pedestrian state, selected action,
planner event, `ammv.pedestrian_force_vectors`, status, outcome, steps, success, collisions,
clearance, speed, force metrics, and every numeric metric present in both JSONL rows. The maximum
per-frame absolute delta and maximum episode-metric absolute delta were both `0.0`.

Conclusion: this compact classic-family adapter slice did not expose a non-identical AMMV/default
pair suitable for behavioral-difference annotation or AMMV trace-panel work. This is a scoped
diagnostic negative, not proof that AMMV-aware Social Force has no benefit in general.

## Limitation

The comparison covers the nested `algorithm_metadata.simulation_step_trace` fields present in the
ignored local JSONL rows and preserves only compact deltas and checksums in git. Treat this artifact
as a diagnostic parity screen for this compact adapter-mode matrix, not as a benchmark-strength or
general AMMV equivalence claim.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/summary.json
scripts/validation/check_docs_proof_consistency.py --path docs/context/issue_2434_ammv_scenario_sweep.md --path docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/README.md --path docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/summary.json
git diff --check
```
