# Issue #2432 AMMV Trace Selection Evidence

Date: 2026-06-06

This bundle records a diagnostic-only local trace-selection check for Issue #2432. It broadens the
Issue #2428/#2430 AMMV/default Social Force parity finding from one promoted 20-frame pair to the
three-seed Issue #2168 head-on-corridor slice with regenerated 100-frame step traces.

## Files

- `summary.json`: machine-readable result, commands, raw-output checksums, and candidate pair
  comparison records.
- `candidate_pair_comparison.csv`: flat seed-by-seed comparison table.

Raw regenerated JSONL outputs are not tracked. Their checksums are preserved in `summary.json`.

## Result

Seeds `111`, `112`, and `113` all produced 100-frame default/AMMV traces. Each pair had
`per_frame_max_abs_delta = 0.0` over robot state, pedestrian state, selected action, planner event,
and `ammv.pedestrian_force_vectors`.

Conclusion: no non-identical AMMV/default behavioral-difference pair was found in this local
adapter-mode head-on-corridor slice. Use a different scenario/family, direct planner-mode mechanism
trace source, or additional instrumentation before spending more annotation or panel work on AMMV
behavioral-difference evidence.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/summary.json
scripts/validation/check_docs_proof_consistency.py --path docs/context/issue_2432_ammv_trace_selection.md --path docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/README.md --path docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/summary.json
git diff --check
```
