## Issue #5785: Package B 27-Cell Diagnostic Table

> Claim scope: diagnostic summary only, not paper-facing benchmark evidence. The table renders fields from the committed report. Its `certified` and `replayable` labels are not independently reverified because the referenced raw candidate/replay bytes are unavailable. Issue #6131 owns that residual proof.

| objective | sampler | budget | seed | best_valid_objective | certified_valid_failures | replayable_valid_failures | replay_success_rate | invalid_candidate_rate | signed_property_violations | held_out_family_status | fallback/degraded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| worst_case_snqi | random | 16 | 1101 | 10.0000 | 1 | 1 | 1.0000 | 0.500 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | coordinate | 16 | 1101 | -1.0000 | 0 | 0 | - | 0.062 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | optuna | 16 | 1101 | -1.0000 | 0 | 0 | - | 0.562 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | random | 16 | 2202 | 10.0000 | 1 | 1 | 1.0000 | 0.312 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | coordinate | 16 | 2202 | -1.0000 | 0 | 0 | - | 0.062 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | optuna | 16 | 2202 | 23.0000 | 3 | 3 | 1.0000 | 0.500 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | random | 16 | 3303 | -1.0000 | 0 | 0 | - | 0.688 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | coordinate | 16 | 3303 | -1.0000 | 0 | 0 | - | 0.062 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | optuna | 16 | 3303 | 10.0000 | 3 | 3 | 1.0000 | 0.375 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | random | 32 | 1101 | 12.0000 | 2 | 2 | 1.0000 | 0.500 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | coordinate | 32 | 1101 | -1.0000 | 0 | 0 | - | 0.062 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | optuna | 32 | 1101 | -1.0000 | 0 | 0 | - | 0.688 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | random | 32 | 2202 | 30.0000 | 5 | 5 | 1.0000 | 0.375 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | coordinate | 32 | 2202 | -1.0000 | 0 | 0 | - | 0.062 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | optuna | 32 | 2202 | 23.0000 | 3 | 3 | 1.0000 | 0.688 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | random | 32 | 3303 | 10.0000 | 1 | 1 | 1.0000 | 0.656 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | coordinate | 32 | 3303 | -1.0000 | 0 | 0 | - | 0.062 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | optuna | 32 | 3303 | 10.0000 | 3 | 3 | 1.0000 | 0.688 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | random | 64 | 1101 | 12.0000 | 6 | 6 | 1.0000 | 0.453 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | coordinate | 64 | 1101 | -1.0000 | 0 | 0 | - | 0.062 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | optuna | 64 | 1101 | -1.0000 | 0 | 0 | - | 0.844 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | random | 64 | 2202 | 30.0000 | 6 | 6 | 1.0000 | 0.406 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | coordinate | 64 | 2202 | -1.0000 | 0 | 0 | - | 0.062 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | optuna | 64 | 2202 | 23.0000 | 3 | 3 | 1.0000 | 0.844 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | random | 64 | 3303 | 25.0000 | 2 | 2 | 1.0000 | 0.594 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | coordinate | 64 | 3303 | -1.0000 | 0 | 0 | - | 0.062 | - | not_evaluated_narrow_archive | none |
| worst_case_snqi | optuna | 64 | 3303 | 10.0000 | 3 | 3 | 1.0000 | 0.844 | - | not_evaluated_narrow_archive | none |

### Stop-rule decision

**NO PERFORMANCE DIRECTION CLAIMED.** The 27 rows and sampler totals are internally consistent with the committed report. This table does not establish replay validity, independent confirmation, a sampler ranking, or evidence for the signed-objective hypothesis.

### Exclusions and caveats

- learned failure proposal #2921: stretch/out of scope
- held-out-family yield: not evaluated (narrow archive caveat)
- paper-facing success claims: forbidden at this tier
- confirmation tier: artifact-level review of certification/replay/independent-seed
- report_status: diagnostic_local_nominal; schema adversarial-sampler-comparison.v3; budgets=[16, 32, 64]; seeds=[1101, 2202, 3303]
- source report: `report.json` in this evidence bundle
