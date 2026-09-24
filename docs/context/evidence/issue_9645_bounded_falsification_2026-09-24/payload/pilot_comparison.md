## Adversarial sampler comparison (diagnostic tier)

> Claim boundary: diagnostic-only; not paper-facing benchmark evidence. Execution mode: `CPU-empirical`. A finite search budget cannot establish that no counterexample exists outside the evaluated rows or support a general method-superiority claim.

| objective | sampler | budget | seed | best_valid_objective | certified_valid_failures | replayable_valid_failures | replay_success_rate | invalid_candidate_rate | signed_property_violations | held_out_family_status | fallback/degraded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constraints_first_lexicographic_v1 | random | 16 | 1101 | 0.0000 | 0 | 0 | - | 0.000 | - | not_evaluated_narrow_archive | none |
| constraints_first_lexicographic_v1 | optuna | 16 | 1101 | 0.0000 | 0 | 0 | - | 0.000 | - | not_evaluated_narrow_archive | none |
| constraints_first_lexicographic_v1 | random | 16 | 2202 | 0.0000 | 0 | 0 | - | 0.000 | - | not_evaluated_narrow_archive | none |
| constraints_first_lexicographic_v1 | optuna | 16 | 2202 | 0.0000 | 0 | 0 | - | 0.000 | - | not_evaluated_narrow_archive | none |

### Stop-rule decision

**DIRECTION NARROWED (diagnostic).** Configured samplers were compared under matched CPU-empirical budgets with no degraded execution. The finite-budget result does not establish that no critical candidate exists outside the evaluated rows or support a general method-superiority claim.

### Exclusions and caveats

- learned failure proposal #2921: stretch/out of scope
- held-out-family yield: not evaluated (narrow archive caveat)
- paper-facing success claims: forbidden at this tier
- confirmation tier: artifact-level review of certification/replay/independent-seed
- report_status: diagnostic_local_nominal; schema adversarial-sampler-comparison.v3; budgets=[16]; seeds=[1101, 2202]
- source report: docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/pilot_comparison.json
