## Issue #5326 durable objective-comparison table (diagnostic tier)

> Claim scope: not paper-facing benchmark evidence. The `--synthetic` CPU path is reproducible by construction; the `--empirical` CPU path runs the real `pysocialforce` evaluator and produces certified/replayable failures without Slurm/GPU. Matched-budget confirmation at paper tier still requires artifact-level review of certification/replay/independent-seed evidence.

| objective | sampler | budget | seed | best_valid_objective | certified_valid_failures | replayable_valid_failures | replay_success_rate | invalid_candidate_rate | signed_property_violations | held_out_family_status | fallback/degraded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constraints_first_lexicographic_v1 | random | 16 | 1101 | 0.0000 | 0 | 0 | - | 0.000 | - | not_evaluated_narrow_archive | none |
| constraints_first_lexicographic_v1 | optuna | 16 | 1101 | 0.0000 | 0 | 0 | - | 0.000 | - | not_evaluated_narrow_archive | none |
| constraints_first_lexicographic_v1 | random | 16 | 2202 | 0.0000 | 0 | 0 | - | 0.000 | - | not_evaluated_narrow_archive | none |
| constraints_first_lexicographic_v1 | optuna | 16 | 2202 | 0.0000 | 0 | 0 | - | 0.000 | - | not_evaluated_narrow_archive | none |

### Stop-rule decision

**DIRECTION NARROWED (diagnostic).** Both objectives compared under matched CPU-synthetic budgets with no degraded execution. This is a contract/structure check only; it does not constitute benchmark evidence for the signed-objective hypothesis (requires artifact-level confirmation of certification/replay/independent-seed evidence).

### Exclusions and caveats

- learned failure proposal #2921: stretch/out of scope
- held-out-family yield: not evaluated (narrow archive caveat)
- paper-facing success claims: forbidden at this tier
- confirmation tier: artifact-level review of certification/replay/independent-seed
- report_status: diagnostic_local_nominal; schema adversarial-sampler-comparison.v3; budgets=[16]; seeds=[1101, 2202]
- source report: output/issue9645-pilot/comparison.json
