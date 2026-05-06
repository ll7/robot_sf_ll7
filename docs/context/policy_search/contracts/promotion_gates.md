# Promotion Gates

Structured source of truth: `configs/policy_search/promotion_gates.yaml`.

## Nominal Sanity

- success rate at least `0.80`
- collision rate at most `0.02`

## Tier A

- success rate at least `0.30`
- collision rate at most `0.05`

## Tier B

- success rate at least `0.264`
- collision rate at most `0.055`

## Tier C

- success rate at least `0.248`
- collision rate at most `0.050`

## Scenario-Stratified Guard

- classic collision rate at most `0.07`
- Francis collision rate at most `0.05`

Classic and Francis success-rate floor values remain `null` in the machine-
readable gate file until the frozen baseline split is refreshed from a primary
artifact rather than approximated from the notes.

## Promotion CLI Guard

`scripts/tools/promote_policy_search_candidate.py` fails closed for local
stage summaries unless the stage runner recorded `decision: pass`. Promotion-
scale stages (`full_matrix`, `full_matrix_h500`, and `robustness_extension`)
are evaluated directly against the configured promotion gates because their
pass/fail decision is the promotion report itself.

The promotion decision also fails closed when the summary candidate is not
registered in `docs/context/policy_search/candidate_registry.yaml`, or when the
candidate's named promotion gate is absent from
`configs/policy_search/promotion_gates.yaml`.

Use `--gate-name <gate>` to apply a stricter diagnostic gate than the candidate
registry default. The h500 strict-gate reports use `--gate-name nominal_sanity`
so candidates above the experimental `tier_b` gate but slightly above the
`0.0200` collision ceiling are marked `revise` instead of promoted.
