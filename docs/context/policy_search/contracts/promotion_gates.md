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