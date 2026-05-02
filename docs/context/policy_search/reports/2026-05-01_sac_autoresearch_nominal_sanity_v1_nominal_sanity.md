# Candidate Report: sac_autoresearch_nominal_sanity_v1 (nominal_sanity)

## Decision

revise

## Hypothesis

A SAC policy trained directly on the nominal_sanity scenario surface with ego-frame SocNav observations may beat the best deterministic local policy-search success rate while preserving fail-closed SAC evaluation.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `sac`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/ai/autoresearch/trained_high_success/manual_nominal_sanity/summary.json`
- Git commit: `f7c7d44a2c52769f83f337d977ff7204e9f89268`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.3889 | 0.0556 | 5.6619 | 1.9885 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.5833 | 0.0000 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.3333 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `1`
- `static_collision`: `7`
- `timeout_low_progress`: `7`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | +0.1478 | +0.0000 |
| orca | -0.0177 | +0.3534 | -4.2774 |
| ppo | -0.0815 | +0.2896 | -3.4694 |
