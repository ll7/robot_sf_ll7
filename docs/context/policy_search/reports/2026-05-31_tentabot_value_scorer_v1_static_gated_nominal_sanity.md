# Candidate Report: tentabot_value_scorer_v1_static_gated (nominal_sanity)

## Decision

revise

## Hypothesis

A clean-room Tentabot-style primitive value scorer can preserve the v0 candidate lattice and auditable value terms while demoting low-clearance commands unless they make positive progress without worsening static clearance. The static gate is an exploratory Robot SF safety tier, not upstream Tentabot parity or benchmark evidence.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: local ignored output at
  `output/policy_search/tentabot_value_scorer_v1_static_gated/nominal_sanity/latest/summary.json`
- Git commit recorded by the local run: `629c8402560999df3138e802b5e803f25bae5992`
  plus the uncommitted branch diff that introduced this candidate. Use the tracked tables below as
  compact diagnostic evidence, not benchmark-grade reproducibility provenance.

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2222 | 0.1111 | 0.2222 | 4.3436 | 1.6152 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0833 | 0.1667 | 0.3333 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `3`
- `static_collision`: `2`
- `timeout_low_progress`: `9`

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline
deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior,
generalization, or planner superiority. The raw summary JSON stayed in ignored local output; the
tracked tables in this file are the durable compact evidence for this revise/stop boundary.

Compared with the same-run `tentabot_value_scorer_v0` nominal-sanity baseline, this static-gated
variant kept success flat at `4/18`, reduced low-progress timeouts from `11` to `9`, and worsened
static collisions from `1` to `2` plus near-miss episodes from `3` to `4`. Treat this as negative
diagnostic evidence for stronger hand-tuned static penalties, not an improvement claim.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2080 | -0.1300 | n/a |
| orca | +0.0378 | +0.0756 | n/a |
| ppo | -0.0260 | +0.0118 | n/a |
