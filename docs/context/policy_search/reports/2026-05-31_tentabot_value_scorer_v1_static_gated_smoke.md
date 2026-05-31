# Candidate Report: tentabot_value_scorer_v1_static_gated (smoke)

## Decision

pass

## Hypothesis

A clean-room Tentabot-style primitive value scorer can preserve the v0 candidate lattice and auditable value terms while demoting low-clearance commands unless they make positive progress without worsening static clearance. The static gate is an exploratory Robot SF safety tier, not upstream Tentabot parity or benchmark evidence.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: local ignored output at
  `output/policy_search/tentabot_value_scorer_v1_static_gated/smoke/latest/summary.json`
- Git commit recorded by the local run: `629c8402560999df3138e802b5e803f25bae5992`
  plus the uncommitted branch diff that introduced this candidate. Use the tracked tables below as
  compact diagnostic evidence, not benchmark-grade reproducibility provenance.

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.8193 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline
deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior,
generalization, or planner superiority. The raw summary JSON stayed in ignored local output; the
tracked tables in this file are the durable compact evidence for this smoke result.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
