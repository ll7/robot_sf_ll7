# Issue #1554 Job 13198 Constraints-First Analysis

This packet analyzes completed Slurm job `13198` before any duplicate S20/H500
planner-family compute. It uses the retained public result summaries under
`output/issue1554-s20-h500-split-mem180-run/13198` plus private-ops job metadata
only for job provenance.

## Inputs

- Job: `13198` (`2026-06-issue1554-s20-h500-split-mem180-run`), Slurm status
  `completed`, evidence status `valid`.
- Config: `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml`.
- Public commit: `12a188de7246aad3b9088ea76e6a25a20029f976`.
- Episode rows: `8640` total, `9` planners, S20 per planner, 960 episodes per planner.
- Campaign warning: `SNQI contract status=fail with snqi_contract.enforcement=warn;
  campaign marked soft contract warning.`.

## Method

The constraints-first profile follows the canonical headline harness convention:
rank by `success` while requiring safety metrics (`collision`, `near_miss`) to be
present. Confidence intervals are 95% bootstrap intervals over per-seed means
(`2000` samples, seed `123`). Adjacent planner statements are `ci_separable`
only when adjacent confidence intervals do not overlap; overlapping intervals
are `not_statistically_distinguishable_budget`.

## Outputs

- `packet.json` records provenance, methodology, claim counts, the SNQI role, and the seed-budget decision.
- `constraints_first_metrics.csv` gives per-planner constraints-first metrics and confidence intervals.
- `adjacent_rank_claims.csv` lists constraints-first adjacent statements plus SNQI diagnostic adjacent statements.
- `claim_decision.md` states the decision boundary and seed-budget recommendation.

## Headline Answer

Constraints-first `ci_separable` adjacent statements: ppo over orca, orca over
prediction_planner, prediction_planner over socnav_sampling, socnav_sampling
over sacadrl, goal over social_force.

Constraints-first `not_statistically_distinguishable_budget` adjacent
statements: hybrid_rule_v3_fast_progress_static_escape over
scenario_adaptive_hybrid_orca_v1, scenario_adaptive_hybrid_orca_v1 over ppo,
sacadrl over goal.

SNQI changes the nominal order, but it does not change any decision in this
packet. It is explanatory only because `snqi_diagnostics.json` reports
`contract_status=fail` with enforcement `warn` and rank-alignment Spearman
`-0.19999999999999998`.
