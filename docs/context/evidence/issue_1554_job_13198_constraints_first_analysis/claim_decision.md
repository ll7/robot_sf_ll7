# Claim Decision

## Decision

Job `13198` is valid S20 constraints-first adjacent-rank evidence for the statements below, but it does not support a blanket paper-grade strict ordering. The campaign has nine successful planner rows and 8,640 episode rows, with a soft SNQI contract warning.

## Planner Statements

`ci_separable` under constraints-first success ranking:

- ppo over orca
- orca over prediction_planner
- prediction_planner over socnav_sampling
- socnav_sampling over sacadrl
- goal over social_force

`not_statistically_distinguishable_budget` under constraints-first success ranking:

- hybrid_rule_v3_fast_progress_static_escape over scenario_adaptive_hybrid_orca_v1
- scenario_adaptive_hybrid_orca_v1 over ppo
- sacadrl over goal

`diagnostic_only` statements:

- All SNQI adjacent-rank statements in `adjacent_rank_claims.csv`; SNQI contract status is `fail` with enforcement `warn`.
- Any statement that uses SNQI to reorder planners is explanatory only, not decision-changing.

## SNQI Role

SNQI changes the nominal order for 7 planners, but it does not change the constraints-first decision. The SNQI contract failed (`rank_alignment_spearman=-0.19999999999999998`), so SNQI can explain component sensitivity and disagreement but cannot promote or reverse planner-ranking claims.

## More Seed-Budget Compute

More seed-budget compute is conditionally justified only if the manuscript needs strict adjacent ordering for: hybrid_rule_v3_fast_progress_static_escape over scenario_adaptive_hybrid_orca_v1, scenario_adaptive_hybrid_orca_v1 over ppo, sacadrl over goal. It is not justified for the already `ci_separable` adjacent statements, and more SNQI compute alone is not justified until the SNQI contract issue is resolved.

## Evidence Grade

- Benchmark row status: `successful_evidence` for all nine planner rows (`status=ok`, `benchmark_success=true`).
- Claim grade: S20 constraints-first adjacent-rank evidence with explicit CI downgrade boundaries.
- Paper-facing boundary: do not state a strict total planner ordering; state only the separable adjacent statements and budget-limited caveats above.
