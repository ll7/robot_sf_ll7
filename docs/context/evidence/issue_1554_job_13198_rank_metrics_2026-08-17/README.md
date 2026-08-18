# Issue #1554 Job 13198: rank-metric diagnostic

> Claim boundary: this is retained-input diagnostic evidence, not benchmark,
> ranking, dissertation, manuscript, or paper-facing evidence. The packet
> explicitly remains blocked for research admission pending #3216 rank-stability
> validation and source-level metric completeness.

## Question and provenance

The follow-up asked whether the retained Job 13198 outputs could support an
identity-preserving planner-rank diagnostic before any new #1554 benchmark
submission. Jobs 14483 and 14485 repaired producer encoding and planner identity
attribution; Job 14494 produced the final metric-coverage packet.

| Field | Value |
| --- | --- |
| Queue campaign | `issue1554_job13198_analysis_rank_metrics_20260817` |
| Source job | `13198` |
| Diagnostic job | `14494` |
| Public commit | `0acda10dc38f427009721c04cc9771f89f479366` |
| Config | `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml` |
| Preserved artifact | `wandb://ll7/robot_sf/campaign-issue1554_job13198_analysis_rank_metrics_20260817:v0` |
| Preserved manifest digest | `sha256:bc7cd0c2a29366f82a26694829679b2eab99b7090f4c9241edddbfd3ec8fb9d1` |

The private preservation receipt reports terminal scheduler state `COMPLETED`,
exit `0:0`, derived exit `0:0`, and verified artifacts. Raw rows and private
cluster/worktree paths remain in the durable private preservation system.

## Observed result

The final packet contains 16,143 primary episode metric rows and 25,920
adjacent-rank diagnostic rows. Planner identity is complete in the rank packet
for these nine observed keys:

`goal`, `hybrid_rule_v3_fast_progress_static_escape`, `orca`, `ppo`,
`prediction_planner`, `sacadrl`, `scenario_adaptive_hybrid_orca_v1`,
`social_force`, and `socnav_sampling`.

The identity repair is a data-quality result: the earlier retained-input
normalizer omitted `planner_key`/`algo`, so it emitted rank rows without reliable
planner attribution. The repaired packet retains those source identities and
keeps the repair separate from any behavioral interpretation.

Metric coverage remains deliberately fail-closed:

| Metric | Row-level numeric rows | Summary-level numeric rows | Status |
| --- | ---: | ---: | --- |
| `clearance` | 0 | 19 | summary-only |
| `low_progress` | 0 | 19 | summary-only |
| `min_distance` | 0 | 10 | summary-only |
| `timeout` | 0 | 0 | missing |

Therefore `rank_claim_status=diagnostic_only_pending_3216`,
`research_admission=blocked_pending_3216`, and the scientific outcome is
`inconclusive`. The packet does not establish planner superiority, rank
stability, success, collision, timeout, clearance, or benchmark performance.

## Interpretation and next proof

What this supports:

- planner identity can now be traced through the retained rank rows;
- the missing-identity defect was repaired without collecting new benchmark data;
- the evidence pipeline correctly refuses promotion while timeout is absent and
  several metrics exist only at summary level.

What this does not support:

- any planner ranking or adjacent-rank claim;
- a claim that one planner is better, safer, or more stable;
- a benchmark, dissertation, manuscript, or publication result;
- a decision to rerun the full #1554 campaign.

The next smallest proof is the existing #3216 rank-stability validation plus
source-level completion of the missing metric contract, especially timeout.
Until those gates pass, preserve this as diagnostic evidence and keep new
benchmark dispatch fail-closed.

## Reproduction boundary

The packet used retained Job 13198 inputs and the public commit/config listed
above. Reproduction requires the private queue and preservation contract. This
public note intentionally omits cluster names, partitions, local paths,
credentials, raw rows, and generated output trees.
