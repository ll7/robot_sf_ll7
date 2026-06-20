# Closed-Loop Forecast Falsification Paper Plan - Issue #3193 (2026-06-20)

Issue: [#3193](https://github.com/ll7/robot_sf_ll7/issues/3193)
Parent lane: [#2835](https://github.com/ll7/robot_sf_ll7/issues/2835)
Experiment card:
[`experiments/issue_3193_closed_loop_forecast_falsification.yaml`](experiments/issue_3193_closed_loop_forecast_falsification.yaml)

Status: protocol-frozen synthesis plan. This note creates no benchmark result,
no planner-ranking claim, and no paper-facing empirical evidence.

## Frozen Claim Boundary

Central falsifiable claim:

> On the frozen Robot SF research-engine scenario suite, forecast methods that
> improve open-loop ADE/FDE by at least 10 percent over constant velocity do not
> improve same-seed closed-loop navigation outcomes beyond the no-forecast
> baseline unless they also reduce forecast-risk false-positive stopping without
> increasing collision, near-miss, timeout, or low-progress rates.

This is a falsification protocol, not a conclusion. The claim is tested only by
the ordered child issues below. Forecast metrics remain secondary evidence and
must not be promoted to navigation-success evidence without same-seed closed-loop
support.

## Hypotheses

H0: Open-loop forecast improvement does not transfer to same-seed closed-loop
navigation improvement. Under H0, forecast variants either match the no-forecast
baseline, worsen progress/safety, or increase false-positive stopping enough to
offset any risk reduction.

H1: At least one non-oracle forecast variant improves same-seed closed-loop
navigation relative to no-forecast by meeting every primary decision threshold
below while preserving fail-closed row-status accounting.

## Protocol Freeze

Protocol frozen: 2026-06-20.

Frozen endpoints, thresholds, scenario families, seed ladder, variants, and
figure/table mappings may not change after result inspection. Any change requires
a dated amendment in this note, a linked issue comment on Issue #3193, and an
updated experiment card before rerunning affected campaigns.

Non-goals:

- no ADE/FDE-only navigation claim;
- no oracle-observation transferable claim;
- no real-world pedestrian realism claim;
- no default planner-risk behavior change before false-positive stopping is
  bounded;
- no learned-predictor launch until Issue #2916 and Issue #2966 satisfy the gate.

## Design Matrix

Scenario families are frozen from
[`configs/benchmarks/issue_3059_research_engine_suite_v0.yaml`](../../configs/benchmarks/issue_3059_research_engine_suite_v0.yaml):

| Family | Role in the paper plan |
| --- | --- |
| `frame_consistency_sanity` | Sanity gate; excludes social forecast claims. |
| `static_obstacle_detour` | Static-control baseline for non-pedestrian confounds. |
| `topology_and_local_minima` | Route-progress and timeout sensitivity. |
| `paired_pedestrian_interactions` | Primary simple social-interaction slice. |
| `crowd_flow_and_density` | Density and pedestrian-flow stress slice. |
| `social_protocol_francis` | Social-protocol diagnostic slice; remains diagnostic unless row status proves eligibility. |

Forecast variants:

- `none`;
- `cv`;
- `semantic`;
- `interaction_aware`;
- `risk_filtered`.

Seed ladder:

- S5: seeds `111` through `115`; first smoke/diagnostic pass.
- S10: seeds `111` through `120`; escalate only when S5 rows are eligible and
  uncertainty remains decision-relevant.
- S20: seeds `111` through `130`; escalate only when S10 uncertainty remains
  decision-relevant and endpoints remain frozen.

The seed ladder is governed by `scripts/tools/seed_sufficiency_gate.py`. Rows
with fallback, degraded, unavailable, failed, denominator-invalid, or
missing-provenance status remain visible exclusions rather than success evidence.

## Primary Closed-Loop Metrics

Primary metrics are evaluated for same scenario, seed, and planner-consumer
condition across variants:

| Metric | Unit | Direction | Threshold for H1 support |
| --- | --- | --- | --- |
| `success_rate` | proportion | higher is better | no decrease vs `none`; S10/S20 lower confidence bound must be non-negative for delta |
| `collision_rate` | proportion | lower is better | no increase vs `none`; any increased collision rate forces `revise` or `stop` |
| `near_miss_rate` | proportion | lower is better | no increase vs `none`; increased near misses without success gain forces `revise` |
| `min_distance_m` | meters | higher is better | median delta must be non-negative when safety-rate deltas are inconclusive |
| `low_progress_or_timeout_rate` | proportion | lower is better | must not increase vs `none`; decrease can support H1 only with no safety regression |
| `false_positive_stop_rate` | proportion of forecast-risk stop events not followed by collision/near-miss avoidance within the risk horizon | lower is better | must decrease vs `cv`/`semantic` risk arms and must not exceed `none` by more than 0.02 absolute |
| `runtime_ms_p95` | milliseconds | lower is better | p95 runtime must remain within 1.25x `none` or be classified as non-deployable diagnostic evidence |

Decision thresholds are intentionally conservative because this protocol is
paper-facing if executed later. If the seed-sufficiency gate reports insufficient
power, the decision is `escalate_to_s10`, `escalate_to_s20`, or
`diagnostic_only`, not H1 support.

## Secondary Forecast Metrics

Forecast metrics are reported separately from the navigation-success claim:

- ADE and FDE at the frozen horizon ladder;
- miss rate;
- calibration/reliability where probabilistic outputs exist;
- collision-relevant forecast error;
- planner-relevant risk error.

Open-loop forecast improvement is a necessary context variable, not a sufficient
paper claim. Any table that reports ADE/FDE must include a visible note that
closed-loop metrics are the primary endpoint.

## Decision Rules

Use the first rule that applies:

1. `stop`: any forecast variant increases collision rate, materially increases
   near-miss rate, or repeatedly creates false-positive stopping with no safety
   benefit under eligible same-seed rows.
2. `revise`: forecast variants show open-loop gains but closed-loop outcomes are
   mixed, degraded, denominator-invalid, or dominated by false-positive stopping.
3. `continue`: at least one non-oracle forecast variant meets all primary H1
   thresholds through the seed-sufficiency gate and has no fallback/degraded
   rows counted as success.
4. `diagnostic_only`: execution is limited to smoke fixtures, missing row-status
   provenance, or unsupported scenario diversity.
5. `blocked`: required eligible rows, durable artifacts, or planner-consumed
   forecast variants are absent.

## Figure And Table Plan

Every figure/table is mapped to a child issue and a result-store query template.
The canonical row source is the campaign result store from Issue #3076, with
episode rows conforming to `scripts/tools/campaign_result_store.py`.

| Planned output | Child issue | Query template |
| --- | --- | --- |
| Table 1: frozen protocol matrix | Issue #3193 | Read this note plus the experiment card; no result-store rows. |
| Table 2: open-loop forecast baseline ladder | Issue #2915 | `SELECT scenario_family, forecast_variant, COUNT(*) AS rows, AVG(ade_m) AS ade_m, AVG(fde_m) AS fde_m, AVG(miss_rate) AS miss_rate FROM forecast_result_store WHERE row_status IN ('native','adapter') GROUP BY scenario_family, forecast_variant` |
| Table 3: risk-eligibility denominator repair | Issue #2904 | `SELECT actor_class, observation_tier, COUNT(*) AS rows, SUM(risk_scoring_eligible) AS eligible_rows FROM forecast_eligibility_store GROUP BY actor_class, observation_tier` |
| Figure 1: open-loop gain vs closed-loop delta | Issue #2916 | Join Issue #2915 forecast rows to closed-loop rows by `scenario_id`, `seed`, and `forecast_variant`; plot ADE/FDE delta against success, collision, near-miss, and progress deltas vs `none`. |
| Figure 2: false-positive stopping by variant | Issue #2916 | `SELECT scenario_family, forecast_variant, AVG(false_positive_stop_rate) AS fp_stop_rate, AVG(collision_rate) AS collision_rate, AVG(near_miss_rate) AS near_miss_rate FROM closed_loop_result_store GROUP BY scenario_family, forecast_variant` |
| Table 4: planner-consumed forecast slice | Issue #2966 | `SELECT scenario_family, scenario_id, seed, forecast_variant, row_status, success, collision, near_miss, progress_m, false_positive_stops FROM campaign_episodes WHERE planner = 'PredictionPlannerAdapter'` |
| Figure 3: seed-sufficiency escalation | Issue #2916 and Issue #2966 | `scripts/tools/seed_sufficiency_gate.py` output over S5/S10/S20 deltas for primary closed-loop metrics. |
| Table 5: final continue/revise/stop decision | Issue #2916 and Issue #2966 | Aggregate child issue decision summaries; require row-status counts and uncertainty caveats before classification. |
| Appendix Table A: learned predictor gate | Issue #2844 and Issue #2845 | Only populated after Issue #2916 and Issue #2966 return `continue`; otherwise record `stopped_by_gate` with no training results. |

## Child Issue Sequence

1. Issue #2915: compare CV, semantic-CV, and interaction-aware baselines before any
   learned predictor expansion.
2. Issue #2904: repair forecast-risk eligibility denominators for actor class and
   observation tier.
3. Issue #2916: run the same-seed closed-loop forecast-risk coupling gate.
4. Issue #2966: run the planner-consumed forecast slice with `none`, `cv`,
   `semantic`, `interaction_aware`, and `risk_filtered`.
5. Issue #2844: scope or implement a lightweight learned probabilistic predictor only
   if Issue #2916 and Issue #2966 return `continue`.
6. Issue #2845: assess heavier transformer/diffusion-style predictors only after
   Issue #2844 or a recorded stop/revise decision.

Missing child issue: none required for the protocol freeze. If Issue #2915 or
Issue #2916 cannot emit the query fields above, open a narrow schema/adapter
follow-up instead of changing this plan after results are visible.

## Evidence Grade Ladder

- `idea`: this protocol note and experiment card.
- `protocol_frozen`: endpoints and thresholds fixed before heavy execution.
- `implementation_ready`: child issues have concrete configs and no unresolved
  placeholders.
- `preflight_passed`: each child campaign has validated configs, result-store
  paths, and fail-closed row-status checks.
- `finalized`: child result stores exist but claims are not yet promoted.
- `claim_reviewed`: row-status exclusions, uncertainty, and non-claims have been
  reviewed against the evidence policy.
- `paper-grade`: only after durable artifacts, same-seed eligible rows,
  seed-sufficiency gates, and claim review all pass.

## Validation For This Plan

This PR should run only cheap synthesis validation:

```bash
uv run python scripts/tools/validate_experiment_registry.py
uv run python scripts/tools/sync_ai_config.py --check
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```

Heavy experiments are explicitly out of scope for Issue #3193.
