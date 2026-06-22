# Research-Engine Month-One Synthesis (2026-06)

> Status: current — dated synthesis snapshot (2026-06-22). This is a
> **synthesis/index only**: it records the landed-vs-pending state of the
> research-engine roadmap and links each child's declared evidence. It makes
> **no new benchmark or paper-facing claim**; cite the linked issue/note
> evidence, not this surface, for any result.

Companion to the [Researcher's Guide](../researchers_guide.md). Tracks epic
[#3057](https://github.com/ll7/robot_sf_ll7/issues/3057) — "build Robot SF
continuous social-navigation research engine."

## Landed children

The platform/contract layer of the roadmap has landed. These children are
CLOSED/COMPLETED; the linked notes carry their declared evidence boundaries.

| Child | What landed | Declared tier | Durable pointer |
|---|---|---|---|
| [#3058](https://github.com/ll7/robot_sf_ll7/issues/3058) | research-engine gap audit vs existing surfaces | diagnostic/synthesis | [`issue_3058_research_engine_gap_audit.md`](issue_3058_research_engine_gap_audit.md) |
| [#3059](https://github.com/ll7/robot_sf_ll7/issues/3059) | frozen research-engine scenario suite v0 | proposal/contract | [`issue_3059_research_engine_suite_v0.md`](issue_3059_research_engine_suite_v0.md) |
| [#3060](https://github.com/ll7/robot_sf_ll7/issues/3060) | baseline planner readiness matrix | diagnostic/contract | [#3060](https://github.com/ll7/robot_sf_ll7/issues/3060) |
| [#3061](https://github.com/ll7/robot_sf_ll7/issues/3061) | influence + social-compliance metric contract | diagnostic/contract | [`issue_3061_social_compliance_metric_contract.md`](issue_3061_social_compliance_metric_contract.md) |
| [#3062](https://github.com/ll7/robot_sf_ll7/issues/3062) | standardized campaign manifest + artifact flow | contract | [`issue_3062_campaign_manifest_flow.md`](issue_3062_campaign_manifest_flow.md) |
| [#3063](https://github.com/ll7/robot_sf_ll7/issues/3063) | automated campaign comparison reports | analysis-only | [`issue_3063_campaign_comparison_report.md`](issue_3063_campaign_comparison_report.md) |
| [#3064](https://github.com/ll7/robot_sf_ll7/issues/3064) | pedestrian behavior-model variant validation | diagnostic/preflight | [`issue_3064_behavior_variants_inventory.md`](issue_3064_behavior_variants_inventory.md) |
| [#3069](https://github.com/ll7/robot_sf_ll7/issues/3069) | multi-robot research smoke path | smoke/diagnostic-only | [`issue_3069_smoke.yaml`](../../configs/multi_robot/issue_3069_smoke.yaml), [`run_multi_robot_smoke_issue_3069.py`](../../scripts/validation/run_multi_robot_smoke_issue_3069.py) |
| [#3070](https://github.com/ll7/robot_sf_ll7/issues/3070) | fairness + heterogeneous-pedestrian disruption metric scoping | proposal | [`issue_3184_distributional_disruption_metrics.md`](issue_3184_distributional_disruption_metrics.md) |

Taken together these establish the **result-store contract**
([`issue_3076_campaign_result_store_contract.md`](issue_3076_campaign_result_store_contract.md))
and the evidence plumbing that empirical campaigns build on. None of these are
benchmark or paper-facing results on their own — they are the scaffolding that
makes such results gradable.

## Pending / blocked children

The empirical-run and resource-gated layer is still open.

| Child | Lane | Status | Blocker / gate |
|---|---|---|---|
| [#3065](https://github.com/ll7/robot_sf_ll7/issues/3065) | real-trajectory ingestion + staging contract | blocked | external data (`resource:external-data`) |
| [#3066](https://github.com/ll7/robot_sf_ll7/issues/3066) | robot influence on pedestrian-flow v0 | ready (local) | empirical run + evidence grading |
| [#3067](https://github.com/ll7/robot_sf_ll7/issues/3067) | sensor-noise / partial-observability robustness slice | ready (local) | empirical run + evidence grading |
| [#3068](https://github.com/ll7/robot_sf_ll7/issues/3068) | curriculum-learning launch packet (PPO) | ready | compute (`resource:slurm`) |

## Reading this synthesis

- **What is proven now:** the contracts, suite freeze, metric definitions,
  manifest/artifact flow, and reporting machinery exist and are tested. The
  multi-robot path runs as a diagnostic-only smoke.
- **What is not proven:** no headline planner-comparison, robustness, or
  influence result is claimed here. Those depend on the pending empirical
  children and must carry their own predeclared evidence.
- **How to extend:** follow the [Researcher's Guide](../researchers_guide.md)
  steps 1–6 and add a new row here (with the child's declared tier and a durable
  pointer) when a lane lands.
