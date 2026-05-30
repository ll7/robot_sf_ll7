# Issue #1625 Learned Planner Arbitration Assessment

Date: 2026-05-30

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1625>

## Scope

This note assesses whether Robot SF should train a learned arbitration policy that selects among
existing planners instead of replacing the local planner with an end-to-end learned controller. It
does not implement an arbiter, train a model, run a benchmark, or change benchmark metrics.

The short answer is: learned arbitration is a promising hybrid-learning direction, but the next
step should be a data-contract and labeling packet, not a runtime implementation. The repository
already has a non-learning portfolio surface (`policy_stack_v1`) and static scenario selectors, but
it does not yet have durable per-step or per-window labels that prove a learned arbiter can be
trained without hindsight leakage.

## Existing Surfaces

| Surface | Current role | Relevance to arbitration |
| --- | --- | --- |
| `robot_sf/planner/policy_stack_v1.py` | Scores `goal` and `risk_dwa` proposals, records selected proposal, proposal statuses, score components, and hard-shield interventions. | Best runtime host for a future arbiter because it already separates proposal collection, scoring, rejection, fallback/degraded/not-available status, and diagnostics. |
| `robot_sf/planner/hybrid_portfolio.py` | Switches between risk-DWA, ORCA, prediction, and optional MPPI heads with local-risk thresholds and hysteresis. | Best prior example of runtime planner-head arbitration; useful as a behavior baseline, but less clean than `policy_stack_v1` for learned proposal/label contracts. |
| `docs/context/issue_932_hybrid_portfolio_diagnostics.md` | Documents `selected_head_counts`, `fallback_count`, `last_decision`, `active_head`, and `hold_remaining`. | Defines existing decision telemetry a trace packet can reuse or supersede with a versioned arbitration schema. |
| `configs/algos/policy_stack_v1.yaml` | Config-first portfolio runtime with explicit proposal sources and hard-stop clearance. | Minimal smoke target for any future learned scorer or selector fixture. |
| `docs/context/issue_926_policy_stack_v1_contract.md` | Contract for common proposal payloads, status semantics, risk scoring, shield decisions, and diagnostics. | Defines the fail-closed boundary a learned arbiter should inherit. |
| `docs/context/issue_1004_policy_stack_v1_runtime.md` | Records the first runtime slice and its caveats. | Shows the stack runs through map-runner but does not yet prove planner quality. |
| `configs/policy_search/candidates/planner_selector_v1.yaml` | Static hybrid-portfolio candidate that combines tuned planner parameters. | Useful baseline for scenario-aware selection, but it is not a learned dynamic arbiter. |
| `docs/context/policy_search/reports/2026-04-29_planner_selector_v1_smoke.md` | Smoke report for `planner_selector_v1`; 3/3 episodes timed out with low progress and no collisions. | Shows that a selector can run end-to-end while still failing progress, so runtime availability is not enough. |
| `docs/context/issue_673_hybrid_portfolio_benchmark.md` | Hybrid portfolio paper-surface comparison kept as testing-only after low success, high collision, and high runtime. | Negative evidence against assuming planner-head switching is automatically safer or stronger. |
| `docs/context/issue_1023_experimental_benchmark_candidates.md` | Records scenario-adaptive hybrid ORCA evidence and overfitting caveats. | Shows selector-style rows can be valuable, but benchmark-distribution knowledge must be treated cautiously. |
| `docs/context/policy_search/experiment_ledger.md` | Compact candidate, stage, success, collision, and decision history. | Useful for episode-level labels; too coarse for safe per-step learned switching. |
| `docs/context/issue_1618_learned_policy_adapter_interface.md` | Learned-policy observation, action, provenance, and fallback contract. | Required metadata boundary if the arbiter itself becomes a learned local policy. |
| `docs/context/issue_1624_hybrid_learning_architecture.md` | Hybrid-learning architecture scaffold. | Places arbitration as a possible future arbiter over residual, risk, imitation, and classical proposal branches. |

## Candidate Arbitration Formulations

| Formulation | Action space | Best available labels | Main risk | Recommendation |
| --- | --- | --- | --- | --- |
| Episode-level planner choice | Select one planner for the whole episode. | Existing policy-search summaries and episode outcomes. | Encodes scenario distribution and may overfit to benchmark IDs. | Good first supervised baseline for analysis only. |
| Scenario-family planner choice | Select planner from declared scenario metadata before rollout. | Scenario-adaptive candidate evidence. | Can become a hand-coded benchmark selector rather than a general policy. | Keep as non-learning comparator, not the first learned target. |
| Per-window planner choice | Select planner every fixed window, such as 1-2 seconds. | Would need synchronized multi-planner rollouts or counterfactual traces. | Labels are unavailable today without costly counterfactual execution. | Promising after trace packet exists. |
| Per-step planner choice | Select planner every decision step. | Not currently durable; would require dense counterfactuals. | Unsafe oscillation, action discontinuity, and hindsight leakage. | Reject as first implementation target. |
| Risk-triggered override | Use learned risk to switch from progress planner to safer planner. | Learned-risk launch packet plus future hard-guard diagnostics. | Risk model may become a hidden guard replacement. | Consider only as auxiliary scorer under hard guard. |
| Residual enable/disable | Decide whether a bounded ORCA residual is active. | Future ORCA-residual diagnostics. | Requires trained residual evidence first. | Defer until Issue #1475 produces data. |
| Guard mode selection | Select guard strictness or fallback source. | Guard veto/fallback diagnostics. | Learning guard relaxation is high risk. | Reject for now except offline analysis of fixed guard modes. |

## Observation Features

A deployable arbiter should use only inference-available features:

- robot pose, heading, velocity, route/subgoal progress, and previous command,
- local pedestrian state, clearance, density, relative velocity, and time-to-collision estimates,
- static-map clearance or occupancy/risk-surface features already available to the planner,
- proposal-level diagnostics from candidate planners, such as command, score components,
  availability status, rejection reason, and guard veto status,
- bounded history of recent progress, command changes, and intervention counts.

Forbidden arbiter inputs:

- final episode outcome, future collision/near-miss labels, future route completion, or future
  oracle trajectories at deployment time,
- benchmark scenario ID as an unconstrained feature for a learned model,
- fallback/degraded planner outputs labeled as successful planner choices,
- guard-disabled counterfactuals presented as deployable behavior.

## Label And Reward Sources

Current durable sources can support only coarse analysis:

- `docs/context/policy_search/experiment_ledger.md` gives stage-level success, collision,
  near-miss, and decision status for candidates.
- Benchmark episode records contain scenario, seed, metrics, termination reason, outcome, integrity,
  algorithm metadata, and extensible diagnostic fields.
- `policy_stack_v1` diagnostics already expose selected proposal, proposal status counts, score
  components, rejection reasons, and shield interventions.
- `hybrid_portfolio` diagnostics expose selected head counts, fallback counts, hysteresis state, and
  the last desired/selected head decision.

Missing for a learned arbiter:

- synchronized same-state candidate commands from multiple planners,
- per-window labels that distinguish progress improvement from unsafe hindsight,
- switch-rate and dwell-time diagnostics,
- counterfactual safety labels for candidate commands that were not executed,
- durable trace manifests with checksums and train/validation/evaluation split policy.

Existing negative evidence matters: `planner_selector_v1` smoke ran end-to-end but made no progress,
and the earlier `hybrid_portfolio` paper-surface comparison was kept testing-only after lower
success, higher collision, and higher runtime than predictive baselines. A learned arbiter should
therefore be framed as a falsifiable data question, not as an obvious promotion path.

## First Smoke Criteria If Pursued Later

Do not start with a trainable model. The first implementation should be a deterministic
`policy_stack_v1` selector fixture that exercises the data path and diagnostics:

- candidate sources: `goal`, `risk_dwa`, and one explicitly available classical or adapter planner
  from the `hybrid_portfolio`/ORCA/prediction family,
- selector action: episode-level or fixed-window source choice, not per-step unrestricted switching,
- switch constraints: dwell time at least 1 second, bounded command delta, and hard guard active,
- diagnostics: selected source, previous source, switch reason, proposal statuses, guard veto,
  fallback/degraded/not-available status, and command discontinuity,
- validation: `planner_sanity_simple` smoke plus one topology-heavy atomic scenario,
- success boundary: smoke proves instrumentation only; it is not evidence that learned arbitration
  improves navigation.

## Recommendation

`collect_data_first`

Learned arbitration should not be implemented as a trainable planner yet. The repository should
first define an arbitration trace packet that can be produced by `policy_stack_v1` without changing
benchmark metrics. That packet should declare:

- proposal sources and command/action contracts,
- inference-available observation features,
- fixed switching cadence and dwell constraints,
- exact label generation policy,
- train/validation/evaluation split rules,
- leakage exclusions,
- fallback/degraded/not-available handling,
- compact smoke command and expected diagnostics.

Once that packet exists and a deterministic selector fixture proves the trace shape, a later issue
can decide between supervised best-planner imitation, contextual bandits, or offline RL. Until then,
episode-level planner choice is the only label family mature enough for analysis, and it is too
coarse to justify a learned runtime arbiter.

## Follow-Up Boundary

No implementation issue is opened from this assessment. A future follow-up should be created only
when it can name a concrete trace packet and smoke command, for example:

`workflow: define policy_stack_v1 arbitration trace packet`

That follow-up should remain docs/tooling-first and should not train a model. A separate learned
arbiter implementation issue should wait until the trace packet has produced durable, leakage-safe
training/evaluation data.

## Validation

This is a documentation-only assessment. Validate with:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
