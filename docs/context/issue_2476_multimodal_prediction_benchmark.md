# Issue #2476 Multimodal Prediction Benchmark Contract (2026-06-06)

Status: proposal, not benchmark evidence.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2476
- Parent roadmap issue: https://github.com/ll7/robot_sf_ll7/issues/2469
- Interface prerequisite: PR #2495 / issue #2475, merged probabilistic prediction interface
- Executable follow-up: https://github.com/ll7/robot_sf_ll7/issues/2496
- Proposal artifact: `configs/benchmarks/multimodal_prediction_contract_issue_2476.yaml`
- Policy boundary: `docs/context/issue_691_benchmark_fallback_policy.md`

## Goal

Define the benchmark contract needed to ask whether multiple pedestrian future hypotheses improve
Robot SF local planning compared with reactive and single-trajectory prediction baselines. This
note scopes the benchmark question and required evidence. It does not claim that multimodal
prediction improves planner performance.

## Resolved Dependency

PR #2495 introduced the minimal probabilistic prediction interface and has merged. The first
executable follow-up (#2496) can therefore run from `origin/main` and should prove that the
prediction contract can produce native reactive, single-trajectory, and multimodal rows before any
broader benchmark campaign is proposed.

## Benchmark Question

Can multiple future hypotheses for each pedestrian improve local planning?

The minimum useful comparison is:

| Comparator | Role | Interpretation |
| --- | --- | --- |
| `reactive_no_prediction` | Baseline | Uses current pedestrian state only. |
| `single_trajectory_prediction` | Baseline | Uses one deterministic future per pedestrian. |
| `multimodal_equal_weight` | Treatment | Uses multiple futures per pedestrian with equal or normalized weights. |
| `multimodal_confidence_weighted` | Treatment | Uses multiple futures and consumes confidence weights. |
| `oracle_future_upper_bound` | Diagnostic only | Measures headroom if included; not deployable evidence. |

The oracle row is optional and must be reported separately. It may support a headroom diagnosis, but
it must not be counted as a benchmark success row for deployable local planning.

## Scenario And Metric Contract

Start with a smoke-sized scenario surface that includes at least one occlusion-sensitive interaction
and one open crossing. Candidate reusable matrices are
`configs/scenarios/classic_interactions_francis2023.yaml` and `configs/scenarios/sanity_v1.yaml`.
Useful scenario families include blind corners, crossings, bottlenecks, and overtaking.

Primary metrics should include success rate, collision rate, minimum pedestrian distance, and time
to goal. Secondary metrics may include path efficiency, command smoothness or jerk, replan count,
and timeout rate.

The treatment is only promising if a multimodal row improves a safety metric without materially
worsening success rate or time to goal against both baselines. Exact thresholds should be fixed in
the executable follow-up once the smoke harness and metric scale are visible.

## Required Trace Fields

Each executable row should record enough prediction provenance to distinguish native multimodal
planning from fallback execution:

- planner key and prediction mode,
- prediction source,
- prediction horizon and `dt`,
- hypothesis count per pedestrian,
- confidence vector,
- selected or weighted hypothesis id when applicable,
- fallback or degraded reason,
- collision, success, timeout, minimum pedestrian distance, and time to goal.

If a row cannot satisfy the prediction interface, it should fail closed as `not_available` or
`failed`. Per `docs/context/issue_691_benchmark_fallback_policy.md`, fallback or degraded
prediction inputs are a caveat, not benchmark-strengthening evidence.

## Decision Rule

The first executable smoke passes only when all non-oracle comparator rows run natively and emit the
required trace fields. Promote to a broader benchmark only if the treatment rows improve at least
one safety metric without a material success or time-to-goal regression against both baselines.

Stop or revise if single-trajectory prediction and multimodal rows are indistinguishable on smoke
traces, or if multimodal rows require fallback or degraded prediction inputs.

## Recommended Follow-Up

Use the blocked follow-up issue after #2495 merges:

`benchmark: add executable multimodal prediction smoke` (#2496)

Minimum proof for that follow-up, now implemented as
`configs/benchmarks/multimodal_prediction_smoke_issue_2496.yaml` and
`scripts/validation/run_multimodal_prediction_contract_smoke.py`:

- a runner-visible smoke config or fixture,
- native reactive, single-trajectory, and multimodal rows,
- required trace fields emitted for every row,
- fail-closed behavior for missing or degraded prediction inputs.

## Validation For This Pass

This pass is docs/config only. Cheap validation is sufficient:

- inspect the diff,
- parse `configs/benchmarks/multimodal_prediction_contract_issue_2476.yaml` as YAML,
- run the context catalog proof check after adding this note to `docs/context/catalog.yaml`.
