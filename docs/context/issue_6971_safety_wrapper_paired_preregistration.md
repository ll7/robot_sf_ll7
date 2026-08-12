# Issue #6971 — Safety-wrapper paired-campaign preregistration

Plain-language summary: this document freezes the experiment that can answer whether the
planner-agnostic safety wrapper measurably reduces exact collisions, and records its expected
cost. It is a design contract, not a campaign result.

## Claim boundary

The preregistration is diagnostic planning evidence only. It does not run compute, establish a
safety gain, certify deployment safety, generalize beyond the configured software and 48-scenario
suite, or change paper/dissertation claims. A future campaign is blocked unless every native row,
paired key, retained metric, and provenance field passes the fail-closed contracts.

The machine-readable source of truth is
[`configs/benchmarks/issue_6971_safety_wrapper_paired_preregistration.yaml`](../../configs/benchmarks/issue_6971_safety_wrapper_paired_preregistration.yaml).
The historical #4830 campaign snapshot remains non-identifiable for the normalized paired metrics;
this design does not reinterpret that snapshot:
[`issue_4830_safety_wrapper_factorial_v1`](evidence/issue_4830_safety_wrapper_factorial_v1/README.md).

## Frozen question and design

The primary estimand is the within-planner, within-scenario, within-seed difference

`wrapper_on exact_collision_probability - wrapper_off exact_collision_probability`.

Negative values mean fewer episode-level exact collisions with the fixed wrapper enabled. The
three planners are ORCA, Social Force, and Prediction Planner. The suite resolves to 48 scenarios
under a pinned manifest digest and resolved scenario-ID digest, the paired seed list is exactly
`111`, `112`, `113`, and the pairing key is
`(planner, scenario_id, seed)`. This produces 144 paired keys per planner, 432 paired keys total,
and 864 retained arm rows.

The wrapper-on thresholds are fixed at a 2.0 m caution radius, 0.5 m/s capped speed, 1.0 s
time-to-collision veto, and 0.3 m clearance veto. There is no per-planner tuning. The wrapper-off
arm is the paired counterfactual and retains an explicit arm identity.

## Outcomes and retained fields

There is one primary outcome: `metric_values.exact_collision_probability`, an episode-level
Bernoulli probability in `[0, 1]`. Secondary safety outcomes are near-miss probability, minimum
predicted separation, false-positive stop rate, stop/yield latency, and wrapper intervention rate.
Completion probability and timeout progress are preregistered task-performance cost measures.
All outcomes are reported, including no-gain outcomes; a secondary cannot become primary after
rows are inspected.

Every row must retain all eight fields from
[`paired_effect_metric_contract.v1`](../../configs/benchmarks/paired_effect_metric_contract_v1.yaml)
under `metric_values`: exact collision probability, near-miss probability, minimum predicted
separation, completion probability, timeout progress, false-positive stop rate, stop/yield latency,
and wrapper intervention rate. Missing, nonfinite, boolean-typed, out-of-range, or semantically
substituted values fail closed. Legacy fields such as clearing distance are not aliases.

## Analysis and decision rules

The estimator is wrapper-on minus wrapper-off for each complete paired key, followed by equal-weight
planner-stratum means and an equal-weight pooled three-planner summary. A deterministic 2,000-sample
95% bootstrap resamples complete scenario-seed blocks within each planner, preserving the off/on
pair. Binary ties remain zero deltas. Missing or failed rows are not imputed, deleted, or replaced
with new seeds.

The smallest effect worth detecting is a 0.05 absolute reduction in exact-collision probability.
The precision target is a pooled primary interval width at most 0.10 (half-width at most 0.05),
not a significance threshold. If the interval is wider, the result is inconclusive and the seed
budget cannot be changed post hoc.

- Measured safety gain on this fixed suite: the pooled primary interval upper bound is at most
  `-0.05`, its total width is at most `0.10`, completion and timeout-progress interval lower
  bounds are at least `-0.05`, and no planner-specific primary point estimate exceeds `+0.05`.
- No gain: the pooled primary interval lower bound is greater than `-0.05`; any harm is disclosed.
- Inconclusive: complete native data meet neither rule, including an interval wider than `0.10`.
- Blocked: any incomplete pair, invalid retained field, provenance mismatch, fallback/degraded row,
  or infrastructure failure. Blocked is not a result.

These rules support only a fixed-suite, configured-pipeline diagnostic statement. They do not
support universal safety or paper-facing promotion.

## Cost and submission boundary

The planning estimate is 1.0 compute-hours, 0.5 wall-clock hours on the two-worker runner, and
0.5 GiB of storage for retained rows, manifests, logs, and validation artifacts with videos
disabled. It is derived from the prior #4830 shape: 864 episodes in 800.6803789430123 seconds at
1.07908226893305 episodes/second. The estimate includes a twofold wall-time allowance and carries
approximately ±50% time uncertainty and up to 2x storage uncertainty. The metadata-only public
snapshot is not treated as a raw-row size proxy.

The campaign is explicitly not submitted by this change. A separate maintainer go/no-go decision
is required after preregistration review and a runner preflight.

## Acceptance checklist

- All required design, outcome, retained-field, analysis, promotion, and cost sections are
  populated in the machine-readable packet.
- Independent review confirmed that the estimand is identifiable from the exact #6970 retained
  schema after reconciling the interval-width precedence and no-gain/inconclusive boundary.
- The cost estimate is stated as a planning estimate with uncertainty.
- The campaign remains explicitly unsubmitted; compute authorization is a separate maintainer
  decision.

## Validation

```text
uv run python scripts/validation/check_preregistration_inference_contract.py --json configs/benchmarks/issue_6971_safety_wrapper_paired_preregistration.yaml
uv run python scripts/benchmark/check_paired_effect_metric_contract.py --contract configs/benchmarks/paired_effect_metric_contract_v1.yaml --json
uv run pytest -q tests/benchmark/test_issue_6971_safety_wrapper_preregistration.py
```

Related design and historical evidence: [#3501 safety-wrapper context](issue_3501_safety_wrapper.md),
[#4830 evidence snapshot](evidence/issue_4830_safety_wrapper_factorial_v1/README.md), and the merged
[#6970 metric contract](../../configs/benchmarks/paired_effect_metric_contract_v1.yaml).
