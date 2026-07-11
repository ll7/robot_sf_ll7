# Issue #3574 Pre-Specified Rank-Reversal Test

This note records the preregistered rank-reversal test primitive added for issue #3574. It is
the *pre-specified rank-reversal test* named in the issue Definition of Done ("Rank-order
sensitivity across >=3 planners per population composition reported with bootstrap P(A beats B)
and a pre-specified rank-reversal test").

## What this is

A distinct, additive statistical primitive that complements the descriptive bootstrap
rank-sensitivity report:

- `compute_bootstrap_rank_sensitivity` (issue #4591) reports bootstrap P(A beats B) and a
  **descriptive** ranking-list disagreement flag that fires on *any* ordering difference,
  including differences within bootstrap sampling noise.
- `pre_specified_rank_reversal_test` (this slice) is a **formal hypothesis test**. It only
  declares a reversal when a planner pair has a bootstrap-determined ordering (percentile CI
  excludes zero) in **both** arms with **opposite** signs, so a reversal cannot be triggered by
  sampling noise.

## Decision rule (declared before results)

- **Null hypothesis (H0):** planner rank order is stable across the heterogeneous and
  mean-matched homogeneous population compositions (no reversal).
- **Test statistic:** per-seed paired performance difference per planner pair, per arm,
  bootstrapped into a percentile confidence interval.
- **Significance level:** `alpha` (default `0.05`); CI level `1 - alpha`.
- **Decision:** reject H0 iff some planner pair is determined in both arms with opposite signs.
- The `alpha`, CI method, and decision rule are echoed in the output `pre_registration` block as
  `declared_before_results: true`, so the comparison is auditable when real paired campaign
  episode records arrive.

## Why "pre-specified"

The test parameters and decision rule are fixed before the results are inspected (preregistration),
not fit to the observed outcome. This is what the issue's literature grounding demands: a
planner-specific, rank-reversal claim must be declared ex ante, not reported post hoc.

## Entrypoint

```bash
uv run python scripts/benchmark/run_rank_reversal_test_issue_3574.py \
  --manifest output/issue_3574_mean_matched_harness/manifest.json \
  --records output/issue_3574_mean_matched_harness/episode_records.jsonl \
  --output-dir output/issue_3574_mean_matched_harness
```

The CLI fails closed on the mean-matched integration-readiness check before rendering any test.
Outputs use schema `heterogeneous_rank_reversal_test.v1` and write
`rank_reversal_test.json` plus a compact `rank_reversal_test.md`.

## Claim boundary

- Evidence status: `diagnostic-only` analysis primitive.
- This slice establishes **no** benchmark, rank-stability, realism, or sim-to-real claim on its
  own.
- A campaign conclusion requires real paired episode records that pass the readiness check; until
  then the test is exercised on synthetic fixtures only.
- The mean-matched paired ablation campaign, realized-distribution audit completion, and any
  rank-stability conclusion remain open under the issue.
