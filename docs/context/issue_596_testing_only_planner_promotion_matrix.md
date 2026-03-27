# Issue 596 Testing-Only Planner Promotion Matrix

This note consolidates the current promotion status for the six testing-only planners named in
issue 596. It is a review aid and evidence index. The source-of-truth promotion policy remains:

- `docs/benchmark_experimental_planners.md`

## Summary

All six planners remain testing-only. The existing benchmark notes already contain enough negative
evidence to justify keeping the fail-closed guard in place.

The issue-596 atomic suite and verified-simple subset add a better stage-1 gate for any future
revisit, but they do not by themselves overturn the current benchmark verdicts.

## Planner matrix

| Planner | Evidence note | Broad verdict | Main blocker | Required next step before reconsideration |
| --- | --- | --- | --- | --- |
| `risk_dwa` | `docs/context/issue_679_risk_dwa_benchmark.md` | Keep testing-only | Goal-reaching is too weak and collisions are too high relative to predictive baselines. | Hypothesis-driven change that improves progress without losing its reactive safety/runtime strengths. |
| `mppi_social` | `docs/context/issue_677_mppi_social_benchmark.md` | Keep testing-only | Runtime is far too high for the observed success level. | Concrete search/runtime simplification plus rerun on verified-simple before broader benchmarking. |
| `predictive_mppi` | `docs/context/issue_675_predictive_mppi_benchmark.md` | Keep testing-only | Worse success/collisions plus extreme runtime cost. | New control-search hypothesis, not more reruns of the unchanged config. |
| `hybrid_portfolio` | `docs/context/issue_673_hybrid_portfolio_benchmark.md` | Keep testing-only | Portfolio switching does not beat simpler baselines on outcomes or runtime. | Explain and change the arbitration logic before any new benchmark spend. |
| `stream_gap` | `docs/context/issue_681_stream_gap_benchmark.md` | Keep testing-only | Safety is strong but success is zero. | Restore commitment/progress on verified-simple scenarios first. |
| `gap_prediction` | `docs/context/issue_671_gap_prediction_benchmark.md` | Keep testing-only | Veto behavior is over-conservative and prevents goal reaching. | Fix progress recovery, then rerun verified-simple before full paper-surface comparison. |

## Promotion path

For any future candidate among these six planners, the intended path is:

1. pass `configs/scenarios/sets/verified_simple_subset_v1.yaml` at a calibrated bar
2. remain contradiction-free there
3. present a concrete hypothesis for why the broader benchmark result should improve
4. rerun broader benchmark evidence
5. only then consider removing `allow_testing_algorithms: true`
