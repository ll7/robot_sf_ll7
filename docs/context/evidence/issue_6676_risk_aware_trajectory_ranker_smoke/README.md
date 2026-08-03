# Risk-aware trajectory ranker smoke comparison

This diagnostic compares deterministic motion primitives with finite radial-basis-function (RBF) candidate proposals under the same held-out start states, local goals, actor predictions, candidate budget, risk estimator, and deterministic hard gates.

## Claim boundary

- Evidence status: `smoke/diagnostic` only.
- Baseline: `deterministic_primitive` from `generate_primitive_candidates`.
- Variant: opt-in deterministic RBF proposals from `generate_rbf_candidates`; this is not a trained policy.
- The report separately records candidate validity, `verify_trajectory` fallback-brake rejection, actuator non-feasible rejection, selection differences, model-score reliability checks, and timing.
- Risk scores are constant-velocity model scores, not calibrated real-world collision probabilities. Timing is local offline wall time, not online performance evidence.
- Planner-loop wiring, online adaptation, nominal benchmark evidence, planner improvement, safety, and real-world claims remain deferred.

## Reproduction

From the repository root, run:

```bash
uv run python scripts/benchmark/run_risk_aware_trajectory_ranker_comparison.py \
  --config configs/benchmark/risk_aware_trajectory_ranker_comparison.yaml \
  --output <external-report>.json \
  --output-md <external-report>.md
```

The committed config points to `tests/benchmark/fixtures/risk_aware_trajectory_ranker_comparison_v1.yaml`, whose rows are all marked `held_out` and `valid`. The JSON report records the command, config and fixture SHA-256 digests, seed, exact commit SHA, matched-comparison policy, and fallback/degraded exclusions. Keep generated reports outside the worktree or promote only a small provenance-bearing manifest; do not commit raw timing logs.

The report is diagnostic evidence for issue [#6676](https://github.com/ll7/robot_sf_ll7/issues/6676), not nominal benchmark evidence.
