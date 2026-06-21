# Issue #3342 Near-Field Turn Budget S20 Diagnostic

**Date:** 2026-06-21 · **Evidence tier:** `diagnostic-local` · **Decision:** S20 does not support adopting the near-field turn-budget signal

## Source

This bundle preserves a compact local S20 follow-up for Issue #3342 after the predictive evaluation
summary path was extended to emit aggregate collision rate and uncertainty fields. The local run used
`predictive_proxy_selected_v2_full` from the model registry cache, the
`predictive_hardcase_portfolio_v1` scenario set, and the new
`configs/benchmarks/predictive_hard_seeds_s20_v1.yaml` seed manifest.

The six rows in `summary.json` cover:

- variants: `baseline`, `nearfield_turn`, `nf_speedcap_only`;
- slices: clean and `robustness_smoke_v1` observation noise;
- metrics: success rate, collision rate, mean minimum distance, and confidence intervals.

Raw episode JSONL remains in ignored local run output and is not mirrored here.

## Result

| Slice | Variant | Episodes | Success | Collision | Mean min distance |
|---|---:|---:|---:|---:|---:|
| clean | baseline | 20 | 0.200 | 0.800 | 3.277 |
| clean | nearfield_turn | 20 | 0.150 | 0.850 | 3.223 |
| clean | nf_speedcap_only | 20 | 0.150 | 0.850 | 3.223 |
| noisy | baseline | 20 | 0.200 | 0.800 | 3.276 |
| noisy | nearfield_turn | 20 | 0.150 | 0.850 | 3.224 |
| noisy | nf_speedcap_only | 20 | 0.150 | 0.850 | 3.224 |

The local S20 slice does **not** reproduce the Issue #3215 near-field-turn signal. Baseline was
0.20 success / 0.80 collision in both clean and noisy slices; both near-field variants were
0.15 success / 0.85 collision. Intervals are wide and overlapping, so this is a diagnostic negative,
not a benchmark-strength rejection or paper-facing claim.

## Claim Boundary

- Diagnostic-local evidence only; not benchmark-strength, paper-grade, or release evidence.
- No fallback or degraded rows are promoted as success.
- Collision metric availability is now explicit in the evaluation summary path, but high collision
  rates remain an interpretation result, not a quality-gate failure in this diagnostic run.
- S30 is configured by `configs/benchmarks/predictive_hard_seeds_s30_v1.yaml` but was not run in
  this bundle.
- Use `summary.json` for compact review. Recreate raw local output from the tracked configs and
  command rather than depending on worktree-local generated contents.

## Reproduction

The worker ran the equivalent of the following local campaign loop for each slice and variant:

```bash
uv run python scripts/validation/evaluate_predictive_planner.py \
  --checkpoint <local predictive_proxy_selected_v2_full model-cache path> \
  --scenario-matrix configs/scenarios/sets/predictive_hardcase_portfolio_v1.yaml \
  --seed-manifest configs/benchmarks/predictive_hard_seeds_s20_v1.yaml \
  --algo-config configs/algos/hardcase_authority/prediction_planner_authority_<variant>.yaml \
  --output-dir <local generated run directory> \
  --tag <variant> \
  --min-success-rate 0.0 \
  --workers 4 \
  --observation-noise configs/observation_noise/robustness_smoke_v1.yaml
```

For the clean slice, omit `--observation-noise`.
