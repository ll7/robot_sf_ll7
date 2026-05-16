# Issue #1236 Optimizer-Backed Adversarial Sampler Pilot

## Goal

Issue #1236 asks whether optimizer-backed adversarial samplers can find lower-SNQI or
failure-inducing candidates more efficiently than the existing random and coordinate-refinement
samplers. This note records the first bounded implementation and its evidence boundary.

## Decision

The first optimizer pilot uses `optuna`, which is already in `pyproject.toml`, instead of adding a
new CMA-ES or Bayesian-optimization dependency. `OptunaCandidateSampler` implements the existing
`FeedbackCandidateSampler` contract through Optuna's ask/tell API, so
`run_adversarial_search(...)` remains unchanged and preserves deterministic sequential manifest
ordering.

The sampler:

- maps the current `SearchSpaceConfig` scalar bounds to Optuna suggestions,
- emits `CandidateSpec` values that still pass normal search-space validation,
- receives objective feedback through `observe(evaluation)`,
- records invalid or unscored proposals as failed Optuna trials,
- raises an actionable error if `optuna` is unavailable.

## Implementation Surfaces

- `robot_sf/adversarial/samplers.py` adds `OptunaCandidateSampler`.
- `robot_sf/adversarial/__init__.py` exports the sampler for pilot harnesses.
- `scripts/tools/compare_adversarial_samplers.py` runs random, coordinate, and optuna samplers on
  the same `SearchConfig` and emits an `adversarial-sampler-comparison.v1` JSON summary.
- `tests/adversarial/test_adversarial_search.py` covers deterministic proposals, feedback,
  dependency failure, and the synthetic comparison helper.

## Validation

Red proof:

```bash
uv run --active pytest tests/adversarial/test_adversarial_search.py -q -k 'optuna_candidate_sampler'
```

This initially failed during collection because `OptunaCandidateSampler` did not exist.

Green targeted proof:

```bash
uv run --active pytest tests/adversarial/test_adversarial_search.py -q -k 'optuna_candidate_sampler or sampler_comparison_synthetic'
uv run --active ruff check robot_sf/adversarial/samplers.py robot_sf/adversarial/__init__.py scripts/tools/compare_adversarial_samplers.py tests/adversarial/test_adversarial_search.py
```

Synthetic comparison smoke:

```bash
uv run --active python scripts/tools/compare_adversarial_samplers.py \
  --output-dir output/adversarial/issue1236_sampler_comparison \
  --budget 3 \
  --seed 123 \
  --synthetic \
  --out-json output/adversarial/issue1236_sampler_comparison/summary.json
```

The smoke wrote deterministic manifests for `random`, `coordinate`, and `optuna`. On this tiny
synthetic surface, `random` had the best objective at budget 3 (`-0.1583`) while coordinate
(`-1.05`) and optuna (`-1.0641`) did not improve. That is a useful negative/neutral pilot signal,
not evidence against optimizer search generally; repeated seeds and real benchmark evaluation are
still needed before making claims about efficiency.

## Follow-Up Boundary

This PR does not promote optimizer-backed adversarial search to paper-facing evidence and does not
add candidate-level parallelism. A stronger follow-up should run repeated-seed comparisons on
`configs/scenarios/templates/crossing_ttc.yaml` with
`configs/adversarial/crossing_ttc_space.yaml`, report first-failure iteration and invalid-candidate
rate, and preserve replay bundle paths for any claimed counterexamples.
