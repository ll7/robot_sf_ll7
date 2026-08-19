# Issue #4360 bounded search harness

This preparation layer turns a finite adversarial search description into reproducible candidate
overlays, so infeasible proposals can be inspected before a separately approved simulator run.
It is capability plumbing; it does not execute a campaign or rank planners.

## Claim boundary

- Evidence status: `diagnostic-only` preparation contract.
- No simulator, planner, real manifest, campaign, SLURM job, benchmark result, or paper-facing
  claim is produced by this slice.
- The rollout budget is recorded for later coordination but is not consumed here.
- Objective ordering remains a declared vector. The objective/search-space decision for #7382 and
  the real-manifest execution boundary for #7340 remain separate follow-ups.

## Typed manifest

`robot_sf/adversarial/search_harness.py` defines the preparation contract:

- `SearchVariable` binds a name and physical unit to finite inclusive continuous or integer
  bounds.
- `CrossVariableConstraint` stores a restricted arithmetic/boolean expression and evaluates it
  without executing arbitrary code.
- `ObjectiveVector` records named components, units, and directions without scalarizing them.
- `SeedPolicy` separates the search seed from held-out replay seeds and deterministically derives
  candidate seeds.
- `RolloutBudget` records candidate count, rollouts per candidate, and maximum steps.

The tracked [fixture manifest](../../configs/adversarial/issue_4360_search_harness_fixture.yaml)
uses the YAML mapping form for readability. Canonical JSON serialization uses an ordered variable
list so sampler dimension order is included in the manifest digest.

## Preparation seams

`prepare_equal_budget_baselines` emits exactly the declared candidate budget for both:

1. `random`, a seeded uniform proposal baseline;
2. `quasi_random`, a seeded Halton low-discrepancy (space-filling) proposal baseline.

Each candidate is checked against scalar bounds and cross-variable predicates before an adapter is
called. `MappingOverlayAdapter` provides a generic nested-field seam, while
`CandidateSpecOverlayAdapter` reuses the existing pure
`robot_sf.adversarial.bundle.build_candidate_payload` path. Successful preparation returns an
`ImmutableScenarioOverlay`; rejected candidates carry stable stage and reason codes. Neither seam
loads a map, writes a scenario file, or invokes simulation.

## Focused proof

```bash
uv run pytest -q tests/adversarial/test_search_harness.py
uv run ruff check robot_sf/adversarial/search_harness.py \
  robot_sf/adversarial/materialize.py \
  tests/adversarial/test_search_harness.py
```

These checks prove manifest round-tripping, digest stability, equal-budget determinism, immutable
overlay behavior, reuse of the existing materializer, and pre-simulation rejection accounting.
They are not campaign or benchmark evidence.
