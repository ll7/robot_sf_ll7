# Nominal-Sanity Suite

```yaml
suite_id: policy_search_nominal_sanity
benchmark_track: policy_search_nominal_sanity
status: runnable_local_diagnostic
```

## Purpose

Run a candidate over a small nominal policy-search matrix before any broader stress or full-matrix
campaign. This stage is the first useful signal for common route-following, crossing, doorway, and
following-human behavior.

## Scenarios And Seeds

- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Scenario IDs:
  - `planner_sanity_simple`
  - `classic_head_on_corridor_low`
  - `classic_crossing_low`
  - `classic_doorway_low`
  - `classic_overtaking_low`
  - `francis2023_following_human`
- Seeds: `111`, `112`, `113` for each scenario
- Horizon: `120`
- Workers: `2`

## Eligible Planners

Policy-search candidates with successful smoke evidence and a valid candidate registry entry.
Candidates with missing artifacts or adapter prerequisites should fail closed before this stage.

## Metrics

Success rate, collision rate, near-miss rate, low-progress timeouts via termination/failure
taxonomy, scenario exclusions, scenario-family split, mean minimum distance, and mean speed.

## Canonical Command

```bash
uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate <candidate_id> \
  --stage nominal_sanity \
  --output-dir output/policy_search/<candidate_id>/nominal_sanity/manual \
  --workers 2
```

## Expected Runtime

Typically minutes for lightweight candidates. Treat runtime as candidate-dependent; learned-model
hydration, simulator startup, and planner-specific rollouts can dominate.

## Claim Boundary

Nominal sanity is triage and promotion-gate evidence for policy search. It is not a paper-grade
benchmark and does not by itself prove generalization or safety.

## Caveats

The configured gate in `configs/policy_search/funnel.yaml` is useful for local screening, but any
planner promotion or paper-facing claim needs durable evidence and the appropriate benchmark
contract. Fallback/degraded rows remain caveats.
