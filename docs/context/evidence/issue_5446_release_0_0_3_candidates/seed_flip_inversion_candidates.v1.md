<!-- AI-GENERATED (robot_sf#5446) - NEEDS-REVIEW -->
# Seed-flip & held-out planner-inversion candidates (issue #5446)

- schema: `seed_flip_inversion_candidates.v1`
- rows: 20160 | eligible cells: 96 | excluded rows: 17280
- candidates: 50 (seed-flip 45, upset 5) | Pareto-selected: 2

> Analysis tooling: seed-flip / held-out planner-inversion CASE CANDIDATES only. Not a benchmark metric, not a planner-ranking claim. Confirmation runs are a separate exact-compute packet. Candidates below the triage seed count or without held-out strength are flagged; they are proposals, not established effects.

## Archetype availability
- `seed_flip`: available
- `planner_upset`: available
- `causal_divergence`: unavailable
- `disagreement_recovery`: available

## Selected candidates (Pareto frontier)
- **seed_flip** `classic_doorway_medium` / `ppo`: flip entropy=1.000 bits, n_seeds=30, CI=0.33-0.67; disagreement=1.000 bits
- **planner_upset** `classic_realworld_double_bottleneck_high` / `goal`: heldout skill gap=0.330; disagreement=0.000 bits
