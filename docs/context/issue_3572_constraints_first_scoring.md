# Issue #3572 — Constraints-first (non-compensatory) scoring layer (first increment)

**Status:** diagnostic / post-hoc analysis. **Evidence grade:** idea-level; claims no
planner is "safe" — it only ranks under explicit admissibility. Keeps the existing
compensatory composite available for comparison.

## What this is

`robot_sf/benchmark/constraints_first_scoring.py` adds an **optional, post-hoc** scoring
layer over existing episode records. The headline scoring is *compensatory* (a collision
can be offset by faster/smoother motion); for safety-relevant navigation that is the wrong
default. This layer does **not** touch the simulator or the metric producers.

## Primitives (`constraints_first_scoring.v1`)

- **`collision_upper_confidence_bound(n_events, n_episodes, confidence=0.95)`** — one-sided
  Clopper–Pearson upper bound. At `n_events = 0` it reproduces the rule-of-three
  (`≈ 3/N`), so a low-N planner with zero observed collisions is not reported as
  collision-free.
- **`AdmissibilityGates` + `is_episode_admissible`** — lexicographic gates
  (collision → near-miss severity → timeout/deadlock); comfort/efficiency rank only among
  admissible runs.
- **`survivorship_aware_metric`** — each metric reported unconditionally and conditioned on
  safe success, plus the survivorship-bias delta (conditioning flatters planners that fail
  more often).
- **`constraints_first_planner_summary`** — admissibility rate, collision rate + UCB, and
  survivorship-aware comfort/efficiency for one planner.
- **`ranking_inversion`** — per-planner ranks under the compensatory composite vs the
  constraints-first order, and the planners whose rank changes (the empirical justification
  for non-compensatory evaluation).

## Scope boundary

Pure and side-effect free — no change to the simulator, metric producers, or default
benchmark reporting. Wiring these primitives into the benchmark report/CLI (and emitting a
ranking-inversion table per campaign) is a deliberate follow-up.

## Tests

`tests/benchmark/test_constraints_first_scoring.py` (16 tests): rule-of-three at k=0, UCB
monotonicity and validation, each admissibility gate, the survivorship delta, the planner
summary contract, and ranking-inversion detection / no-inversion / mismatched-set handling.

## References

- Francis et al., *Principles and Guidelines for Evaluating Social Robot Navigation
  Algorithms*, ACM THRI 2025.
- Stratton et al., *Characterizing the Complexity of Social Robot Navigation Scenarios*,
  RA-L 2025.
