# Issue #5318 — Bounded BRNE Integration Decision

Date: 2026-07-11
Issue: <https://github.com/ll7/robot_sf_ll7/issues/5318>
Decision: **GO for bounded, non-benchmark BRNE integration (exploration tier).**
Parents: #5311 (source-side smoke), PR #5313 (merged smoke + contract mapping).

## Decision

**Conditional GO** for a bounded integration exploration. The upstream
pure-numpy/numba BRNE core runs under the 100 ms control budget for
sparse-to-moderate crowds (<=6 interacting agents) and outputs native unicycle
`(v, omega)` directly compatible with Robot SF. Integration is scoped to
corridor-class scenarios only; BRNE does not handle arbitrary static geometry.

This is **not** a blanket go and **not** a benchmark-arm approval. BRNE is
testing-only and exploratory until the validation path below is complete.

## Admissible Scenario Class

**Corridor-class scenarios only**: single-passage maps where corridor bounds
(`corridor_y_min`, `corridor_y_max`) are a faithful representation of the
static obstacle structure. BRNE has no native mechanism for arbitrary polygon
obstacles, walls, doorways, or round obstacles — only y-axis corridor clipping.

Admissible maps are those where the robot's entire feasible path lies within a
constant-width corridor (e.g., `maps/svg_maps/corridor_*.svg` or equivalent
synthetic maps). Scenarios with T-junctions, room corners, or round obstacles
are **out of scope** for this integration tier.

## Static-Obstacle Treatment

Corridor bounds only, matching the upstream implementation:

- `corridor_y_min` / `corridor_y_max` define the feasible y-band.
- Trajectory samples leaving the corridor are zeroed by the upstream `coll_beck`
  mask.
- No static-obstacle cost terms are injected into the proximity matrix.
- This is **upstream-faithful behavior**, not a local extension.

If a future integration needs arbitrary static geometry, it would require a
labeled, documented cost extension that departs from upstream — tracked as a
separate decision, not part of this bounded tier.

## Pedestrian-Policy Declaration

BRNE samples human trajectories as Gaussian-process motion around a
constant-velocity mean; it is crowd-model-agnostic on the human side. For a
fair comparison within the Robot SF benchmark, the bounded integration uses
**ORCA pedestrians** by default:

- ORCA's cooperative velocity-obstacle model aligns better with BRNE's
  cooperative Nash-equilibrium assumption than social-force does.
- If a social-force comparison is desired, it must be explicitly labeled and
  documented with the crowd-model caveat, as for SICNav (#4870) and CrowdNav
  (#4871).

## Fail-Closed Dense-Crowd / Budget Policy

When the interacting-agent count exceeds budget at the current `num_samples`:

1. **Step-budget enforcement**: if `brne_nav` solve time exceeds the configured
   step budget (`time_per_step_in_secs`, default 100 ms), the adapter logs the
   overrun and returns a zero-motion action `{"v": 0.0, "omega": 0.0}` for that
   step. The episode continues; the planner does not crash.
2. **Adaptive `num_samples`** (optional, opt-in): when
   `adaptive_num_samples=True`, the adapter reduces `num_samples` in powers of
   the BRNE meshgrid constraint (perfect squares: 196 -> 144 -> 100 -> 64 -> 49)
   until the solve fits budget or the minimum is reached. Each reduction is
   logged.
3. **Maximum-agent cap**: agents beyond `maximum_agents` (default 8, matching
   upstream) are excluded from the BRNE solve (sorted by distance ascending).
   This is the upstream convention, not a local override.

The zero-motion fallback is fail-closed: it does not claim the planner solved
successfully when the budget was exceeded.

## Reproducible Validation Path

The bounded integration must pass all of the following before a benchmark-arm
proposal is considered:

1. **Adapter integration test** (CPU-only): the new `BRNEPlanner.step()` returns
   valid `{"v", "omega"` actions against synthetic observations with the staged
   upstream core, and the solve completes under budget for <=5 agents.
2. **Corridor-class scenario smoke**: run the BRNE adapter on a small set of
   corridor-class scenarios and verify goal-reaching and non-degenerate behavior
   (no permanent zero-motion, no corridor violations).
3. **Source-side smoke still passes**: `tests/baselines/test_brne_source_smoke.py`
   and `scripts/tools/probe_brne_source_harness.py` remain green (no regression
   in the upstream contract).

## Condition for Benchmark-Arm Proposal

A benchmark-arm proposal is permitted only after:

1. The corridor-class smoke above passes with documented success rates.
2. The adapter is exercised on the verified-simple scenario subset (#596 gate)
   with corridor-class maps only.
3. A maintainer explicitly approves the benchmark arm (separate issue, not this
   decision).

## Claim Boundary

This decision authorizes an **exploration-tier integration**: a BRNE planner
adapter registered as testing-only, scoped to corridor-class scenarios, with
fail-closed budget enforcement. It is **not** a benchmark claim, not a
paper-facing result, and does not promote BRNE beyond testing-only in the
planner readiness roster.

## Related

- `docs/context/issue_5311_brne_source_smoke.md` — source-side smoke evidence
  (PR #5313, merged).
- `docs/benchmark_experimental_planners.md` — planner readiness roster.
- `docs/context/issue_4870_sicnav_smoke.md` — closest analog (MPC comparator).
- `docs/context/issue_4871_crowdnav_pred_attng_smoke.md` — learned-baseline
  comparator smoke.
