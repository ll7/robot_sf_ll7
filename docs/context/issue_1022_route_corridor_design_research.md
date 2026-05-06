# Issue #1022 Route-Corridor Design Research

Date: 2026-05-06

Related issues:

- #1022: <https://github.com/ll7/robot_sf_ll7/issues/1022>
- #884: <https://github.com/ll7/robot_sf_ll7/issues/884>

Related docs:

- `docs/context/issue_884_classic_merging_diagnostics.md`
- `configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape.yaml`
- `robot_sf/planner/hybrid_rule_local_planner.py`
- `robot_sf/planner/grid_route.py`

## Goal

Convert the remaining #884 classic-merging recovery work from an exhausted tuning loop into a
research-backed planner design. This note does not claim a benchmark improvement and does not close
#884. It identifies the evidence that exists, the diagnostics still missing, and the narrow
implementation split that should happen next.

## Current Repo Evidence

The current candidate remains `hybrid_rule_v3_fast_progress_static_escape`. Its config enables:

- `static_clearance_escape`,
- `static_recenter`,
- guarded `static_corridor_transit`,
- route guide support through the `hybrid_rule_v3_teb_like_rollout.yaml` base config.

`HybridRuleLocalPlannerAdapter` currently generates these local candidate families:

- dynamic-window velocity samples,
- direct path-follow candidates,
- optional `route_guide` command from `GridRoutePlannerAdapter`,
- stop, creep, and left/right rotate candidates.

It records per-step `last_decision()` diagnostics for selected source, top candidates,
aggregate rejection counts, moving-command rejection counts, source-level rejection counts,
nearest static/pedestrian distance, predicted TTC, and goal-distance progress windows.

`GridRoutePlannerAdapter` already computes an occupancy-grid A* route and applies a clearance-map
penalty so routes prefer corridor centers. The only public route-geometry surface is
`route_waypoint(observation)`, which returns one world-space waypoint. It does not expose the path,
route tangent, corridor width, lateral offset, or route-arc progress.

## Regenerated Five-Seed Evidence

Command shape:

```bash
rtk env LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --scenario-name <scenario> \
  --seed <seed> \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_1022_current_main_repro/<scenario>_<seed>_h500
```

The command was run for:

- `classic_merging_low` seeds `111`, `113`,
- `classic_merging_medium` seeds `111`, `112`, `113`.

Output root:

- `output/ai/autoresearch/issue_1022_current_main_repro/`

| Scenario | Seed | Outcome | Final goal distance | Selected sources | Last moving rejection signature |
|---|---:|---|---:|---|---|
| `classic_merging_low` | 111 | timeout, no obstacle collision | `22.8216` | `dynamic_window=456`, `creep=44` | `static_clearance=57`; dynamic window, creep, path-follow, route-guide all blocked |
| `classic_merging_low` | 113 | route-complete success at step `330` | `2.0800` before terminal step | `dynamic_window=288`, `creep=32`, `path_follow=9`, `route_guide=2` | none in final decision |
| `classic_merging_medium` | 111 | timeout, no obstacle collision | `23.0213` | `dynamic_window=457`, `creep=40`, `path_follow=3` | `static_clearance=51`; dynamic window, creep, path-follow, route-guide all blocked |
| `classic_merging_medium` | 112 | timeout, no obstacle collision | `22.5234` | `dynamic_window=397`, `creep=94`, `path_follow=8`, `route_guide=1` | `static_clearance=57`; dynamic window, creep, path-follow, route-guide all blocked |
| `classic_merging_medium` | 113 | timeout, no obstacle collision | `6.8960` | `dynamic_window=393`, `creep=105`, `path_follow=2` | `static_clearance=56`; dynamic window, creep, path-follow blocked |

Observed implication: after the guarded corridor-transit partial fix, the five-seed target set no
longer reproduces the earlier low-density obstacle collision on current main. Four seeds still
time out, and their terminal decisions still show moving commands rejected by static clearance.
This supports the #884 diagnosis that the current planner cannot represent or score the needed
route-corridor maneuver.

Important diagnostic gap: the regenerated traces do not include observations or route-corridor
geometry. They cannot answer whether the robot is before, at, or past the route corner; whether the
route tangent points into the corridor center; or whether a safe corridor-centered primitive exists
near the final stall. That missing geometry should be added before behavior changes.

## External Research Takeaways

Primary and official sources support a small route-corridor primitive rather than another broad
constant sweep.

- Dynamic Window Approach is a velocity-space local collision-avoidance method from Fox, Burgard,
  and Thrun. It is a good match for bounded command sampling, but #884 shows a case where the
  existing local velocity samples do not express the needed static-corridor maneuver.
  Source: <https://cir.nii.ac.jp/crid/1363388843526965632>.
- State-lattice motion primitives encode robot mobility constraints into precomputed feasible
  primitives and improve search over kinodynamically valid motions. This supports adding a very
  small primitive family rather than only increasing DWA sample density.
  Source: <https://publications.ri.cmu.edu/kinodynamic-motion-planning-with-state-lattice-motion-primitives/>.
- Nav2's Smac State Lattice planner is a practical precedent for offline minimum control sets,
  in-place rotations, and cost-aware search, but it is heavier than the first #884 follow-up should
  be. Source: <https://docs.nav2.org/configuration/packages/smac/configuring-smac-lattice.html>.
- Nav2 Regulated Pure Pursuit uses forward collision checking, curvature speed regulation, and
  obstacle-proximity speed scaling. This maps directly to the #884 collision-side concern: slow
  before high-curvature or high-cost route corners rather than entering the hard static band at
  speed. Source: <https://docs.nav2.org/configuration/packages/configuring-regulated-pp.html>.
- The Regulated Pure Pursuit implementation notes call out tight-space overshoot, curvature-based
  speed reduction, and proximity-based slowing as practical motivations. Source:
  <https://github.com/ros-navigation/navigation2/blob/main/nav2_regulated_pure_pursuit_controller/README.md>.
- ORCA is primarily a reciprocal multi-agent collision-avoidance formulation. It is relevant for
  pedestrian interactions, but it is not the strongest fit for the observed static-corridor
  clearance deadlock. Source: <https://gamma-web.iacs.umd.edu/ORCA/>.
- Convex feasible-set and control-barrier-corridor work gives useful long-term framing for safe
  local corridors, but these approaches are probably too heavy for the next PR. Sources:
  <https://www.ri.cmu.edu/publications/the-convex-feasible-set-algorithm-for-real-time-optimization-in-motion-planning/>
  and <https://arxiv.org/abs/2603.06494>.

## Design Options

### Option A: Diagnostic-Only Route-Corridor Attribution

Add route-corridor geometry to `last_decision()` and the step trace before changing planner
behavior. Minimum fields:

- route waypoint,
- route path length or cell count,
- route tangent heading,
- route heading error,
- route-corner or tangent-change distance,
- robot lateral offset to the route segment,
- corridor center clearance / width estimate from the occupancy-grid clearance map,
- route-arc progress over 1 s and 3 s windows,
- route-guide candidate acceptance/rejection detail.

Pros:

- Directly addresses the current evidence gap.
- Low behavior risk because it is diagnostic-only.
- Produces the proof surface needed to design and review a corridor primitive.

Cons:

- Does not improve #884 by itself.
- Requires exposing more of `GridRoutePlannerAdapter` than the current `route_waypoint()` surface.

Decision: do first.

### Option B: Narrow `corridor_subgoal` Primitive

After Option A, add a guarded `corridor_subgoal` source inside
`HybridRuleLocalPlannerAdapter`. Generate a small number of route-centered primitives from the
existing grid route rather than adding a new global planner.

Candidate trigger signals:

- stalled or near-stalled `progress_windows["3s"]`,
- moving `static_clearance` rejections dominate,
- rejection spread includes dynamic-window and route/path sources,
- no nearby pedestrian inside the existing slow/stop distance,
- current static clearance remains above occupied-cell collision,
- route-corridor geometry is available.

Candidate acceptance:

- every rollout pose is unoccupied,
- hard static and dynamic safety checks pass,
- positive route-arc progress is predicted,
- route tangent alignment and corridor centering improve over stop/creep,
- speed is capped in narrow/high-curvature corridor states.

Pros:

- Directly targets the four timeout seeds.
- Reuses existing grid-route machinery and current candidate evaluation/scoring patterns.
- Keeps hard safety fail-closed.

Cons:

- Prior unguarded/weakly guarded corridor attempts caused obstacle-collision regressions, so the
  trigger and rollout proof must be strict.
- Needs new unit tests plus five-seed and nominal/stress proof before any benchmark claim.

Decision: second follow-up, after Option A exposes enough geometry.

### Option C: Anticipatory Route-Corner Speed Regulation

Add a Regulated-Pure-Pursuit-inspired speed term for route corners: when route curvature or tangent
change is high and static clearance is shrinking, lower the preferred linear speed before the hard
static band collapses.

Pros:

- Addresses the collision-side version of #884 if it reappears under future changes.
- Smaller than a full primitive library.

Cons:

- On current main, the five-seed set no longer shows an obstacle collision, so this is not the
  primary current failure.
- If applied alone, it can convert progress into more timeouts.

Decision: include as an acceptance criterion or secondary term in Option B, not as the first
standalone behavior PR.

### Option D: Heavier TEB / State-Lattice / Safe-Corridor Planner

Replace or wrap the local policy with a short-horizon optimizer, state lattice, or safe-corridor
method.

Pros:

- Stronger theoretical and practical fit for constrained local trajectory generation.
- Could solve a broader class of route-corner and corridor failures.

Cons:

- Larger integration and benchmark-provenance burden.
- More likely to exceed the narrow #884 recovery scope.

Decision: keep as research background unless Options A and B fail.

## Recommended Follow-Up Split

1. [#1027](https://github.com/ll7/robot_sf_ll7/issues/1027): Trace-level route-corridor
   attribution for classic merging.
   Diagnostic-only. Expose route/corridor geometry and route-arc progress in `last_decision()` and
   step traces. Validate on the five #884 seeds.
2. [#1028](https://github.com/ll7/robot_sf_ll7/issues/1028): Corridor-subgoal recovery primitive
   for `hybrid_rule_local_planner`.
   Add `corridor_subgoal` only after diagnostic attribution proves the trigger geometry. Preserve
   hard static/dynamic safety and fail closed.
3. [#1029](https://github.com/ll7/robot_sf_ll7/issues/1029): Classic-merging validation matrix.
   Run the five target seeds, then `nominal_sanity` and `stress_slice`; run a full matrix or
   justified targeted matrix before claiming #884 improvement.

## Implementation Acceptance Criteria

Any future behavior PR must:

- keep strict hard static-collision safety,
- avoid fallback/degraded success claims,
- preserve `classic_merging_low` seed `113` route-complete success,
- recover at least one of the remaining timeout seeds before a partial-improvement claim,
- introduce no obstacle-collision regressions in the five target seeds,
- run horizon-500 `nominal_sanity` and `stress_slice` with no new obstacle collisions,
- report whether the planner ran in native, adapter, fallback, or degraded mode.

A closing PR for #884 should keep the stricter bar from the parent issue: evaluate all five named
seeds and show an improvement over the current `130/141` baseline with a full matrix or a justified
targeted matrix.

## Current Conclusion

#1022 answers the routing question: the next implementation should not be another scalar-tuning
pass. Start with diagnostic route-corridor attribution, then implement a small guarded
`corridor_subgoal` primitive if the trace geometry confirms the route-corner/static-corridor
hypothesis. Keep heavier TEB, lattice, and safe-corridor approaches as later options if the narrow
primitive cannot satisfy the five-seed proof.
