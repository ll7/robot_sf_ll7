# Issue #4360 — Reactive adversarial stress testing: residual-control design

Status: capability-only (post-freeze item 2, first runtime implementation).
Evidence grade: not benchmark evidence. No paper, metric, or safety claim.

This note records the decided design for the bounded residual-control reactive
adversary so the first post-freeze session started from a decided contract
rather than re-deriving one. It documents what the capability-only slice ships,
the claim boundary it respects, and the slices it deliberately defers.

## Plain-language summary

The adversary adds a small, hard-bounded "nudge" (a residual acceleration) on
top of the normal Social Force pedestrian behavior. The normal behavior
(`pysocialforce.forces.SocialForce`) keeps running unchanged — the adversary
only perturbs it, it never replaces it. The nudge is recomputed every 0.5 s
(the macro-action cadence) and held constant in between, and every nudge is
clamped so a pedestrian cannot exceed speed, acceleration, jerk, heading-turn,
route-deviation, walkable-space, or inter-agent-separation limits. It is off by
default and changes nothing about existing pedestrian models.

## What ships (the contract)

- `robot_sf/ped_npc/residual_adversary.py`:
  - `ResidualAdversaryConfig` — validated, opt-in config (`is_active=False`
    default) with every hard bound.
  - `ResidualAdversaryPolicy` — the `Protocol` that future Covariance Matrix
    Adaptation Evolution Strategy (CMA-ES), Monte Carlo Tree Search (MCTS), or
    Proximal Policy Optimization (PPO) adversaries implement;
    `propose_residual(observation) -> (N, 2)`.
  - `ScriptedPullResidualAdversaryPolicy` — a deterministic, bounded example
    policy for runtime wiring and tests. It is NOT the search or learned
    adversary.
  - `BoundedResidualAdversary` — the stateful controller that holds the
    macro-action proposal and enforces every hard bound each physics step.
  - Pure, individually-testable bound helpers: `clamp_magnitude`,
    `rate_limit_jerk`, `bound_speed`, `bound_heading_change`,
    `bound_route_deviation`, `project_residual_displacement_walkable`,
    `enforce_inter_agent_separation`.
- `robot_sf/sim/sim_config.py`: `SimulationSettings.residual_adversary` field,
  normalized and validated in `__post_init__`.
- `robot_sf/training/scenario_loader.py`: a scenario's
  `simulation_config.residual_adversary` mapping is validated and applied to
  `SimulationSettings`, so the opt-in parameters can be stored with a
  reproducible scenario rather than only constructed in Python.
- `robot_sf/sim/simulator.py`: lazy `_build_residual_adversary` +
  `_apply_residual_adversary`; the residual is added to the already-computed
  pedestrian forces in both `step_once` paths so the base law is preserved.
- `configs/adversarial/issue_4360_residual_adversary.yaml`: documented opt-in
  example.
- Tests: `tests/adversarial/test_residual_adversary.py` (bounds, cadence,
  projection, separation, opt-in, base-law preservation, fail-closed) and
  `tests/sim/test_residual_adversary_wiring.py` (opt-in gating, perturb-not-
  replace, finite-state smoke).

## Hard bounds (all fail-closed on non-finite input)

| Bound | Knob | Mechanism |
| --- | --- | --- |
| Speed | `max_speed_delta_mps` + pysf `max_speeds` | resulting speed `|v + r·dt|` capped at `min(max_speeds, |v| + delta)` |
| Acceleration | `max_residual_accel_mps2` | row-wise magnitude clamp of `r` |
| Jerk | `max_jerk_mps3` | move applied `r` toward the held proposal by at most `max_jerk·dt` per step |
| Heading change | `max_heading_change_per_macro_rad` | perpendicular (turning) component of `r` capped per step |
| Route deviation | `max_route_deviation_m` | residual displacement scaled so the would-be position stays within the corridor of the reference polyline |
| Walkable space | `obstacle_projection_margin_m` + bounds | residual displacement redirected so the would-be position clears obstacle segments and stays inside map bounds |
| Inter-agent separation | `min_separation_m` | targeted residual displacements scaled so separation is not reduced below the minimum |

## Base-law preservation (perturb, not replace)

The residual is added to the forces PySocialForce already computed
(`pysf_sim.compute_forces()`), immediately before `_step_pedestrians`. The
Social Force contribution is unchanged; only an additive, bounded residual is
injected. With `is_active=False` (the default) or a zero-proposal policy, the
applied residual is exactly zero and stepping is identical to a simulator
without the wiring. The existing scripted `AdversarialPedForce` semantics and
all pedestrian-model defaults are unchanged.

## Macro-action cadence

`macro_steps = round(macro_action_dt_s / dt_s)` (default `0.5 s / 0.1 s = 5`).
Every `macro_steps` physics steps the controller requests a fresh proposal from
the policy and holds it constant in between. Jerk rate-limiting bounds how fast
the applied residual can move toward the held proposal, so a step change in the
proposal still produces a smooth, bounded residual.

## Reproducible runtime smoke

The one canonical CPU command for deterministic residual-adversary smoke uses the
existing unit-test path with the deterministic `ScriptedPullResidualAdversaryPolicy`:

```bash
scripts/dev/run_worktree_shared_venv.sh -- \
  uv run pytest tests/adversarial/test_residual_adversary.py \
  tests/sim/test_residual_adversary_wiring.py \
  -q
```

This exercises the full bound pipeline (speed, acceleration, jerk, heading,
route, walkable-space, inter-agent separation), macro-action cadence, opt-in
gating, perturb-not-replace base-law preservation, and simulator wiring. The
`ScriptedPullResidualAdversaryPolicy` is deterministic; combined with the
fixed-timestep `BoundedResidualAdversary`, every step residual is identical
across repeated runs with the same config and inputs. The test loads
`configs/adversarial/issue_4360_residual_adversary.yaml`, binds its template
`seed: null` to the fixed smoke seed `42`, and advances `20` steps at `dt=0.1 s`.
The seed field on `ResidualAdversaryConfig` is reserved for future randomized
policies (CMA-ES, MCTS, PPO); the bundled scripted policy does not consume it.

**Output / claim-status**: These tests prove capability-only smoke evidence —
runtime wiring, finite-state behavior, and bound enforcement for the deterministic
scripted policy. This is **not** benchmark, safety, stress-strength, or
paper-facing evidence. The YAML example is a documented parameter template: copy
its `residual_adversary` mapping beneath a scenario's `simulation_config` key.
The scenario loader validates that nested mapping before passing it to the
runtime config.

## Claim boundary (what this slice does NOT do)

This is a capability-only slice. It makes **no** benchmark, planner-ranking,
safety, or paper-facing claim. It defines **no** new stress-case metric. It
implements **no** CMA-ES/MCTS search-baseline adversary and **no** PPO/learned
adversary. It runs **no** matched-compute comparison against the open-loop
scenario-optimization pipeline. Naming discipline: "reactive adversarial stress
testing", **not** "most-likely failure search" (the latter would require a
calibrated pedestrian-behavior probability model we do not have).

## Grid-search baseline (issue #6911, first search-baseline slice)

`robot_sf/ped_npc/residual_search_baseline.py` ships the smallest useful,
deterministic, config-first residual search-baseline slice. It implements
`ResidualAdversaryPolicy` via `GridSearchResidualPolicy` and evaluates a small
explicit action grid of candidate residual accelerations against a simple
objective proxy.

**Algorithm**: For each macro-action boundary, for each targeted pedestrian
independently:

1. Enumerate `num_directions` (default 8) evenly-spaced angular directions in
   `[0, 2*pi)`.
2. For each direction, evaluate `num_magnitudes` (default 3) magnitude levels
   from `0` to `max_residual_accel_mps2`.
3. Score each candidate with the objective proxy (lower is better):
   `-weight_approach * approach_speed + weight_distance * distance_to_robot`.
4. Select the candidate with the lowest score (closest approach).
5. Emit a zero-residual candidate as a baseline comparison.

**Objective proxy**: Negative weighted approach closeness — a heuristic, NOT a
calibrated pedestrian-behavior probability model. It combines approach speed
(velocity component toward the robot after applying the candidate) and distance
to the robot.

**Deterministic contract**: Given a fixed seed, action grid, budget, and macro
cadence, repeated runs produce identical proposal sequences and diagnostic
metadata. The seed deterministically orders candidate evaluation, and the
finite macro budget is enforced: after it is exhausted the policy emits zero
proposals without evaluating more candidates.

**Diagnostic records**: Each macro-action boundary emits a JSON-serializable
record with config identity/schema, optional source revision, algorithm and
objective identifiers, seed, explicit candidate order, grid size, budget, bound
settings, and accepted/rejected/invalid candidate counts. The
`write_diagnostics(path)` helper writes the full accumulated diagnostics and
records whether the finite budget was exhausted.

**Config**: `configs/adversarial/issue_4360_residual_search_baseline.yaml`

**Tests**: `tests/adversarial/test_residual_search_baseline.py` — config
validation, deterministic repeated output, fail-closed invalid candidates,
bound-preserving integration with `BoundedResidualAdversary`, diagnostic
record emission, and JSON serialization.

**Canonical smoke command**:

```bash
uv run pytest tests/adversarial/test_residual_search_baseline.py \
  tests/adversarial/test_residual_adversary.py \
  tests/sim/test_residual_adversary_wiring.py -q
```

**Claim boundary**: Capability-only. No benchmark, metric, schema,
planner-ranking, safety, or paper-facing claim. The objective proxy is a
heuristic, not a calibrated model. No CMA-ES/MCTS or PPO adversary and no
matched-compute comparison. Objective/metric synthesis, matched-compute
comparison, planner integration, PPO, and benchmark campaigns remain deferred.

## Deferred slices (pre-registered plan)

- CMA-ES or MCTS search-baseline adversary — sequenced **before** any PPO
  adversary — will implement `ResidualAdversaryPolicy`. The grid-search
  baseline (issue #6911) is the first search-baseline slice; CMA-ES/MCTS
  remain separate candidate algorithms to evaluate after this deterministic
  interface smoke.
- PPO / learned adversary — only after the search baseline is measured.
- Matched-compute comparison vs open-loop scenario optimization (the claim to
  test: reactivity finds failures open-loop search cannot, at equal simulator
  budget). Related: #5303 (transfer matrix) and #5305 (certified archive) run on
  the open-loop pipeline first and become the comparison set.
- Stress-case validity/strength metrics (issue item 4) — deferred to a follow-up
  that requires a maintainer Domain-Aware Approval before merge-readiness.

## Residual risks and limitations

- The walkable-space projection uses obstacle segments and rectangular map
  bounds; it is a push-out/clamp projection, not a full polygon containment
  test, and degrades to a no-op when geometry is unavailable (the kinematic and
  separation bounds still fire). If nominal state is already inside the
  controller's configured clearance margin, the controller suppresses its
  residual instead of attempting an unsafe partial repair.
- Route-deviation references each route-following pedestrian's assigned route
  polyline. Pedestrians without a route assignment (for example, crowded-zone
  or scripted pedestrians) intentionally have no route-deviation projection.
- The policy observation carries one robot pose. In a multi-robot simulator,
  this first capability slice uses the first simulator robot as its reactive
  reference; explicit multi-robot target selection is deferred.
- In pedestrian-centric simulations, the externally controlled ego-pedestrian
  row is never an adversary target. It remains visible to inter-agent separation
  checks so targeted non-player pedestrians cannot be nudged through it.
- The bundled scripted policy is a placeholder baseline; adversarial strength
  is not claimed and not measured here.
