# Issue #7809: Joint-perturbation preparation boundary

Issue: <https://github.com/ll7/robot_sf_ll7/issues/7809><br>
Parent: <https://github.com/ll7/robot_sf_ll7/issues/7392><br>
Preparation base: `origin/main` / `f8ed55074fdf9178cfaa3edcc533986f337e2218`

## Status and claim boundary

This is a diagnostic-only preparation packet extracted from blocked Issue #7392. It records
observed field owners, existing pure-data seams, symbolic cost accounting, and focused rejection
probes. It does not select the scientific contract, change an authored scenario, run a simulator,
planner, optimizer, campaign, or benchmark, or produce a robustness, safety, likelihood, or
paper-facing claim.

The author-selected direction in [Issue #7382](https://github.com/ll7/robot_sf_ll7/issues/7382)
is feasibility-first preparation toward a lexicographically ranked CMA-ES direction. That direction
is recorded as preparation context only; this note does not wire CMA-ES, choose an objective, or
authorize execution.

All numeric literals in the focused tests are local contract probes. They are not proposed #7392
variables, bounds, budgets, seeds, objectives, or held-out settings.

## Observed field and interface inventory

The following surfaces are present on the requested base. “Observed” describes existing code; it
does not imply admission to a future joint vector.

| Surface | Observed fields and units | Canonical owner and preparation boundary |
| --- | --- | --- |
| Explicit pedestrian definitions | `start`, `goal`, and trajectory waypoints are 2D map coordinates (m); `speed_m_s` is m/s; `start_delay_s` and `wait_s` are seconds; POI and role fields are authored metadata/behavior controls. | `robot_sf/nav/map_config.py` and `robot_sf/training/scenario_loader.py`. Loader overrides are the existing typed/deep-copy seam. Goal and trajectory are mutually exclusive. |
| Authored routes | Spawn/goal IDs, ordered 2D waypoints (m), and source route metadata. | `robot_sf/nav/global_route.py` plus route override handling in `robot_sf/training/scenario_loader.py`. `GlobalRoute` has no generic speed, heading, or spawn-time member. |
| Scenario simulation controls | Pedestrian density, speed multiplier, and route-spawn seed are existing scenario-level controls. | `robot_sf/sim/sim_config.py` and scenario loading. These controls affect broad scenario behavior and are not evidence that a new joint search should expose them. |
| Historical candidate seam | `Pose2D.x/y/theta`, candidate start/goal, `spawn_time_s`, pedestrian speed/delay, and integer scenario seed. | `robot_sf/adversarial/config.py`. `Pose2D.theta` serializes, but `as_waypoint()` emits only `[x, y]`; candidate start/goal target semantics therefore remain unresolved. |
| Pure overlay bridge | Nested mapping patches, candidate identity, adapter identity, provenance, and stable source/patch/materialized digests. | `robot_sf/adversarial/materialize.py:ImmutableScenarioOverlay` and `search_harness.py:MappingOverlayAdapter` / `CandidateSpecOverlayAdapter`. The bridge is data-only and does not write files or invoke runtime code. |
| Existing perturbation families | Route offsets, single-pedestrian start-delay/speed/wait/trajectory offsets, occluder timing, and density families. | `robot_sf/scenario_certification/perturbation_family_registry.py` and `perturbation_preflight.py`. These are single-family preflight interfaces, not a frozen joint vector. |

## Existing rejection and downstream interfaces

- `robot_sf/adversarial/search_harness.py` owns typed finite variables, inclusive bounds,
  restricted cross-variable predicates, objective vectors without scalarization, separated seed
  policy, rollout-budget records, and deterministic pre-adapter rejection ledgers.
- `ImmutableScenarioOverlay` snapshots and recursively freezes source, patch, and provenance data.
  An empty patch is expected to preserve the frozen source content and source/materialized digest;
  this is content identity, not Python object identity.
- `robot_sf/adversarial/feasibility_first.py` owns the existing four-check diagnostic vocabulary
  (`kinematic_reachability`, `behavioral_consistency`, `geometry_traffic`, and
  `simulator_validity`). It is a pre-execution feasibility surface, not a simulator result.
- `robot_sf/scenario_certification/v1.py` and the existing certification bridge classify geometry,
  route, kinodynamic, dynamic, and infrastructure eligibility. Certification is not an independent
  behavioral-plausibility or likelihood claim.
- `robot_sf/adversarial/runtime.py:validate_multi_ped_runtime_plausibility` owns independent
  data-level checks for config validity, scripted speed caps, obstacle clearance/intersection, and
  minimum start separation. The focused probe calls this predicate directly, without environment
  construction.
- `robot_sf/adversarial/archive.py` curates failure representatives only from executed search
  records. Preparation cannot populate an archive.
- `robot_sf/adversarial/independent_outcomes.py` owns independent outcome rows and their gating;
  `robot_sf/adversarial/disjoint_evaluation.py` owns held-out eligibility and disjoint-family/seed
  gating. `robot_sf/adversarial/held_out_preflight.py` remains an outcome-free preflight and does
  not establish independent outcomes. None of these surfaces defines the #7392 held-out count,
  seed set, or re-evaluation rule.

The adjacent [Issue #4360 search-harness note](issue_4360_search_harness.md) remains the context
owner for typed preparation and rejection accounting. Existing
[`scenario_perturbation_manifest.v1` documentation](../scenario_perturbation_manifest.md) remains
the owner for versioned single-family manifest semantics; no second schema is introduced here.

## Executable pure-data probes

`tests/adversarial/test_issue_7809_joint_preparation.py` exercises only existing owners:

1. `test_zero_or_empty_overlay_preserves_source_identity_and_immutability` covers both an empty
   patch and a patch that writes the source value (a zero-value perturbation). It checks frozen
   source/materialized content and equal digests, snapshots surviving caller mutation, and nested
   mutation failures.
2. `test_pre_adapter_infeasibility_rejection_never_calls_adapter` uses the existing
   `FiniteSearchSpaceManifest` constraint path. Every candidate receives the existing
   `constraint:minimum_clearance:unsatisfied` manifest-stage rejection; adapter validation and
   materialization counters remain zero, and `simulation_executed` remains false.
3. `test_runtime_plausibility_rejects_speed_cap_without_simulator` calls the existing runtime
   plausibility predicate on a typed map/config fixture and records its existing speed-cap error.
   It does not call `build_multi_ped_adversarial_robot_config`, create an environment, or step a
   simulator.

These are interface and fail-closed tests, not evidence that any candidate is realistic, safe,
critical, robust, or benchmark-eligible.

## Symbolic dimensionality and rollout-cost accounting

The following symbols deliberately remain variables until the complete #7382 contract is decided:

- `k_i` is the number of scalar coordinates admitted for field/actor surface `i`, and
  `D = Σ_i k_i` is the joint dimension. The admitted surfaces and every `k_i` are
  **decision-pending**.
- `A` is the number of preparation/search arms, `S` the number of search seeds, and `N` the
  candidate budget per arm and seed. `A`, `S`, and `N` are **decision-pending**. Existing
  random/quasi-random preparation names are capabilities, not a selected scientific baseline.
- `R` is rollouts per candidate and `H` is the maximum steps per rollout. Both are
  **decision-pending**; `RolloutBudget` records such fields but does not consume them here.
- `Q` is the number of manifest/adapter/plausibility checks applied to one candidate. Its exact
  contents are **decision-pending**.

The auditable preparation and execution counts are therefore:

```text
proposal_rows       = A × S × N
declared_rollouts   = proposal_rows × R
step_upper_bound    = declared_rollouts × H
pure_prepare_cost   = O(proposal_rows × (D + Q))
rollout_cost        = C_reset + H × C_step + C_finalize
total_rollout_cost  = declared_rollouts × rollout_cost
```

If a later contract has a separate held-out arm, its rows must remain explicit rather than being
silently folded into search rows:

```text
evaluation_rows = (S_search × N_search × R_search)
                + (S_held_out × N_held_out × R_held_out)
```

The values and bounds for all symbols, the objective vector/order and scalarization policy, novelty
rule, stop rule, seed partition, held-out family/ID rules, and CMA-ES configuration are
**decision-pending**. No formula above authorizes a rollout or campaign.

## Fail-closed blockers

### Heading

Heading is not currently a generic admitted scenario field. `Pose2D.theta` is candidate metadata
that `as_waypoint()` drops; robot initial heading is derived from the first route segment, while
pedestrian heading is derived from current velocity in the simulator. Until an actor target,
runtime binding, units/range, and effective-payload test are approved, a heading coordinate is
**absent/inert and rejected fail-closed**. This packet does not add a heading adapter.

### Timing

Timing has heterogeneous effectiveness. Explicit `single_pedestrians.start_delay_s` is a loader
field. The candidate route payload can emit `ped_routes.spawn_time_s`, but `GlobalRoute` and route
coercion do not retain a generic route timing member. In template pedestrian mode,
`pedestrian_delay_s` is provenance-only because the loader permits waits only on explicit
trajectories. A future timing coordinate must prove an effective runtime payload change for its
bound actor; metadata-only timing is **rejected fail-closed**. This packet does not choose a timing
surface or alter the adapter.

Other unresolved blockers include the candidate start/goal actor target, the admitted plausibility
prior, and the distinction between certification eligibility and behavioral plausibility. A
criticality or objective value must not waive an independent rejection.

## Ownership boundaries and next decision

- [Issue #7315](https://github.com/ll7/robot_sf_ll7/issues/7315) owns the feasibility-first
  scenario-search contract and validation fixtures.
- [Issue #7340](https://github.com/ll7/robot_sf_ll7/issues/7340) owns a real versioned manifest and
  any approved execution/campaign path; its diagnostic artifacts are not a source of #7392 values.
- [Issue #1433](https://github.com/ll7/robot_sf_ll7/issues/1433) is historical design guidance,
  not a new implementation owner. Its old bounds, budgets, seeds, and objectives cannot be copied
  into this deferred contract.
- The complete #7392/#7382 author decision must freeze the variable/actor map, bounds, constraints,
  objective/order, budget, seeds, held-out rules, and effective heading/timing semantics before a
  separate implementation worktree wires any optimizer or execution path. The existing #7031
  governance gate remains outside this preparation slice.

## Validation boundary

The intended low-risk proof path is:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest -q \
  tests/adversarial/test_issue_7809_joint_preparation.py \
  tests/adversarial/test_search_harness.py
scripts/dev/run_worktree_shared_venv.sh -- uv run ruff check \
  tests/adversarial/test_issue_7809_joint_preparation.py \
  tests/adversarial/test_search_harness.py robot_sf/adversarial
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/dev/check_docs_evidence_integrity.py
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/dev/check_docs_evidence_integrity.py --full
```

No command in this slice should submit or run a simulator, planner, optimizer, campaign, benchmark,
or SLURM job. Any later native, adapter, fallback, degraded, unavailable, invalid, or failed rows
must remain separately classified; none are created by this note or its tests.
