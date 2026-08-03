<!-- AI-GENERATED (robot_sf#6640, Gate 0) - NEEDS-REVIEW -->

# Issue #6640 — Gate 0 post-hoc feasibility audit (collision-envelope radius sensitivity)

This note records **Gate 0** of the maintainer-approved radius-sensitivity campaign (parent #6600;
validity study #3207). Gate 0 inspects the frozen `0.0.3.post1` release episode rows and the metric
contract and emits a machine-readable decision listing which radius-sensitivity outcomes can be
re-derived from retained fields and which require replay. It runs **no production compute**, changes
**no** frozen metric semantics, release config, or manifest, and establishes **no** planner ranking,
radius-sensitivity result, or paper-facing claim.

## Plain-language summary

The campaign asks whether planner rankings and scenario-family readings are stable when the robot
collision-envelope (planning proxy) radius changes across `0.5 m`, `0.8 m`, and the `1.0 m` release
baseline. Gate 0's job is to decide, *before any replay*, which outcomes the frozen release rows can
answer on their own.

**Answer:** none under the frozen release provenance. Because the radius changes both the simulator
collision geometry and planner behaviour, each radius arm produces a *different trajectory*, so any
metric computed on that trajectory differs across arms and cannot be recovered from the retained
`1.0 m` baseline rows. The release config also does not retain the effective robot/pedestrian radius,
and its scenario-matrix checksum does not pin the referenced map asset bytes. Therefore even the
tempting parameter and static-map margin diagnostics remain replay-required until those provenance
gaps are closed. This is a fail-closed boundary, **not** a radius sweep.

## The decision

Canonical machine-readable artifact:
[radius_sensitivity_gate0_audit_issue_6640.json](radius_sensitivity_gate0_audit_issue_6640.json)
(schema `radius_sensitivity_gate0_decision.v1`). Regenerate or re-validate with:

```bash
uv run python scripts/benchmark/build_radius_sensitivity_gate0_decision.py \
  --output docs/context/radius_sensitivity_gate0_audit_issue_6640.json
uv run python scripts/benchmark/build_radius_sensitivity_gate0_decision.py \
  --validate docs/context/radius_sensitivity_gate0_audit_issue_6640.json
```

The decision classifies **24** outcomes: **0 re-derivable**, **24 replay-required**.

- Re-derivable: none — the exact effective radius and map asset provenance are not retained/pinned.
- Replay-required: every listed outcome, including the
  radius/threshold metadata and static-map margin diagnostics, plus every
  radius-aware clearance family (`human_collisions`, `near_misses`, `min_clearance`,
  `mean_clearance`), the fixed-threshold counts (`wall_collisions`, `agent_collisions`), the
  radius-independent geometry metrics (`clearing_distance_min`, `min_distance`, …), binary
  `success`, aggregate collision counts, simulator obstacle/pedestrian contact, executed-traversal
  feasibility, planner behaviour and rankings, SNQI, scenario-family transitions, and all
  kinematic/efficiency metrics.

## Why the boundary is here

- **Radius binds the clearance family.** In `robot_sf/benchmark/metrics.py`,
  `clearance[t,k] = center_distance[t,k] − (robot_radius + ped_radius)`. `human_collisions` and
  `near_misses` are *threshold counts* over the per-timestep clearance distribution; the aggregate
  frozen rows retain only the counts, not the distribution, so reclassifying them at a new radius
  requires the full replay trajectory. `min_clearance`/`mean_clearance` shift linearly with the
  radius *for one fixed trajectory*, but the cross-arm trajectory differs, so the cross-arm value is
  not re-derivable from the baseline aggregate.
- **`wall_collisions`/`agent_collisions` use the fixed `COLLISION_DIST` (`0.25 m`) centre-distance
  threshold** and do not subtract the radius in the metric formula — but the radius still binds the
  simulator collision geometry during trajectory generation, so these counts remain
  trajectory-dependent.
- **Threshold reclassification requires replay.** `robot_sf/benchmark/threshold_sensitivity.py`
  recomputes near-miss/comfort counts from full replay trajectories (`replay_steps`/`replay_peds`),
  never from aggregate frozen rows.
- **Static geometry is provenance-blocked here.** A planner-free oracle (#5574) could measure the
  doorway gap and reparameterise the swept envelope (`margin = gap − 2 × radius`) without replaying
  a learned planner, but this release manifest hashes only the scenario matrix, not its included map
  asset bytes. The exact geometry must therefore be recovered before that margin is re-derivable.
  Even a positive static margin would not reconstruct scripted-traversal or planner feasibility —
  the #5574 `0.5 m` probe reclassifies the narrow doorway as solvable yet its scripted traversal
  still collided.

## Findings and risks for later gates

- **Radius-default inconsistency (Gate 1 input).** The collision-envelope radius default is not
  uniform across the metric contract: `metrics.py` `EpisodeData` defaults to
  `robot_radius=1.0 m / ped_radius=0.4 m`, while `runner.py` defaults to
  `DEFAULT_BENCHMARK_ROBOT_RADIUS_M=0.3 m / 0.35 m`. The frozen release config does not
  manifest-declare the radius. Gate 1 (binding canary) must confirm the per-row effective radius
  before any reclassification, and must prove the radius binds consistently to simulator collision
  geometry, obstacle/pedestrian contact, feasibility/oracle calculations, metric metadata, and
  planner inputs.
- **Frozen provenance gaps are Gate 0 blockers for post-hoc reclassification.** The effective
  per-row robot/pedestrian radius is not manifest-declared, and the scenario matrix checksum does
  not checksum the referenced map assets. Gate 1 or a provenance-preserving replay must close both
  gaps before either diagnostic can be reclassified.
- **Success has two gates.** The frozen collision reconciliation warns the bundle lacks
  `reached_goal_step`/`horizon` inputs to recompute the success-timing gate, so binary `success` is
  replay-required on both its collision and timing components.
- **Do not infer a sweep from the static-margin reclassification alone** (stop condition #5). Gate 2
  (production sweep) and Gate 3 (analysis) are required for any ranking or family-level verdict.

## Scope and ownership

- Module: `robot_sf/benchmark/radius_sensitivity_gate0_audit.py` (pure, deterministic, no simulation).
- Test: `tests/benchmark/test_radius_sensitivity_gate0_audit.py`.
- Build/validate CLI: `scripts/benchmark/build_radius_sensitivity_gate0_decision.py`.
- This is a diagnostic decision record; all current outcomes are replay-required and none is
  radius-sensitivity evidence.
