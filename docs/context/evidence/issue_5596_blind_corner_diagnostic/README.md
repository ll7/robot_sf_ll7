<!-- AI-GENERATED (robot_sf#5596, 2026-07-14) - NEEDS-REVIEW -->

# Issue #5596 — Blind-corner zero-success diagnostic

Diagnostic-only analysis requested by
[issue #5596](https://github.com/ll7/robot_sf_ll7/issues/5596). It explains why
`francis2023_blind_corner` stays zero-success for the scripted traversal at both the
nominal 1.0 m and reduced 0.5 m collision-envelope radii, even though the retained
feasibility packet (#5574) marks the geometry `hard_but_solvable`.

This directory contains a **reproducible machine-readable report** plus the canonical
command that regenerates it. It is diagnostic-only evidence: it does **not** promote the
cell to `infeasible-as-configured`, change a benchmark denominator, or support a
planner-ranking claim.

## Canonical command

```bash
python scripts/tools/run_blind_corner_diagnostic_issue_5596.py \
  --output docs/context/evidence/issue_5596_blind_corner_diagnostic/blind_corner_diagnostic.json
```

Determinism check (the report is reproducible): run the command twice and compare the
normalized JSON / checksums. The rollout seed is pinned (blind-corner manifest seed 219).

## What the report contains

- `oracle_verdict`: the existing planner-free oracle (no learned planner) driven with the
  `goal` scripted controller at the nominal and reduced envelope radii. Reported margins:
  corridor-vs-envelope width, kinematic lower-bound completion steps vs horizon.
- `route_follow_intervention_verdict`: a fail-closed placeholder for the intended
  **route-following** lane. The canonical map runner does not yet support carrying robot
  pose and waypoint goals across sub-episodes, so this lane reports `blocked` rather than
  pretending that reset-per-waypoint episodes are a continuous intervention. Stateful
  adapter work is tracked in [#5636](https://github.com/ll7/robot_sf_ll7/issues/5636).
- `straight_line_vs_route_clearance`: straight-line (goal beeline) vs certified-route
  obstacle clearance for each envelope radius.
- `mechanism`: bounded classification of the three competing explanations from the issue.

## Findings (reproducible, provenance-pinned)

Provenance: `scenario_manifest = configs/scenarios/francis2023.yaml`, cell
`francis2023_blind_corner`, rollout seed 219, source commit captured by `git_hash` in
the episode records emitted by the oracle runner.

1. **Geometry is `hard_but_solvable`, not infeasible.** The certifier reports
   `classification = hard_but_solvable`, `benchmark_eligibility = eligible`,
   `min_static_clearance_m = 1.0` (nominal) / `1.5` (reduced), corridor-vs-envelope
   margin `+2.0 m` / `+3.0 m`, and a kinematic lower-bound completion of ~192 steps
   (nominal) / ~190 steps (reduced) — far inside the 400-step horizon. So the route is
   not horizon- or envelope-bound, and not geometrically infeasible.

2. **The route-following intervention is unavailable and therefore cannot discriminate
   the mechanism.** The `goal` lane is diagnostic input, but the canonical map runner has
   no supported contract for injecting waypoint goals while preserving robot pose across
   sub-episodes. The report records `route_follow_intervention_status=blocked` at both
   radii, so no route-follow success or failure is interpreted.

3. **The certifier's own A* path clips the inner corner in the retained diagnostic trace.**
   The straight goal beeline has
   ~0 m clearance (expected — it cuts the corner). But the **certified route** also has
   ~0 m clearance: the certifier computes `min_static_clearance_m = 1.0` on the
   *authored 4-waypoint route line* (which bends around the corner at `(27, 24.5)` and
   stays clear at ~1.15 m), yet the A* `planned_path` it returns cuts **diagonally**
   across the inner corner (vertices ~`(23.25, 25.25)`→`(25.25, 23.25)`). A radius-1.0 m
   circle collides on 6 of the 73 planned vertices (4 at radius 0.5 m). The
   `inflated_collision_free_path = True` claim is geometrically inconsistent with the
   path geometry the oracle then consumes.

### Mechanism status

The retained clearance trace is consistent with a **certifier/planner-path artifact** —
the inflated A* path does not honor the clearance the certifier attributes to the route
line — but the mechanism remains **unresolved** until the stateful route-follow adapter
in [#5636](https://github.com/ll7/robot_sf_ll7/issues/5636) is implemented and rerun.
The scripted-controller explanation is not promoted because the intervention is blocked.

Explanation **#3 (scenario runtime / map interpretation drift)** is `not_established`:
the run uses the same scenario manifest, robot kinematics, horizon, and map as the
release surface, with no deliberate diagnostic change.

## Stop rule / limitations

Per the issue, the analysis stays diagnostic-only. It does **not** relabel the
blind-corner cell or change any campaign denominator. The certifier inconsistency is
reported as a mechanism finding; promoting the cell to `infeasible-as-configured`
requires a separate reviewed issue with benchmark-grade evidence (e.g. a corrected
planner-path clearance check).

## Files

- `blind_corner_diagnostic.json` — generated report (regenerate with the command above).
- `README.md` — this file.
