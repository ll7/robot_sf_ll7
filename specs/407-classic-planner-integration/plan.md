"""
Purpose: Plan for integrating classic global planning with local SocNav planners and
establishing evaluation artifacts for baseline comparisons.
"""

# Classic Planner Integration Plan (Spec 407)

## Context, Goals, Non-goals
**Context**: Local SocNav planners (Sampling/Social-Force/ORCA/SA-CADRL) are reactive
and struggle in tight geometry or dense traffic. The classic global planner already
exists in `robot_sf/planner/` but is not wired as a waypoint provider for these
local policies during baseline evaluation.

**Goals**
- Provide a **hybrid baseline**: classic global path + local SocNav planner for
  short-horizon avoidance.
- Make the integration available in **render + benchmark tooling** so we can
  compare baselines in the same workflow.
- Add **unit-level tests** to validate ORCA (and hybrid) behavior in controlled
  scenarios.

**Non-goals**
- Full RL training changes (handled in separate training specs).
- Changes to the core scenario definitions or map geometries (unless needed for
  explicit bug fixes).

## Constraints & assumptions
- Preserve existing default behavior in `render_scenario_videos.py` and benchmark
  runners.
- Keep the local planner interface stable (return `(v, w)` velocity commands).
- Avoid large new dependencies by default; optional integrations should degrade
  gracefully.

## Chosen approach
1. **ORCA adapter hardening**: implement a closer-to-ORCA velocity-constraint solve
   using linear-program-style half-plane constraints (still in-process; no external
   solver required).
2. **Hybrid policy**: add a wrapper that uses the classic planner’s waypoint path
   to compute a **short-horizon preferred velocity**, then feeds it into a local
   planner (e.g., ORCA or Social-Force).
3. **Tooling integration**: extend render + benchmark config to select the hybrid
   baseline with a simple policy name.

## Contracts (APIs/data, error modes)
- Local planner `plan()` continues to return `(v, w)`.
- Hybrid policy exposes `act(obs)` and uses env config + map definition to access
  the classic planner route when enabled.
- Fail-safe behavior: if the classic planner route is unavailable, fall back to
  the local planner’s default goal heading.

## Test plan
- Unit tests:
  - ORCA avoidance tests for head-on, lateral, and far-ped cases.
  - Hybrid policy test: waypoint available ⇒ preferred velocity direction deviates
    from straight-to-goal when a wall blocks the direct line.
- Integration smoke:
  - One-step env reset/step with hybrid planner to ensure spaces are valid and
    observation mode is respected.

## Rollout & back-compat
- Ship hybrid planner behind explicit policy flags; default behavior unchanged.
- Add documentation to the spec folder and extend the scenario video manifest
  to record hybrid settings.

## Metrics/observability
- Record stop reason + collision/success in summary manifests.
- Optional: log min distance to obstacles/peds per step for deeper triage.

## Open questions / follow-ups
- Should the classic planner path be re-computed every step or only at reset?
- Should obstacle inflation be dynamic for local planner clearance?
- Do we want optional integration with external ORCA solvers (RVO2) later?

## Status (2026-01-22)
- Completed: ORCA constraint-based solver in `robot_sf/planner/socnav.py`.
- Completed: SocNav planner support in `scripts/tools/render_scenario_videos.py`.
- Completed: Failure-frame extraction tool in `scripts/tools/extract_failure_frames.py`.
- Completed: ORCA unit tests in `tests/test_socnav_planner_adapter.py`.
- In progress: hybrid classic planner + local policy wiring (not implemented yet).
- Latest evaluation: ORCA sweep (129 videos) stored under
  `output/recordings/scenario_videos_classic_interactions_francis2023_socnav_orca_20260122_175413`.
