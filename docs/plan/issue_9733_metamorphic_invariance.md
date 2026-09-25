# Issue 9733: metamorphic invariance tests

## Goal and scope

Add bounded tests for occupancy-grid resolution, mirrored deterministic episodes,
pedestrian removal, planner physical units, and seeded replay. Keep the normal
test tier small and put broader episode matrices behind `slow`. Change tests and
their documentation only; do not change spawn logic, `socnav_social_force.py`, or
the hybrid planner.

## Evidence and claim boundary

The issue contract is ll7/robot_sf_ll7#9733, a child of #9730. Existing
`tests/metamorphic/` fixtures and planner/drive configuration are the executable
contract. These tests are implementation checks, not benchmark success or
scientific evidence. Expected failures must be strict, narrow, and reference
the observed upstream defects #9724 and #9726.

## Steps and ownership

1. Add independent tests for grid resolution, symmetry and removal, and units
   and replay in separate files.
2. Review the integrated tests for actual planner coverage, tolerance rationale,
   narrow expected failures, and default-tier cost.
3. Run focused tests, the fast metamorphic tier, lint/format, and the repository
   PR readiness gate. Keep failing broad sweeps distinct from focused proof.
4. Commit, push, and open a PR linked to #9733. Do not merge it.

## Decisions and risks

The suite must assert transformed planner behavior, not only a fixture's own
values. Environment reset isolation and JSONL state replay already have tests;
seeded action re-simulation is a separate target here. Cross-host equality has
an explicit numeric tolerance because float representation may vary.
The grid probe also found a DWA resolution counterexample, tracked as #9740;
retain it as a narrow strict expected failure rather than masking the command
change with a larger tolerance. This is not evidence of benchmark performance.

## Recovery and handoff

The task worktree is `issue-9733-metamorphic-invariance` on branch
`test/issue-9733-metamorphic-invariance`. Preserve a dirty tree and any local
validation artifacts if interrupted. No output artifact is scientific evidence.
