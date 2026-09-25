# Issue 9733: metamorphic invariance tests

## Goal and scope

Add bounded tests for occupancy-grid resolution, mirrored deterministic episodes,
pedestrian removal, planner physical units, and seeded replay. Keep every case in
the default tier (no `slow` marker; PR shards skip slow tests). Change tests and
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

The refute review (FIX on `ba4ac942`) found eight undetected mutations. The
follow-up runs mirror, removal and replay on the release `social_force`, `orca`
and hybrid v3 arms through the map-runner policy builder, samples every replay
input from its seed with negative controls, audits the release campaign configs
with a value-pinned violation ledger, and adds a 90-degree rotation relation
because a reflection-type sign error commutes with every mirror. The hybrid v3
arm is not reflection-equivariant at trace level (a discrete near-tie flips);
that is a strict expected failure pending its own issue, while its outcome
relation passes.

## Recovery and handoff

The task worktree is `issue-9733-metamorphic-invariance` on branch
`test/issue-9733-metamorphic-invariance`. Preserve a dirty tree and any local
validation artifacts if interrupted. No output artifact is scientific evidence.
