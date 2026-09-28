# Issue #9764: social-force pair-kernel versioning

## Goal and scope

Reproduce the angle-wrap defect at a source-pinned head, keep historical behavior as the default,
and expose the corrected behavior as an explicit kernel version for the planner and simulator.
This plan covers the kernel selector, compatibility tests, and the bounded paired diagnostic only.
It does not authorize changing the 0.0.7 inputs or treating the wrapped kernel as a 0.0.8 release
input without the #9668 row-equivalence gate.

## Evidence and stop rules

The issue's 2026-09-28 source-pinned reproduction and the current implementation are the authority.
The minimum implementation proof is rotation equivariance for both kernel implementations, legacy
default compatibility, explicit selector propagation, and byte-stable metadata when no selector is
supplied. The paired scenario sample is diagnostic only; a release-row mismatch or an unverified
import/config identity stops release use. Keep raw run artifacts outside Git with checksums and
classify them as `durable-required-private` until a repository-approved promotion route exists.

## Steps and status

1. Preserve `legacy_unwrapped_v1` as the default and add opt-in `wrapped_v2` to both kernels. Done
   at source commit `450ca42b7b98026949992284d37520322d14ed82`.
2. Test explicit selection and omission behavior at planner, simulator, and metadata boundaries.
   Focused tests passed on the implementation head.
3. Run matched diagnostic scenarios `classic_head_on_corridor_medium` and
   `classic_group_crossing_medium`, seeds 111–113, with traces and force recording. Done; 6/6
   terminal outcomes matched, while traces and several metrics changed. Raw evidence and checksums
   are retained in the author's private host store under logical bundle key
   `issue9764-social-force-kernel-paired-diagnostic-450ca42b`.
4. Review the release consequence with #9668: do not select `wrapped_v2` for the release until the
   required old-metric equivalence gate is run on the full release row set and passes, or the author
   updates that release contract.
5. Finish exact-head PR review, hosted checks, and repository readiness before merge.

## Risks and handoff

The correction measurably changes recorded trajectories and metrics in the bounded sample, so the
sample does not establish equivalence or a broad performance effect. The 0.0.7 scenario matrix and
shared campaign config remain unchanged. Preserve local diagnostic artifacts and the worktree until
the PR handoff and any requested #9758 follow-up are recorded.
