# Issue #9764: social-force pair-kernel versioning

## Goal and scope

Reproduce the angle-wrap defect at a source-pinned head, keep historical behavior as the default,
and expose the corrected behavior as an explicit kernel version for the planner and simulator.
This plan covers the kernel selector, compatibility tests, and the bounded paired diagnostic only.
It does not authorize changing the 0.0.7 inputs or treating the wrapped 0.0.8 candidate as
release evidence without the #9668 row-equivalence gate.

## Evidence and stop rules

The issue's 2026-09-28 source-pinned reproduction and the current implementation are the authority.
The minimum implementation proof is rotation equivariance for both kernel implementations, legacy
default compatibility, explicit selector propagation, and byte-stable metadata when no selector is
supplied. The paired scenario sample is diagnostic only; a release-row mismatch or an unverified
import/config identity stops release use. Keep raw run artifacts outside Git with checksums and
classify them as `durable-required-private` until a repository-approved promotion route exists.

## Steps and status

1. Preserve `legacy_unwrapped_v1` as the default and add opt-in `wrapped_v2` to both kernels. The
   integrated branch now also carries the separate #9758 wrapped-kernel regression recheck.
2. Test explicit selection and omission behavior at planner, simulator, and metadata boundaries.
   The focused kernel/planner/simulator/registry/metamorphic command passed 180 tests with one
   strict xfail already tracked under #9733. The candidate-config suite passed 3 tests.
3. Run matched diagnostic scenarios `classic_head_on_corridor_medium` and
   `classic_group_crossing_medium`, seeds 103–105, with traces and force recording. At source
   `ca1b5f1d33ebfa3d466f384f2636bb001dccf23f`, all 12 episode jobs completed without runner
   failures. Five pairs ended in success and one in collision, with no terminal-outcome mismatch.
   Of 4,898 compared robot-attributable force samples, 555 changed by more than `1e-9 m/s²`;
   maximum same-index robot and pedestrian trajectory differences were 0.103352465 m and
   0.675855495 m. The #9758 contact recheck remains a separate narrow test, not a safety claim.
   Raw rows, analysis, code, logs, provenance and checksums are under
   `.git/codex-agent-runs/issue-9764-social-force-final/paired-diagnostic-current-ca1b5f1/`;
   `SHA256SUMS` verified. The tracked evidence note records the principal artifact hashes.
   Earlier 111–113 outputs remain quarantined and uncited.
4. Bind `wrapped_v2` in a distinct 0.0.8 candidate config. Its three identity and seed-split tests
   pass. The historical 0.0.7 scenario matrix remains byte-identical at SHA-256
   `03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c`. Keep the candidate
   outside release evidence until the required old-metric equivalence gate passes on the full
   release row set, or the author updates that contract.
5. Finish exact-head adversarial review, hosted checks, and repository readiness before merge.

## Risks and handoff

The corrected kernel can change trajectories and metrics. The bounded replacement diagnostic
cannot establish equivalence or a broad performance effect. The 0.0.7 scenario matrix and
shared campaign config remain unchanged. Preserve local diagnostic artifacts and the worktree until
the PR handoff and any requested #9758 follow-up are recorded.
