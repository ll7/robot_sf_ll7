# Robot SF 0.0.8 draft release notes

**Status:** draft text for issue #9752. This note does not contain campaign
results or per-arm release acceptance claims.

## Observation-frame correctness

- Risk-DWA now converts SocNav pedestrian velocities from robot-ego frame to
  world frame before TTC and constant-velocity rollout scoring for nested and
  flat observations.
- Guarded PPO now performs the same conversion before its world-frame safety
  rollout. The PPO policy input remains in the ego frame used by its training
  contract.
- The flat-observation frame contract and current release-arm audit are recorded
  in `issue_9752_flat_observation_frame_contract.md`.
- Direct flat-observation regressions and deterministic rotation coverage were
  added for the affected safety paths.

## Compatibility and known limitation

Hybrid v3 flat-observation velocity pass-through remains byte-for-byte
unchanged for comparison with the 0.0.7 behavior. That path retains the known
0.0.7 frame defect and is not presented as fixed by this draft.

## Evidence boundary

The implementation is supported by focused adapter tests, lint/format checks,
and bounded metamorphic checks when those checks are run. A full benchmark
campaign, campaign comparison, and per-arm acceptance decision are intentionally
outside this draft.
