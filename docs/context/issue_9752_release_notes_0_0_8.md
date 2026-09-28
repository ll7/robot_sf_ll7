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
- Structured and flat observation regressions pin the same ego-to-world
  conversion. The deterministic Risk-DWA release arm also has a closed-loop
  scene-rotation test; Guarded PPO is tested at the safety-input boundary because
  its learned primary policy is not an exact trace oracle.

## Compatibility and known limitation

Hybrid v3 flat-observation velocity pass-through remains unchanged for
comparison with 0.0.7. That path retains the known
0.0.7 frame defect and is not presented as fixed by this draft.

## Evidence boundary

The implementation is supported by focused adapter tests, lint/format checks,
and bounded metamorphic checks when those checks are run. A full benchmark
campaign, campaign comparison, and per-arm acceptance decision are intentionally
outside this draft.
