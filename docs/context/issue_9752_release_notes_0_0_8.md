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

## Pending hybrid correction

The historical hybrid v3 flat-observation path retains the 0.0.7 frame defect.
The 0.0.8 release is blocked until a corrected, explicitly versioned hybrid
path is selected and validated for each affected release arm. Historical v3
bytes remain available for the mandatory 0.0.7 comparison; an attributed
result change from the correction is permitted by the #9668 ruling.

## Hybrid v4 development split protocol (#9748)

Before any hybrid v4 tuning, the four pre-registered development-only scenario
parameter variants in
`configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml` are frozen
under distinct `issue_9748_dev_*` identities. The tuning config uses only seeds
1001–1030. The sealed 0.0.8 seeds (D-049), retired seeds 111–140, and
frozen release scenario identities are held out. The existing v4 fast-progress and continuous candidate configs are
inputs to this development split; the eventual release roster remains pending
the #9751 decision.

The split definition and validator establish the protocol boundary. No tuning
run, frozen v4 result, or release-evaluation claim is recorded by this note;
the evidence status is pending actual development runs and their structured
tuning log.

## Evidence boundary

The implementation is supported by focused adapter tests, lint/format checks,
and bounded metamorphic checks when those checks are run. A full benchmark
campaign, campaign comparison, and per-arm acceptance decision are intentionally
outside this draft.
