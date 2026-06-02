# Issue #2104 Component Ablation Pilot

Status: current diagnostic context for issue
[#2104](https://github.com/ll7/robot_sf_ll7/issues/2104).

## Scope

Issue #2104 asks for a compact planner-component ablation matrix for leading hybrid candidates.
This note records the first bounded slice: a retrospective grouped-component pilot over already
tracked S10/H500 candidate evidence.

The manifest is:

- `configs/policy_search/ablation_manifests/issue_2104_component_ablation_pilot.yaml`

The compact evidence bundle is:

- `docs/context/evidence/issue_2104_component_ablation_pilot_2026-06-02/README.md`

## Result

The retrospective pilot is useful as mechanism triage, not as one-factor proof.

- Static escape/recenter versus fast progress: success improves by about `+0.077`, near misses are
  nearly unchanged, normalized runtime improves, and collision increases slightly by about `+0.006`.
- Continuous static checks versus grouped static escape: success improves by about `+0.013`,
  collision drops by about `-0.010`, and near misses drop by about `-2.97`, with a small normalized
  runtime cost.
- Scenario-adaptive ORCA selection versus grouped static escape: aggregate effect is small on this
  slice, but the comparison still helps identify where selector behavior matters.
- Collision-guard v2 versus v1: no aggregate effect appears in the preserved S10/H500 table.

## Claim Boundary

This is `diagnostic_only` evidence. It does not close the full issue #2104 acceptance criteria
because the historical rows do not independently toggle every component. In particular, route
guidance, recentering, speed-envelope changes, static escape, continuous static checks, ORCA
selection, and guard overlays remain partly confounded.

The next executable step is a purpose-built one-factor manifest on a smaller scenario slice with
identical seeds. That follow-up should separate one-factor rows from grouped rows before making
stronger component-contribution claims.
