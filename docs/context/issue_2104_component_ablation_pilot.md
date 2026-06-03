# Issue #2104 Component Ablation Pilot (2026-06-02)

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

Issue [#2170](https://github.com/ll7/robot_sf_ll7/issues/2170) now freezes that pre-execution
contract; see
[issue_2170_one_factor_hybrid_component_manifest.md](issue_2170_one_factor_hybrid_component_manifest.md).
It remains proposal/pre-execution evidence until the planned candidate rows are implemented and the
compact matrix is run with durable effect-size outputs.

Update 2026-06-03: Issue [#2174](https://github.com/ll7/robot_sf_ll7/issues/2174) and Issue
[#2176](https://github.com/ll7/robot_sf_ll7/issues/2176) executed the h80 staged pilot. See
[issue_2174_one_factor_ablation_pilot.md](issue_2174_one_factor_ablation_pilot.md) and
[issue_2176_remaining_one_factor_h80.md](issue_2176_remaining_one_factor_h80.md). The clean h80
rows did not move success, collision, or near-miss rates; the selector-only row is partial because
3/18 jobs failed, consistent with a local missing `rvo2` dependency. The parent research question
therefore remains open for h500 execution or a selector-row rerun after the dependency is available.

Issue [#2178](https://github.com/ll7/robot_sf_ll7/issues/2178) completes that selector rerun after
syncing the `orca` extra. See
[issue_2178_selector_orca_extra_h80.md](issue_2178_selector_orca_extra_h80.md). The corrected h80
selector row writes 18/18 jobs with zero failures and stays flat on success, collision, and
near-miss rate.
