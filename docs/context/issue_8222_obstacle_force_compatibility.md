# Issue #8222: Obstacle-Force Compatibility Boundary

> Status: current policy and implementation-context note. This records the bounded option-B
> compatibility/documentation ruling for [Issue #8222](https://github.com/ll7/robot_sf_ll7/issues/8222)
> at current-main SHA `578ab08b543b9370b4d7556971990d7751b0eba0`. It does not change runtime behavior.

## Decision

Preserve the current obstacle-force law for frozen or unversioned historical campaigns. A corrected
law may be implemented only as an explicit opt-in and remains domain-gated until the runtime
follow-up acceptance boundary below is complete. No benchmark, safety, social-behavior, physical,
or paper-facing claim follows from this note or from the current implementation inspection.

The proposed identifiers are design names only until runtime work lands:

- `legacy_shifted_gradient_v1` — proposed identifier for the preserved historical law.
- `surface_distance_unit_normal_v2` — proposed identifier for a corrected surface-distance/unit-normal law.

These names must not be interpreted as existing runtime metadata, released versions, or evidence of
validation.

## Current implementation boundary

The two obstacle-force sites must not be conflated:

| Site | Current input and geometry convention | Current radius/offset convention |
| --- | --- | --- |
| `fast-pysf/pysocialforce/forces.py` (`obstacle_force`, called by `all_obstacle_forces`) | Map obstacles are world-space line segments represented by endpoints plus an orthogonal vector. The kernel selects the point, endpoint, or segment-intersection branch. | The force distance subtracts `ped_radius` from the selected geometric distance and clamps it to `1e-5`; the `ObstacleForce` caller also raises its force threshold by `agent_radius * sigma`. |
| `robot_sf/planner/socnav_social_force.py` (`SocialForcePlannerAdapter._compute_obstacle_force`) | Occupied grid cells become cell centers using `origin + (index + 0.5) * resolution`, are transformed from ego to world coordinates, and are evaluated as degenerate point obstacles in a vectorized reduction. | Each cell receives `0.5 * sqrt(2) * resolution * social_force_obstacle_radius_scale`; the effective `ped_radius` is the robot radius plus that cell radius. The planner site therefore uses a cell-center offset and cell-derived radius, not the fast-pysf segment convention. |

Relevant configuration and integration paths include:

- `fast-pysf/pysocialforce/config.py` and `fast-pysf/pysocialforce/simulator.py` for the pedestrian
  obstacle-force configuration and map-obstacle wiring;
- `robot_sf/sim/simulator.py` for construction and force selection;
- `robot_sf/planner/socnav_social_force.py` for the planner-side occupancy extraction and force;
- `configs/algos/social_force_example.yaml`, `configs/algos/social_force_planner_sanity.yaml`, and
  `configs/algos/lidar_social_force_issue_1660.yaml` as representative social-force configuration
  surfaces.

The upstream implementation and any derivation may explain provenance or motivate a proposed
correction, but neither is intent evidence, physical validation, benchmark evidence, or safety
validation.

## Runtime follow-up acceptance boundary

The separately scoped runtime correction issue is pending until the parent creates it; the pending
follow-up is tracked from [parent Issue #8222](https://github.com/ll7/robot_sf_ll7/issues/8222), and
no child issue number is assigned here. A future implementation child must, at minimum:

1. carry explicit law metadata and resolve historical frozen/unversioned inputs to
   `legacy_shifted_gradient_v1`;
2. provide analytic parity checks for point, endpoint, and segment cases;
3. check finite and monotonic behavior near contact;
4. reproduce the exact legacy law before comparing the correction;
5. run a bounded diagnostic comparison with explicit legacy/corrected labels and no claim upgrade;
6. obtain domain review before any default-law change.

The corrected behavior is opt-in only until all of these gates and the required review are complete.
No campaign rerun is authorized by this note. The stop boundary is documentation and compatibility
capture only: no runtime code, config default, test, benchmark artifact, release surface, or
historical campaign is changed or rerun here.

## Evidence and handoff boundary

Evidence in this slice is limited to source/config inspection at the stated current-main SHA and
documentation validation. It establishes neither numerical correctness nor physical suitability.
The future runtime child should use the canonical implementation archetype with `evidence_tier: idea`
until executable parity and diagnostic evidence exist. Its prepared body is retained outside all
worktrees at `/home/luttkule/git/robot_sf_ll7/.git/codex-agent-runs/artifacts/issue-8222-20260902/runtime-followup-body.md` in the
agent handoff environment; it is not published by this task.

Validation for this note is limited to targeted reference inspection, `git diff --check`,
`scripts/dev/check_docs_evidence_integrity.py --full`, and
`scripts/validation/check_docs_proof_consistency.py --check-evidence-catalog`, run through the
repository's shared wrapper where available.
