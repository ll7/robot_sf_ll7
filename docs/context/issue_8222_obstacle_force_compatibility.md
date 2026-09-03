# Issue #8222: Obstacle-Force Compatibility Boundary

> Status: current policy and implementation-context note. This records the bounded option-B
> compatibility/documentation ruling for [Issue #8222](https://github.com/ll7/robot_sf_ll7/issues/8222)
> at current-main SHA `6e15eab1c0013486daccef9c239737c292ec3ab4`. The runtime selector described
> below is implemented by the separately scoped follow-up [Issue #8277](https://github.com/ll7/robot_sf_ll7/issues/8277).

## Decision

Preserve the current obstacle-force law for frozen or unversioned historical campaigns. A corrected
law may be implemented only as an explicit opt-in and remains domain-gated until the runtime
follow-up acceptance boundary below is complete. No benchmark, safety, social-behavior, physical,
or paper-facing claim follows from this note or from the current implementation inspection.

The runtime identifiers are:

- `legacy_shifted_gradient_v1` — the preserved historical law and default for missing or unversioned inputs.
- `surface_distance_unit_normal_v2` — the corrected surface-distance/unit-normal law, available only by explicit opt-in.

These names are runtime compatibility selectors, not released model versions and not evidence of
physical, safety, social-behavior, benchmark, or paper-facing validation.

The #8277 implementation resolves unknown explicit selectors fail-closed and emits
`obstacle_force_law_metadata.v1` with the selected law, site, geometry convention, radius/offset
convention, compatibility mode, and explicit `enabled`/`applied` state. The fast-pysf map-segment
site retains its point, endpoint, and segment branches; the planner site retains its
occupancy-cell-center point geometry. The selectors therefore version dispatch without silently
unifying the two sites. Map-runner episode records persist the site payloads under
`algorithm_metadata.obstacle_force_law` using `obstacle_force_law_runtime_record.v1`; the ordinary
robot-environment reset metadata and JSONL sidecar also persist the fast-pysf payload.

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

The separately scoped runtime correction is tracked in [Issue #8277](https://github.com/ll7/robot_sf_ll7/issues/8277),
the child of [parent Issue #8222](https://github.com/ll7/robot_sf_ll7/issues/8222). The child starts
as an implementation idea and remains opt-in and domain-gated; it must, at minimum:

1. carry explicit law metadata and resolve historical frozen/unversioned inputs to
   `legacy_shifted_gradient_v1`;
2. provide analytic parity checks for point, endpoint, and segment cases;
3. check finite and monotonic behavior near contact;
4. reproduce the exact legacy law before comparing the correction;
5. run a bounded diagnostic comparison with explicit legacy/corrected labels and no claim upgrade;
6. obtain domain review before any default-law change.

The corrected behavior is opt-in only until all of these gates and the required review are complete.
No campaign rerun, default-law change, benchmark artifact, release-surface change, or historical
campaign change is authorized by this note. The child may implement opt-in compatibility selectors
and targeted tests, but those checks do not establish benchmark, physical, safety, or social-behavior
evidence.

## Evidence and handoff boundary

Evidence in this slice is limited to source/config inspection, executable parity/near-contact tests,
and documentation validation at the stated current-main boundary. It establishes neither numerical
correctness beyond those targeted checks nor physical suitability. The runtime child in
[Issue #8277](https://github.com/ll7/robot_sf_ll7/issues/8277) uses the canonical implementation
archetype with `evidence_tier: idea`; its targeted tests are diagnostic compatibility evidence only.
Local preparation drafts are disposable and are not a durable dependency of this note.

Validation for this note is limited to targeted reference inspection, `git diff --check`,
`scripts/dev/check_docs_evidence_integrity.py --full`, and
`scripts/validation/check_docs_proof_consistency.py --check-evidence-catalog`, run through the
repository's shared wrapper where available.
