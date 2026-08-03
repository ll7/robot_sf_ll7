# Issue #6645: narrow-doorway radius-binding audit

Status: audited on 2026-08-03 from RobotSF `origin/main` at `eae02066a8fdcf89fae9986c7539e8a4b66919a4`.

Issue: [ll7/robot_sf_ll7#6645](https://github.com/ll7/robot_sf_ll7/issues/6645)

## Result

The audit passes for the authored `francis2023_narrow_doorway` scenario. The doorway
opening is 2.0 m, and the nominal collision envelope is a 1.0 m radius. Therefore the
exact transverse clearance boundary is:

```text
clearance_m = gap_width_m - 2 * envelope_radius_m
             = 2.0 - 2 * envelope_radius_m
```

| Envelope radius | Envelope diameter | Continuous clearance margin |
|---:|---:|---:|
| 0.0 m | 0.0 m | 2.0 m |
| 0.5 m | 1.0 m | 1.0 m |
| 0.8 m | 1.6 m | 0.4 m |
| 1.0 m | 2.0 m | 0.0 m |

The zero-margin nominal case is intentional and is not reported as positive-clearance
geometry. The 0.8 m case has positive continuous clearance, but the planner-free grid
oracle can classify it as `infeasible_by_construction` because its conservative grid
inflation cannot find a route. That classification is retained as an oracle result; it is
not used to change the authored geometry or the continuous-width calculation.

## Source trace

- The scenario selects the authored map and leaves `robot_config` empty, so the effective
  robot setting comes from the runtime default: `configs/scenarios/single/francis2023_narrow_doorway.yaml:2-10`.
- The map defines the two doorway wall segments at `y=1..4` and `y=6..9`, the 2.0 m
  opening, and the route centerline at `y=5`: `maps/svg_maps/francis2023/francis2023_narrow_doorway.svg:10-21`.
- The authoritative collision-envelope default is `DEFAULT_ROBOT_RADIUS = 1.0` m:
  `robot_sf/common/robot_defaults.py:12-34`.
- The oracle defines `minimum_static_clearance_m` as route-to-obstacle distance minus
  robot radius and derives corridor width from the same envelope:
  `robot_sf/scenario_certification/feasibility_oracle.py:621-627`.
- The runtime canary probes simulator collision geometry, contact logic, the feasibility
  oracle, metric/output rows, and planner inputs: `robot_sf/benchmark/radius_binding_canary.py:9-31`.
  Its top-level runner requires all five surfaces to bind the declared radius:
  `robot_sf/benchmark/radius_binding_canary.py:794-809` and `:821-874`.
- The audit records the historical 0.3 m grid-rasterization default and 0.4 m planner
  fallback as non-collision defaults. Their source documents the divergence and assigns
  the benchmark-impacting change to issue #4856:
  `robot_sf/nav/occupancy_grid.py:183-186` and `robot_sf/gym_env/base_env.py:307-314`.

## Machine-readable audit

Run:

```bash
uv run python scripts/validation/audit_issue_6645_narrow_doorway_radius_binding.py \
  --out-json /tmp/issue_6645_radius_binding_audit.json
```

The runner emits `narrow_doorway_radius_binding_audit.v1`, derives the geometry from the
SVG rather than copying the 2.0 m value, checks the zero-clearance boundary, and runs the
existing `radius_binding_canary.v1` at 0.5 m, 0.8 m, and 1.0 m. The report is diagnostic
only and does not promote campaign evidence.

Focused validation on this audit branch:

```text
7 passed — tests/validation/test_audit_issue_6645_narrow_doorway_radius_binding.py
6 passed — tests/scenario_certification/test_envelope_radius_binding_canary.py
```

No map, frozen artifact, default, release, or production campaign output was changed.
No child repair issue is required by this audit. The doorway-variant implementation in
[ll7/robot_sf_ll7#6644](https://github.com/ll7/robot_sf_ll7/issues/6644) remains a separate
follow-up and must use the verified radius semantics and the oracle-first feasibility rule.

Claim boundary: diagnostic geometry and within-simulator binding audit only. This is not
benchmark evidence, a physical-footprint validation, a realism result, or a safety
guarantee.
