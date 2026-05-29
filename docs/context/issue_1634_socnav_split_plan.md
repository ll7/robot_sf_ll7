# Issue #1634 SocNav Module Split Plan

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1634>

## Goal

`robot_sf/planner/socnav.py` is still the compatibility import surface for SocNav-family planners,
but its implementation should move toward focused modules by planner family and shared helper role.
The first extraction in this branch moves the occupancy-grid helper mixin to
`robot_sf/planner/socnav_occupancy.py` without changing planner behavior.

## Compatibility Strategy

- Keep `robot_sf.planner.socnav` as the stable facade for existing imports.
- Re-export moved public helpers from `socnav.py` until downstream callers can migrate naturally.
- Avoid deprecation warnings for the first helper extraction because several in-tree planners still
  import `OccupancyAwarePlannerMixin` from `socnav.py`.
- Move one family at a time, with construction and small `plan()` smoke coverage before each PR.

## Target Modules

- `robot_sf/planner/socnav_occupancy.py`: shared occupancy-grid metadata, world/grid conversion,
  path penalty, and safe-heading helpers. Extracted first because it is self-contained and already
  has focused tests.
- `robot_sf/planner/socnav_config.py`: future home for `SocNavPlannerConfig` after family modules no
  longer need a central mega-module import anchor.
- `robot_sf/planner/socnav_sampling.py`: `TrivialReferencePlannerAdapter`,
  `SamplingPlannerAdapter`, `SocNavBenchSamplingAdapter`, and upstream SocNavBench setup helpers.
- `robot_sf/planner/socnav_social_force.py`: `SocialForcePlannerAdapter` and factory wrapper.
- `robot_sf/planner/socnav_orca.py`: ORCA/HRVO adapters and optional `rvo2` integration guards.
- `robot_sf/planner/socnav_sacadrl.py`: SACADRL helpers, model loader, and adapter.
- `robot_sf/planner/socnav_prediction.py`: prediction planner adapter once dependent modules can
  import it directly without cycles.

## Validation Boundary

The first extraction is behavior-preserving if:

- `from robot_sf.planner.socnav import OccupancyAwarePlannerMixin` and
  `from robot_sf.planner.socnav_occupancy import OccupancyAwarePlannerMixin` resolve to the same
  object.
- Existing occupancy-helper tests still pass.
- Existing SocNav adapter construction and small plan-smoke tests still pass.
- `rg "from robot_sf.planner.socnav" robot_sf tests examples scripts` is reviewed before moving
  additional family classes.

## Current Conclusion

The occupancy helper split is the lowest-risk first step because it removes shared helper logic from
the mega-module while preserving every current in-tree import path. Future PRs should extract one
planner family per branch after confirming import users and optional dependency guards for that
family.
