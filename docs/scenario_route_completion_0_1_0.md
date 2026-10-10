# Ordered pedestrian route completion

For 0.1.0, `FollowRouteBehavior` creates navigators with
`require_final_waypoint=True`. A group passing near its final endpoint while
still targeting an earlier waypoint continues along its route instead of
respawning. Both endpoint proximity and final-waypoint progress are required.
Robot navigators keep their existing versioned success definitions. Empty and
single-waypoint navigation semantics remain unchanged.

The loop and U-turn regression steps the real group behavior and checks both
absence of premature respawn and eventual respawn. It fails twice on fresh
main because the first step respawns the group. Existing navigation tests only
cover monotonic routes, and single-pedestrian behavior tests use a different
controller. No production test seam is added.

The read-only authoring diagnostic is:

```bash
uv run python scripts/validation/check_scenario_archetype_geometry.py \
  --ped-route-completion --map maps/svg_maps/classic_bottleneck.svg
```

It lists pedestrian route segments before the final segment whose distance
from the endpoint is at most 1 m (the route behavior's default threshold).
Loops remain legal with ordered completion; these rows are authoring warnings,
not benchmark eligibility or dynamic collision claims. The production SVG
parser regression distinguishes the earlier segment from the final segment.

Release input bytes remain unchanged. Runtime semantics change only for group
routes exposed to early endpoint proximity; historical 0.0.8 execution remains
on its freeze branch. Development-only empty-world sweeps test robot behavior
for unintended effects; group stepping regressions test the changed pedestrian
behavior directly. No held-out execution or benchmark ranking is admitted.
