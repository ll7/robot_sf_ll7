# T60 capsule footprint interface for Robot SF 0.1.0

The T60 body is a capsule 1.183 m long and 0.478 m wide. In the simulation,
`RobotPose` is `((x, y), heading)` at the body centre, with position in metres
and heading in radians. The 0.705 m centre segment runs along that heading.
Each end cap has radius 0.239 m. The smallest centred disc that covers the
body has radius 0.5915 m.

## Geometry shared by collision, clearance, and pedestrian response

The interface for consumers is `robot_sf.robot.footprint.CapsuleFootprint`.
Its `length` and `width` are the full body dimensions. Both must be finite and
positive, with `length >= width`. Equal dimensions give a point centre segment
and a circular body.

| Method | Result at the supplied pose |
| --- | --- |
| `axis(pose)` | World-frame centre segment as a pair of endpoints, rear first. |
| `circle_gap(pose, centre, radius)` | Distance from the circle centre to the segment, minus both radii. |
| `segment_gap(pose, segment)` | Distance between the centre segment and a wall segment, minus the capsule radius. |
| `capsule_gap(pose, other, other_pose)` | Distance between two independently oriented centre segments, minus both capsules' radii. |

These distances are instantaneous. Positive gaps indicate separation, zero
indicates tangency, and negative gaps indicate overlap. The collision candidate
counts capsule gaps up to 1e-12 m as contact to allow for floating-point
roundoff. A contact solver needs a separate penetration-depth calculation.

For pedestrian response, project the pedestrian centre onto the closed segment
from `axis(pose)`. The projection clamps to an endpoint beyond either end and
gives the common endpoint for a circular body. Use that nearest axis point to
derive the outward direction and subtract both radii for the surface gap.
The pedestrian contact law owns the direction rule when the pedestrian centre
is on the axis and the displacement is zero.

## Configuration and planner admission

The capsule bicycle profiles select `capsule: {length: 1.183, width: 0.478}`.
`BicycleDriveSettings.capsule` holds the resolved `CapsuleFootprint` or `None`.
Validate the selector before constructing a consumer and include the capsule
dimensions in the plant identity. A malformed selector or a declared covering
radius smaller than half the body length fails validation. An unset capsule
selects the disc path. A separate reference profile retains the historical
1.0 m disc.

Consumers need the exact body and the live heading for capsule clearance. An
adapter that cannot honour the selected geometry refuses the combination
before planning. Any expressly approved use of a disc proxy needs a distinct
configuration and an explicit record of where the approximation is used.
The software interface leaves the supported planner roster and any proxy
admission to the release decision.

The geometry signatures are pinned to the
[capsule candidate at e910e265](https://github.com/ll7/robot_sf_ll7/blob/e910e2659d8bb94fb5bb797f3d7770772a150d1b/robot_sf/robot/footprint.py).
The body dimensions and vehicle staging come from
[#10093](https://github.com/ll7/robot_sf_ll7/issues/10093) and
[#10098](https://github.com/ll7/robot_sf_ll7/issues/10098). This document freezes
the interface for integration. The runtime implementation and release evidence
are pending.
