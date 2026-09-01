# `core_contract.v1` simulation-state boundary

Issue #8243 adds a small, additive type boundary for future simulator and
research consumers. It centralizes names without migrating existing call sites:

- `WorldFrame` is the identity alias of `sensor.pedestrian_tracking.PedestrianCoordinateFrame`.
- `Pose2D` is the identity alias of `adversarial.config.Pose2D`.
- `ObservationSnapshot` is the identity alias of the observation-only
  `PedestrianObservationSnapshot` contract from #8066.
- `ForceComponent`/`ForceBreakdown` are identity aliases of the oracle trace's
  `ForceComponentRecord`/`ForceComponents` values.
- `TransitionRecord` is the identity alias of `OracleTransitionTraceV1`.
- `EpisodeRecord` is the identity alias of `benchmark.types.EpisodeRecord`.
- `ActorId` and `TrackId` are opaque, non-empty string type aliases. Adapters
  must preserve source identity explicitly when converting integer or string
  source-local identifiers; the core boundary does not infer equivalence.

The only new state values are frozen `SimTime`, `Twist2D`, and `ActorState`.
`SimTime` carries a discrete step plus elapsed simulation seconds derived from
the configured fixed step. `Twist2D` carries signed planar linear velocity in
metres per second and angular velocity in radians per second. `ActorState`
keeps source identity, optional observation-track identity, coordinate frame,
validity, pose, twist, and decision-point time separate and serializes them with
strict versioned keys.

## Evidence boundary

The focused tests prove import safety, alias identity, finite-value validation,
frozen values, and serialization round-trips. This is implementation-integrity
evidence only. No simulator wiring, metric change, planner result, benchmark
ranking, human-behaviour claim, or paper-facing claim follows from this package.
Consumer migration and transition-stage wiring remain follow-up work under the
parent programme #8241.
