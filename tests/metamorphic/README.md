# Metamorphic environment and planner tests

This package checks whether public environment and planner contracts preserve
relations that should hold regardless of planner quality. It is a deterministic
test suite, not a benchmark or evidence of planner performance, pedestrian-model
validity, metric quality, or a scientific result.

The crowd-only fixture uses three explicit pedestrians, a square synthetic map,
fixed `0.1 s` steps, a fixed seed, and no pedestrian-obstacle force. Its numeric
comparisons use `rtol=0` and `atol=1e-5`: the absolute tolerance covers only
float32 serialization and integration round-off, while zero relative tolerance
keeps small-magnitude frame or identity drift visible. A failure reports the
first divergent step and observation field together with the maximum absolute
error. The trajectory and outcome relations use bounded real `RobotEnv`
episodes with explicit map and seed fixtures; grid and unit checks call planner
or config surfaces directly.

The twelve test modules cover:

| Test | Contract exercised |
| --- | --- |
| `test_scene_translation.py` | Translating scene coordinates translates positions and goals, while velocities and forces remain unchanged. |
| `test_scene_rotation.py` | A 90-degree rotation transforms positions, goals, velocities, and forces in the corresponding frame. |
| `test_row_permutation.py` | Declared pedestrian rows may be reordered without changing actor-associated state after identity matching. |
| `test_reset_isolation.py` | Both A→B and B→A reset orders reproduce fresh seeded episodes. |
| `test_render_independence.py` | Rendering at every observation boundary does not alter simulation state. |
| `test_record_replay_roundtrip.py` | The environment’s compact JSONL state recording replays without numeric drift. This is state-trace replay, not action re-simulation. |
| `test_oracle_isolation.py` | Opt-in privileged traces and randomized simulator identity labels do not change or enter actor-visible observations. |
| `test_replay_determinism.py` | Same-host crowd traces are byte-identical; seeded real `RobotEnv` runs repeat planner commands, projected actions, robot states, outcomes, and fallback metadata. The complete-episode case is marked slow. |
| `test_planner_unit_consistency.py` | Audits physical-unit fields in the planner readiness matrix's representative YAMLs and four known planner defaults against the differential-drive envelope where the semantics allow it. The selected hybrid v3 safety/braking mismatch is a strict expected failure for #9726. |
| `test_grid_resolution_invariance.py` | Native ORCA commands are stable across 0.1, 0.2, and 0.4 m rasters of one wall; DWA commands and social-force obstacle forces have strict expected failures for #9740 and #9724. |
| `test_mirror_symmetry.py` | A mirrored real `RobotEnv` scene gives the mirrored SocialForcePlanner route and robot trajectory with the same outcome. |
| `test_pedestrian_removal.py` | The SocialForcePlanner accepts empty/nonempty agent observations; a paired successful episode preserves success after removing a nearby pedestrian, and an expanded crossing case is marked slow. |

The suite is collected by the repository’s normal `tests/` discovery and
`run_tests_parallel.sh`; it does not alter production code or benchmark metrics.
If a base implementation violates a relation, the failure should be filed as a
separate bug and linked to a strict `xfail` only after the defect is reproduced
and its expected scope is documented.

## Numeric tolerance versus exact representation identity

Two comparison boundaries coexist in this package, and they are not
interchangeable:

- **Numeric metamorphic tolerance** (`assert_trace_equal`, `rtol=0`,
  `atol=1e-5`): the default for the frame-transform, reset, render, replay, and
  oracle-visibility relations. The absolute tolerance covers only float32
  serialization and integration round-off; zero relative tolerance keeps
  small-magnitude drift visible. It proves the *values* of a relation hold.
- **Exact representation identity** (`assert_trace_byte_identical`): used by
  the randomized simulator-identity and same-host replay relations.
  Observation dtype, shape, and C-order byte sequence must match exactly, and
  the JSON-serializable info payload must serialize identically (sorted keys,
  compact separators, strict finite floats). Numeric closeness never
  substitutes for representation identity here, because `allclose` can hide a
  changed dtype, byte ordering, signed zero, or metadata representation while
  values remain within tolerance.

A relation may use both: the tolerant comparison reports the first numeric
divergence with the maximum absolute error, and the byte-identity comparison
reports the first representation divergence with dtype, shape, and the first
differing byte offset.

## Replay across hosts

Run the fast local checks with:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/metamorphic/test_grid_resolution_invariance.py tests/metamorphic/test_mirror_symmetry.py tests/metamorphic/test_pedestrian_removal.py tests/metamorphic/test_planner_unit_consistency.py tests/metamorphic/test_replay_determinism.py -q -m 'not slow'
```

The expanded cases are selected on the cluster with:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/metamorphic/test_pedestrian_removal.py tests/metamorphic/test_replay_determinism.py -q -m slow
```

These synthetic fixtures check implementation contracts only. The DWA
counterexample in #9740 changes its command by 0.16 m/s between rasters; a
tolerance large enough to pass it would hide the wall response. The DWA and
social-force expected failures are narrow and strict. Neither is admitted as
planner-quality or benchmark success evidence.

Run `test_replay_determinism.py` on each host with the same repository revision,
dependency lock, map fixture, and seed. The test proves exact repeatability
only **within each host**. For Mac-to-cluster comparison, capture the same crowd
observation fields, planner commands, projected actions, and robot poses at each
step. Align pedestrian row identities and compare numeric values with `rtol=0`,
`atol=1e-5` in each field's native units (metres, m/s, m/s², radians, or the
recorded force units). Require exact field names, shapes, row identities, step
count, success/collision/truncation outcomes, and fallback status/count/reason.
This tolerance allows float32 serialization and integration round-off, not a
changed outcome or a substituted fallback planner.
No paired Mac/cluster artifact is part of this test suite, so cross-host
agreement is a documented comparison contract rather than a proven result.

## Planner-unit audit boundary

The audit reads every representative YAML path in
`configs/benchmarks/planner_readiness_matrix_v1.yaml` and the defaults of
`HybridRuleLocalPlannerConfig`, `SocNavPlannerConfig`, `DWAPlannerConfig`, and
`RiskDWAPlannerConfig`. Its explicit field inventory fails when a new
unit-bearing name is not classified. It checks finite values and valid signs,
and bounds explicit YAML forward-speed caps by
`DifferentialDriveSettings.max_linear_speed`. DWA's clearance-score
normalization is checked against contact scale, but it is a scoring distance,
not a stopping or safety gate. For the selected hybrid v3, the strict #9726
expected failure checks
pedestrian centre-distance thresholds and the rollout's acceleration and
deceleration against the drive. A new hybrid v4 configuration needs a separate
**passing** safety-envelope test; it must not inherit the v3 expected failure.

Goal tolerances, map extents, surface clearances, and navigation trigger
distances are all metres but are not interchangeable with a robot centre
contact threshold. Angular command caps may exceed the differential drive's
angular cap and are clipped by the actuator; this audit does not call that a
passing feasibility relation. Several legacy planner robot-radius defaults
(`0.25–0.3 m`) are smaller than the drive's `1.0 m` collision radius; they are
explicit exceptions, not proof of collision-radius consistency (see #4856).
Pedestrian radii and proxemic radii are separate quantities. The social-force
parameters and synthetic actuation-score fields are model or diagnostic
quantities, not drive limits.
Implicit `algo=` rows use runtime defaults or a goal heuristic, so the audit
does not claim to cover every runtime override, stochastic policy, or planner
family outside these representative sources. It is not benchmark success
evidence.
