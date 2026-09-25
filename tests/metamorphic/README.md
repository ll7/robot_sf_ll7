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

The test modules cover:

| Test | Contract exercised |
| --- | --- |
| `test_scene_translation.py` | Translating scene coordinates translates positions and goals, while velocities and forces remain unchanged. |
| `test_scene_rotation.py` | A 90-degree rotation transforms positions, goals, velocities, and forces in the corresponding frame. |
| `test_row_permutation.py` | Declared pedestrian rows may be reordered without changing actor-associated state after identity matching. |
| `test_reset_isolation.py` | Both A→B and B→A reset orders reproduce fresh seeded episodes. |
| `test_render_independence.py` | Rendering at every observation boundary does not alter simulation state. |
| `test_record_replay_roundtrip.py` | The environment’s compact JSONL state recording replays without numeric drift. This is state-trace replay, not action re-simulation. |
| `test_oracle_isolation.py` | Opt-in privileged traces and randomized simulator identity labels do not change or enter actor-visible observations. |
| `test_replay_determinism.py` | Same-host crowd traces are byte-identical. On a scene with real-area robot zones and a sampled crowd, the noisy (`noise_std > 0`) SocialForcePlanner baseline and the release `social_force`, `orca` and hybrid v3 arms repeat commands, actions, poses, crowd and outcome for one seed. Negative controls require a different environment seed to move the start and crowd, and a different planner seed to move the command noise. |
| `test_planner_unit_consistency.py` | Audits physical-unit fields in the release campaign's planner configs (resolved through `base_config_path` and every scenario override), the readiness matrix's representative YAMLs, and four planner defaults. Known violations are pinned value-for-value in a ledger; strict expected failures track #9726 (hybrid drive envelope and code defaults) and #9750 (robot-body radius). |
| `test_grid_resolution_invariance.py` | Native ORCA commands are stable across 0.1, 0.2, and 0.4 m rasters of one wall in an unsaturated regime; DWA commands and the default social-force obstacle force have strict expected failures for #9740 and #9724. The #9738 `resolution_independent_v2` twin passes, and is skipped with a reason until that option exists. |
| `test_mirror_symmetry.py` | The release `social_force` and `orca` arms return the transformed trace under y-mirror, x-mirror and a 90-degree rotation, with a pedestrian whose velocity crosses both mirror axes. Hybrid v3 keeps its outcome under each transform; its exact trace is a strict expected failure (a discrete near-tie flips). The SocialForcePlanner baseline and VisibilityPlanner mirror cases remain. |
| `test_pedestrian_removal.py` | The SocialForcePlanner accepts empty/nonempty agent observations and keeps success after a nearby pedestrian is removed. For each release arm, a pedestrian timed to cross the route changes the commands; the successful crossing episode stays successful and no slower without it. |

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

Run the checks with:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/metamorphic -q
```

No case carries a `slow` marker: the release-arm episodes take seconds, and PR
shards run with `-m "not slow"`, so a slow marker would keep them out of CI.

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
count, success/collision/step-limit outcomes, and fallback status/count/reason.
This tolerance allows float32 serialization and integration round-off, not a
changed outcome or a substituted fallback planner.
No paired Mac/cluster artifact is part of this test suite, so cross-host
agreement is a documented comparison contract rather than a proven result.

## Stochastic planners

Exact trace relations (mirror, rotation, removal) apply to deterministic planners
only. A sampling planner (`socnav_sampling`, MPPI, a stochastic learned policy)
draws different samples in a transformed scene even with the same seed. Such
planners are covered by seeded replay here; distribution-level agreement over
seeds belongs in the cluster tier. A reflection-type sign error (for example a
flipped pedestrian `vy`) commutes with every mirror, so only the rotation
relation detects it.

## Planner-unit audit boundary

The audit reads the release campaign planner list: the campaign config named by
`configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml` plus the
successor template. Each `algo_config` is resolved like the map runner does,
through `base_config_path`, for the default scenario and every scenario
override. It also reads every representative YAML in
`configs/benchmarks/planner_readiness_matrix_v1.yaml` and the defaults of
`HybridRuleLocalPlannerConfig`, `SocNavPlannerConfig`, `DWAPlannerConfig`, and
`RiskDWAPlannerConfig`. Its field inventory fails when a new unit-bearing name
is not classified. Rules:

- time steps (`*_dt`, `control_period`): positive and at most ten simulator steps (1.0 s);
- linear speeds, forward acceleration and braking: at most the drive's
  (2.0 m/s, 1.0 m/s², 1.0 m/s²); the unbound SocNav default speed is the one
  stated exception, clipped by the map-runner action adapter;
- angular speeds and angular accelerations: at most ten times the drive's;
- robot-body radii: at least the drive's 1.0 m body;
- pedestrian and proxemic radii: positive and at most 10 m;
- hybrid human speed gates, as centre distances: stop and moderate at least
  contact (1.4 m = 1.0 m body + 0.4 m simulator pedestrian); slow at least
  contact plus the drive's stopping distance.

Known violations are pinned value-for-value, so a new, worsened or fixed
violation fails the ledger test. Strict expected failures flip when #9726
(hybrid envelope, including code defaults) or #9750 (robot-body radius) is
fixed. A new hybrid v4 configuration needs a separate **passing**
safety-envelope test; it must not inherit the v3 expected failure. Goal
tolerances, map extents, surface clearances, and perception or neighbour ranges
are metres but are not a centre-contact threshold, so they get sign and
finiteness checks only.
Pedestrian radii and proxemic radii are separate quantities. The social-force
parameters and synthetic actuation-score fields are model or diagnostic
quantities, not drive limits.
Implicit `algo=` rows use runtime defaults or a goal heuristic, so the audit
does not claim to cover every runtime override, stochastic policy, or planner
family outside these representative sources. It is not benchmark success
evidence.
