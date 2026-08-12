# Issue #6615 ORCA adapter-validation harness

Status: diagnostic implementation complete; the smoke is native and measurable, but it is not
benchmark evidence and does not retire the dissertation's ORCA adapter hedge.

Issue: [#6615](https://github.com/ll7/robot_sf_ll7/issues/6615)

## Contract

`ORCAPlannerAdapter._velocity_world_to_command()` projects a holonomic world-frame velocity into
the executed differential-drive command `(v, w)`. The new `orca_adapter_trace_enabled` field on
`SocNavPlannerConfig` is `False` by default. When enabled, the adapter records one
`orca_adapter_trace.v1` record per projection and exposes the records plus a summary through
`diagnostics()`.

Each record contains:

- the planned world velocity and speed (`planned_velocity_world_mps`, `planned_speed_mps`);
- the executed unicycle command (`executed_command_vw`);
- the instantaneous forward velocity implied by the current robot heading
  (`realized_velocity_world_mps`, `executed_speed_mps`);
- the angle between planned and realized velocity (`angle_error_rad`); and
- realized-minus-planned speed (`speed_delta_mps`).

The realized vector is intentionally instantaneous. The trace does not integrate angular velocity
over a timestep and therefore measures projection divergence, not closed-loop trajectory quality.
When tracing is enabled, non-finite velocity or command values fail closed instead of entering the
JSON-facing trace.

## Analytic validation

`tests/test_socnav_planner_adapter.py` covers aligned motion, full-slowdown reorientation when the
target is behind the robot, the `π/2` slowdown boundary, speed clamping, zero velocity, and the
enabled trace payload. The tests exercise the projection directly with controlled observations and
do not use fallback execution as evidence of native ORCA behavior.

## Native smoke result

The reproducible command is:

```bash
LOGURU_LEVEL=WARNING NUMBA_NUM_THREADS=1 uv run python \
  scripts/benchmark/run_orca_adapter_validation_issue_6615.py \
  --output-dir output/benchmarks/issue_6615_orca_adapter_validation_20260812T055400Z
```

The committed source head was `d3cc9367bac4eaea2b355b4ae1fd0a1918cc4c75`. The run was native
(`rvo2` available), captured four records from four fixed synthetic cases, and reported:

| quantity | mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| angle error (rad) | 1.113113 | 0.655430 | 2.778782 |
| speed delta (m/s) | -0.068717 | -0.037433 | 0.000000 |

The exact ignored-worktree report hashes are:

- JSON: `74d848cc50559a7b06419577bdd00b92f1d7de7a1613e9fda4a2e45ed3abb087`
- Markdown: `973cc7cee931b9d86bb3135a8f8a57458aca98904c3122c5ce303ff33854b76e`

## Claim boundary

This result proves that the adapter projection is testable and that a native smoke can emit the
declared pre/post fields. It does not establish planner quality, native-ORCA equivalence, safety,
ranking, or a dissertation-hedge update. A future hedge-changing campaign needs its own frozen
scenario/seed policy, uncertainty treatment, preregistration, and domain-aware approval.

The machine-readable summary is
[`evidence/issue_6615_orca_adapter_validation_summary.json`](evidence/issue_6615_orca_adapter_validation_summary.json).
