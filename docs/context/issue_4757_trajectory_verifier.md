# Issue #4757 Experimental AMMV Trajectory Verifier

Date: 2026-07-07

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/4757>

Status: Proposal / experimental prototype (not benchmark evidence).

## Goal

Prototype an experimental, opt-in trajectory verifier for AMMV (Autonomous
Micromobility Vehicle) planner outputs. The verifier evaluates candidate or
executed robot trajectories against structured safety/comfort predicates and
returns `accept`, `warn`, or `fallback_brake` without changing existing planner
behavior unless explicitly enabled in a future slice.

This is an **experimental test-time verification prototype**. It is:

- **not a formal safety case**;
- **not conformalized** (no calibrated prediction intervals);
- **not learned** (deterministic predicates only, no trained score model);
- **missing data fails closed to `warn`** rather than fabricating metrics such as
  time-to-collision from missing velocities;
- **default planner behavior unchanged** (no wiring into planner control loops,
  release gates, benchmark scoring, or paper/dissertation claims in the first
  slice).

## Files

- `robot_sf/benchmark/trajectory_verifier.py` - pure evaluator with
  `TrajectoryVerifierConfig`, `VerifierResult`, `verify_trajectory(...)`, and the
  opt-in `verify_episode_trace_window(...)` trace adapter.
- `tests/benchmark/test_trajectory_verifier.py` - deterministic fixtures covering
  `accept`, `warn`, `fallback_brake`, missing-velocity fail-closed, oscillation
  warn, shape validation, decision precedence, and trace-window slicing.
- `configs/benchmarks/trajectory_verifier_default.yaml` - optional default
  thresholds manifest mirroring `TrajectoryVerifierConfig` defaults. The verifier
  does not load this config automatically; it is a discoverability and
  reproducibility pointer for offline analysis.

## Predicate Summary

1. **Minimum footprint clearance** - center distance minus robot and pedestrian
   radii. Below the hard minimum fires `fallback_brake`; below the warning
   threshold (but above the hard minimum) fires `warn`.
2. **Time-to-collision (TTC) proxy** - simple constant-velocity closure model for
   the robot vs the nearest pedestrian. Missing velocities surface as
   `stale_or_missing_state` warning rather than a fabricated TTC. Below the hard
   threshold fires `fallback_brake`.
3. **Braking feasibility** - stopping distance `v^2 / (2 * a)` compared with
   projected clearance along the robot heading for pedestrians within the lateral
   footprint envelope. Infeasible braking near a pedestrian fires `fallback_brake`.
4. **Stale prediction / missing state** - `prediction_age_s` above the configured
   threshold, or missing robot/pedestrian velocities, fires `warn`.
5. **Recovery smoothness / oscillation proxy** - counts large heading changes
   between consecutive moving timesteps. Excessive oscillation fires `warn`.

## Decision Aggregation

Precedence: `fallback_brake > warn > accept`. The `risk_score` is a deterministic
decomposition in `[0, 1]`: `1.0` when any hard predicate fired; otherwise the
maximum soft-predicate contribution (each soft predicate contributes a value in
`[0, 0.5]`); `0.0` for a clean accept. The decomposition is documented in
`_aggregate_risk_score` and covered by the unit tests. No score model is trained.

## Verification Commands

```bash
uv run ruff check robot_sf/benchmark/trajectory_verifier.py tests/benchmark/test_trajectory_verifier.py
uv run pytest tests/benchmark/test_trajectory_verifier.py -q
```

## Out of Scope (First Slice)

- Wiring the verifier into any planner control loop, runner, release gate, or
  benchmark scoring surface.
- Conformal calibration of thresholds.
- Learned risk scoring.
- Any benchmark, paper, or dissertation claim that depends on verifier output.

## Future Hooks

The `verify_episode_trace_window(...)` adapter is the intended entry point for
offline episode-trace diagnostics. It can later support critical-interval or
trajectory-report diagnostics, but does not alter planner commands in this slice.
