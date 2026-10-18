# ORCA with an acceleration reachability filter (A′)

Research-only, diagnostic empirical comparison. No release, publication, safety,
or full-suite claim. Use algorithm `orca_avo` with
[`orca_avo_v1.yaml`](../../configs/algos/orca_avo_v1.yaml); ordinary `orca` keeps
the same implementation and config schema. Native RVO2 and binding to a forward
`DifferentialDriveSettings` environment are required; fallback is rejected.

The adapter filters ORCA's solved world velocity using the control-period disk
`norm(target-current) <= delta * min(max_linear_accel,max_linear_decel)` and the
physical speed bound. The minimum is conservative for asymmetric acceleration
and braking. It searches four radial rings and 32 world-frame angles, current
velocity, the exact proposal projection, and maximum braking. It selects the
safe **sampled** candidate closest to ORCA's solved proposal, rather than claiming
a continuous optimum. Rejected candidates intersect static map segments,
observed local occupancy obstacles, or constant-velocity predicted pedestrians
along the exact exponential trajectory for `a(t)=(target-v(t))/delta`.
Samples are at most one control period or 0.1 s apart; swept chords and an
acceleration-based interpolation padding cover the gaps under that model.

If no sampled candidate is safe, it emits reachable maximum braking and records
`no_safe_target_braking`. That status is an unresolved constraint situation,
never a claim of safety. Diagnostics expose the proposal, selected target,
ball radius, candidate counts, and episode counts of these situations.

The target then passes through ORCA's unchanged differential-drive mapping.
The disk and trajectory are holonomic surrogates: angular limits, heading
projection, occupancy heading changes, and actual pedestrian acceleration can
break trajectory agreement. AVO guarantees assume holonomic reciprocal agents;
results here are empirical only. This stage deliberately measures A′ before
the separate rollout-MPC stage.

Preregistration and reproducible raw traces/scripts are preserved in the
`orca_avo-20261015` evidence packet identified in issue 10404. Its `plan.md`
fixes dev seeds 1001–1030, the six diagnostic scenarios plus other doorway and
bottleneck scenarios, primary collisions per episode and success, a paired
95% seed-block bootstrap, secondary metrics, and the keep/defer rule before
running. Results and validation are recorded there and in the PR.

Focused verification:

```sh
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/planner/test_orca_avo.py tests/planner/test_socnav_orca_module.py -q
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

The compatibility test checks byte-identical commands and trace serialization
against direct original ORCA construction and pins the original ORCA source
digest. The archived replay additionally checks the complete diagnostic episode.
