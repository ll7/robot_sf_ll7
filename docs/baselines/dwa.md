# Dynamic Window Approach baseline

The Dynamic Window Approach (DWA) baseline samples commands the robot can reach within one control
period, then chooses the short rollout with the best goal alignment, obstacle/pedestrian clearance,
speed, and progress score. It provides a transparent classical local-planning comparison point.

Use the canonical configuration with an experimental benchmark profile:

```bash
uv run robot_sf_bench run \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --out output/benchmarks/dwa_smoke/episodes.jsonl \
  --algo dwa \
  --algo-config configs/algos/dwa_classic.yaml \
  --benchmark-profile experimental \
  --workers 1 --no-video --no-resume
```

The implementation is deterministic: it does not sample random commands, and tie-breaking follows
the fixed command lattice order. Its parameters are documented in
[`configs/algos/dwa_classic.yaml`](../../configs/algos/dwa_classic.yaml).

This planner is intentionally gated by `allow_testing_algorithms: true`. It is an implemented
classical baseline, but no full benchmark campaign or paper/dissertation claim is established by
this configuration. It emits differential-drive unicycle `(v, omega)` commands and is not available
for holonomic benchmark rows.
