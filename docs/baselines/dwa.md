# Dynamic Window Approach baseline

The Dynamic Window Approach (DWA) baseline samples commands the robot can reach within one control
period, then chooses the short rollout with the best goal alignment, obstacle/pedestrian clearance,
speed, and progress score. It provides a transparent classical local-planning comparison point.

A quick sanity smoke uses the planner-sanity matrix with an experimental benchmark profile:

```bash
uv run robot_sf_bench run \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --out output/benchmarks/dwa_smoke/episodes.jsonl \
  --algo dwa \
  --algo-config configs/algos/dwa_classic.yaml \
  --benchmark-profile experimental \
  --workers 1 --no-video --no-resume
```

The **standard classic archetype matrix** run (23 graded rows across 11 archetype configs, with
fixed per-scenario seeds) is:

```bash
uv run robot_sf_bench --quiet run \
  --matrix configs/scenarios/classic_interactions.yaml \
  --out output/benchmarks/dwa_archetype_matrix/episodes.jsonl \
  --algo dwa \
  --algo-config configs/algos/dwa_classic.yaml \
  --benchmark-profile experimental \
  --workers 4 --no-video --no-resume \
  --structured-output json
```

Executed archetype-matrix evidence (issue #5020, smoke/nominal only, no comparator) is recorded in
[`docs/context/evidence/issue_5020_dwa_archetype_matrix_2026-07-10/`](../context/evidence/issue_5020_dwa_archetype_matrix_2026-07-10/README.md).

The implementation is deterministic: it does not sample random commands, and tie-breaking follows
the fixed command lattice order. Its parameters are documented in
[`configs/algos/dwa_classic.yaml`](../../configs/algos/dwa_classic.yaml).

This planner is intentionally gated by `allow_testing_algorithms: true`. It is an implemented
classical baseline, but no full benchmark campaign or paper/dissertation claim is established by
this configuration. It emits differential-drive unicycle `(v, omega)` commands and is not available
for holonomic benchmark rows. The executed archetype matrix reached no goals and is reported as
smoke/nominal behavioral evidence for the canonical config, not as benchmark-strength success.
