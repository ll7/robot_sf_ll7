# Issue #3206 Heterogeneous Pedestrian Smoke Report

- Status: `diagnostic_smoke_report`
- Source episode git hash(es): `5dd029bd2b5672dc846b5a138077712a2f393ff4`
- Input rows: worktree-local ignored artifact summarized in this report (episodes.jsonl)
- Claim boundary: diagnostic_smoke_not_benchmark_evidence: summarizes a tiny homogeneous-vs-mixed pedestrian composition smoke. It records metric deltas and distributional-metric readiness limits; it does not establish pedestrian realism, real-world fairness, planner ranking, or a limitation-replacement decision.
- Per-archetype distributional status: `not_computable_from_current_smoke`
## Provenance

- Branch: `issue-3206-archetype-reporting`
- Matrix: `configs/scenarios/sets/issue_3206_heterogeneous_pedestrian_smoke.yaml`
- Planner: `simple_policy`
- Horizon: `80`
- Time step: `0.1`
- Seeds: scenario seeds `101`, `102`, `103`; route spawn seed `3206`; archetype seed `3206`
- Raw episodes: generated under local `output/` or `/tmp`; intentionally not committed.

## Commands

```bash
scripts/dev/run_worktree_shared_venv.sh -- robot_sf_bench --quiet run \
  --matrix configs/scenarios/sets/issue_3206_heterogeneous_pedestrian_smoke.yaml \
  --out output/benchmarks/issue_3206_heterogeneous_pedestrian_smoke/episodes.jsonl \
  --algo simple_policy --workers 1 --horizon 80 --dt 0.1 --no-video --video-renderer none \
  --no-resume --benchmark-profile baseline-safe --structured-output json

scripts/dev/run_worktree_shared_venv.sh -- python scripts/benchmark/build_heterogeneous_pedestrian_smoke_report.py \
  --episodes output/benchmarks/issue_3206_heterogeneous_pedestrian_smoke/episodes.jsonl \
  --output-dir docs/context/evidence/issue_3206_heterogeneous_pedestrian_smoke_2026-06-20
```

## Condition Metrics

| Condition | Episodes | Success | Collisions | Min distance mean | Mean distance mean | Robot within 5 m frac | Distributional status |
|---|---:|---:|---:|---:|---:|---:|---|
| `homogeneous_standard` | 3 | 0.000 | 0.000 | 3.226 | 4.750 | 0.571 | `not_computable` |
| `mixed_balanced` | 3 | 0.000 | 0.000 | 5.010 | 5.706 | 0.042 | `not_computable` |

## Planned Composition

- `homogeneous_standard`: `{"standard": 1.0}`
- `mixed_balanced`: `{"cautious": 0.34, "hurried": 0.33, "standard": 0.33}`

## Issue #3261 Archetype Interpretation (2026-06-22)

Issue #3261 resolves the remaining #3206 archetype-metric question as
documentation-only: do not add named per-archetype metrics for `cautious`,
`standard`, or `hurried` on the current smoke. The implemented #3194
distributional-disruption cohorts remain the metric contract:
`slow_speed_tier`, `fast_speed_tier`, and `extreme_speed_tier`.

Narrative mapping for this smoke is intentionally non-contractual:
`cautious` roughly approximates the slow-speed tier, `standard` approximates
the standard/baseline composition row, and `hurried` roughly approximates the
fast-speed tier, with very high configured speeds remaining covered by
`extreme_speed_tier`. This mapping is only a reader aid for interpreting the
planned pedestrian composition. It is not a schema, metric key, report column,
or benchmark claim.

Named per-archetype distributional metrics should be revived only if a future
paper or dissertation claim must report numbers per named archetype and paired
robot-present/control traces exist for those named archetypes. Until then,
`per_archetype_distributional_status` remains
`not_computable_from_current_smoke`.

## Distributional/Fairness Boundary

The current smoke carries `distributional_disruption` blocks, but support counts are zero
because no control trace was provided. That means per-archetype displacement or delay
fairness-style metrics are not computable from this smoke. The result remains useful as
a runtime and metric-delta smoke only.
