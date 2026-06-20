# Issue #3206 Heterogeneous Pedestrian Smoke

Status: `smoke_complete`

This packet records a tiny paired homogeneous-vs-heterogeneous pedestrian composition smoke.
It is not a full benchmark campaign, not a planner-ranking claim, and not a real-world
pedestrian-behavior realism claim.

## Provenance

- Branch: `issue-3206-archetype-reporting`
- Source commit at run time: `95e42c514c0f00baf624084ec86e625f11c35f2a` plus the runtime wiring
  changes in this PR branch.
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

scripts/dev/run_worktree_shared_venv.sh -- robot_sf_bench --quiet aggregate \
  --in output/benchmarks/issue_3206_heterogeneous_pedestrian_smoke/episodes.jsonl \
  --out output/benchmarks/issue_3206_heterogeneous_pedestrian_smoke/aggregate_by_condition.json \
  --group-by scenario_params.metadata.archetype_condition
```

## Smoke Result

| Condition | Success | Collisions | Min distance mean | Mean distance mean | Robot within 5 m frac | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `homogeneous_standard` | 0.0 | 0.0 | 3.226 | 4.750 | 0.571 | Timeout, no collision, route interaction exposure present. |
| `mixed_balanced` | 0.0 | 0.0 | 5.010 | 5.706 | 0.042 | Timeout, no collision, larger clearance in this three-seed smoke. |

Observed deltas (`mixed_balanced - homogeneous_standard`):

- `min_distance.mean`: `+1.784`
- `mean_distance.mean`: `+0.956`
- `robot_ped_within_5m_frac.mean`: `-0.529`
- `success.mean`: `0.000`
- `collisions.mean`: `0.000`

## Claim Boundary

This is enough to show that the homogeneous-vs-heterogeneous scenario axis reaches benchmark runtime
and can produce reviewable metric deltas on a fixed seed.
It is not enough to claim robust composition dependence, pedestrian realism, or planner ranking.
A larger seed/scenario slice is required before promotion beyond smoke evidence.
