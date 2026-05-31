# Stress-Slice Suite

```yaml
suite_id: policy_search_stress_slice
benchmark_track: policy_search_stress_slice
status: runnable_local_diagnostic
```

## Purpose

Exercise a candidate on harder bottleneck, doorway, group-crossing, intersection, and Francis-style
stress cases after smoke and nominal-sanity checks.

## Scenarios And Seeds

- Scenario matrix: `configs/policy_search/stress_slice_matrix.yaml`
- Seed manifest: `configs/policy_search/stress_slice_seeds.yaml`
- Scenario IDs:
  - `classic_bottleneck_medium`
  - `classic_doorway_medium`
  - `classic_group_crossing_medium`
  - `classic_t_intersection_medium`
  - `francis2023_blind_corner`
  - `francis2023_intersection_wait`
  - `francis2023_parallel_traffic`
  - `francis2023_crowd_navigation`
- Seeds: `111`, `112`, `113` for each scenario
- Horizon: `120`
- Workers: `2`

## Eligible Planners

Policy-search candidates that have at least smoke evidence and a clear reason to test hard cases.
Expensive or artifact-bound planners should document the artifact availability path before running.

## Metrics

Success rate, collision rate, near-miss rate, low-progress timeouts, scenario-family split,
scenario exclusions, mean minimum distance, mean speed, and failure taxonomy.

## Canonical Command

```bash
uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate <candidate_id> \
  --stage stress_slice \
  --output-dir output/policy_search/<candidate_id>/stress_slice/manual \
  --workers 2
```

## Expected Runtime

Usually longer than nominal sanity and highly candidate-dependent. Use local workers conservatively;
route-heavy planners and learned checkpoints may require more time.

## Claim Boundary

Stress-slice results are development evidence for hard-case behavior. They do not certify scenario
coverage and should not be merged into nominal benchmark aggregates without an explicit benchmark
contract.

## Caveats

Treat fallback/degraded rows as limitations. A stress failure can be valuable diagnostic evidence,
but it is not a benchmark promotion signal without durable artifacts and clear row status.
