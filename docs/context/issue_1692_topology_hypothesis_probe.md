# Issue #1692 Topology-Hypothesis Trace Probe

Status: diagnostic-only implementation evidence for
[issue #1692](https://github.com/ll7/robot_sf_ll7/issues/1692).

## Goal

Add a bounded trace probe that can expose route-topology alternatives around static bottleneck
obstacles while preserving the benchmark fallback policy. The probe must report each hypothesis'
static clearance, dynamic clearance, route progress reference, and the local command source selected
by the active planner. It is not a planner implementation and should not be cited as benchmark
success evidence.

## Implementation

The new entry point is `scripts/validation/run_topology_hypothesis_diagnostics.py`.

It runs a policy-search candidate on one scenario/seed, extracts the occupancy-grid route planner's
base path, and masks cells along that path to find alternative A* routes. Alternatives are accepted
only when they differ by path overlap and by a compact low-clearance bottleneck signature, so the
probe does not count same-gap wiggles as separate topology hypotheses. If no step reaches the
minimum hypothesis count, the script exits with status `2` and records
`diagnostic_status: not_available`; this non-zero exit is intentional fail-closed behavior for
automation that expects at least two hypotheses.

## Validation Evidence

Canonical command:

```bash
uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --output-dir output/diagnostics/issue1692_topology_hypothesis_probe/classic_realworld_double_bottleneck_high_seed111 \
  --max-hypotheses 3
```

The generated full trace remains in ignored `output/`; compact durable evidence is copied to
`docs/context/evidence/issue_1692_topology_hypothesis_probe_2026-05-30/summary.json`.

Observed diagnostic output:

- `diagnostic_status`: `diagnostic_complete`
- `claim_boundary`: `diagnostic_only_not_benchmark_success`
- topology availability over 160 steps: `ok=96`, `insufficient_hypotheses=64`
- selected local command sources: `dynamic_window=156`, `path_follow_0.5m=2`, `route_guide=2`
- minimum static clearance by reported rank: `0.20000000298023224 m` for ranks 0, 1, and 2
- minimum dynamic clearance by reported rank: rank 0 `0.6522960007333864 m`,
  rank 1 `0.6510872404758408 m`, rank 2 `0.6392461414108603 m`

The progress summary uses the first and last sample observed for each rank over the diagnostic
horizon. Rank 0 starts with a short route because the episode begins near the first route segment;
after the route target advances, remaining route distance jumps to the longer bottleneck segment.
Treat the progress deltas as trace-local evidence, not as a success or efficiency metric.

Focused tests:

```bash
uv run pytest tests/validation/test_run_topology_hypothesis_diagnostics.py
uv run ruff check scripts/validation/run_topology_hypothesis_diagnostics.py \
  tests/validation/test_run_topology_hypothesis_diagnostics.py
```

The tests cover two-gap recovery, single-gap fail-closed behavior, and summary aggregation for
selected command sources and progress deltas.

## Interpretation

The trace shows the diagnostic path can expose multiple route hypotheses in the target
`classic_realworld_double_bottleneck_high` seed `111` slice and record the selected local command
source alongside static and dynamic clearance. The selected planner still chooses mostly
`dynamic_window`; this evidence supports future planner investigation, but it does not itself prove
a new topology-aware planner behavior or benchmark improvement.
