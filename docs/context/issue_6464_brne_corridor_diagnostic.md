# Issue #6464 — BRNE Corridor Diagnostic

Status: **diagnostic complete; no planner-ranking or benchmark claim**.

BRNE (Bayesian Recursive Nash Equilibrium) now has a fail-closed map-runner
adapter for the approved corridor-only preflight. The bounded run below proves
that the pinned upstream core can execute through the Robot SF adapter and emit
non-degenerate native-core actions. It does not establish safety, realism,
matched-compute parity, planner superiority, or paper evidence.

## Frozen contract

- Scenario: `classic_head_on_corridor_low` from
  `configs/scenarios/issue_6464_brne_corridor_diagnostic.yaml`.
- Seeds: `111`, `112`, and `113`; horizon `500`; timestep `0.1` seconds.
- Geometry: one constant-width corridor only. Doorways, T-junctions,
  arbitrary static geometry, and crowds above the upstream cap are excluded.
- BRNE source: `MurpheyLab/brne` at commit `633a5cd`, GPL-3.0, staged locally
  and not vendored or redistributed.
- The upstream request is `49` samples; its pinned grid produces `42` actual
  samples. The adapter records the effective count and handles the upstream
  plan-step-first tensor layout.
- The upstream random generator is rebound from each declared episode seed at
  planner reset, so repeated runs use the same BRNE sampling stream.
- Fallback is disabled. Fallback, degraded, unknown, failed, over-cap, or
  degenerate rows are unavailable rather than success evidence. Solver stop
  actions carry runtime failure provenance and are excluded even when the
  trace has non-zero displacement.

## Observed result

All three arms covered the exact three seed pairs. The comparator rows provide
paired diagnostic coverage only; they are not a ranking or benchmark result.

| Arm | Exact runs | Native/execution | Eligible | Goal reached | Non-degenerate | Corridor violations | Fallback/degraded | Over cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BRNE | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 0 | 0 | 0 |
| ORCA comparator | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 |
| Social-force comparator | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 0 | 0 | 0 |

The principal finding is therefore bounded and mixed: the native upstream BRNE
core executed through the Robot SF adapter with non-zero motion and stayed
inside the declared corridor, but reached the
goal in none of the three predeclared episodes. This is a hypothesis signal,
not evidence to widen the scenario class or promote the planner.

## Implementation findings

1. Occupancy-grid map observations expose flattened keys such as
   `robot_position`, `robot_velocity_xy`, `goal_current`, and
   `pedestrians_positions`. The BRNE bridge now normalizes those keys into the
   nested state contract before constructing the upstream payload. Without
   this bridge, the first campaign produced all-zero planner inputs and was
   correctly classified as incomplete.
2. The upstream sample generator can return fewer samples than requested and
   returns plan-step-first action tensors. The source wrapper now sizes its
   arrays from the effective upstream count and normalizes both the pinned
   plan-step-first layout and the legacy samples-first fixture layout fail-closed.
3. The SocNav observation contract stores pedestrian velocities in robot-ego
   coordinates. The BRNE bridge rotates them back to world coordinates before
   prediction and preserves the declared robot heading when it is available.
4. Report completion requires unique exact scenario/seed pairs. Goal-reaching
   counts are restricted to eligible rows; unavailable rows are retained only
   as explicit diagnostics.

## Reproduction and evidence

From the dedicated linked worktree, stage the external source and run:

```bash
uv run python scripts/tools/manage_external_repos.py stage brne
uv run pytest -q tests/baselines/test_brne_source_smoke.py tests/baselines/test_brne_planner.py
uv run python scripts/benchmark/run_brne_corridor_diagnostic_issue_6464.py \
  --config configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml \
  --output-dir output/benchmarks/issue_6464_brne_diagnostic_<timestamp>
```

The corrected local report was written to
`output/benchmarks/issue_6464_brne_diagnostic_20260812T022300Z/` and has
report JSON SHA-256
`a9679f3aa636381d79269e6adfe1817f3c8db3d6982debe0473162ba47da9306`.
The report's episode provenance records implementation commit
`af394c62bc349d1edcaa069bbf266b2296512d34`.
Raw episode files and the staged GPL source remain ignored, worktree-local
artifacts. The compact evidence handoff is tracked in
`docs/context/evidence/issue_6464_brne_corridor_diagnostic_summary.json`.

## Next decision

Open follow-up issue [#6923](https://github.com/ll7/robot_sf_ll7/issues/6923)
for a hypothesis-driven diagnostic of the `0/3` BRNE goal-reaching result before
any broader campaign. Candidate checks are goal/heading frame alignment,
progress-versus-interaction weighting, and whether the pinned upstream action
aggregation is appropriate for this control loop. The follow-up preserves the
same native-core-via-adapter, corridor-only, fail-closed boundary.
