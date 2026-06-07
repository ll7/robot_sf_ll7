# Issue #2530 Topology Near-Parity Corrective-Behavior Smoke

Issue: [#2530](https://github.com/ll7/robot_sf_ll7/issues/2530)
Status: current, diagnostic-only result.

## Claim Boundary

This note records a bounded corrective-behavior smoke for the
`topology_hypothesis_near_parity_diversity_gate_v0` direction accepted by
[#2518](issue_2518_topology_near_parity_gate.md). It is not benchmark success evidence and does not
promote `topology_guided_hybrid_rule_v0` out of diagnostic-only status.

## Result

Classification: `revise` for the topology-selection research lane.

The canonical slice completed:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir output/diagnostics/issue2530_topology_near_parity_corrective_smoke
```

Compact evidence:
[evidence/issue_2530_topology_near_parity_corrective_smoke_2026-06-07/summary.json](evidence/issue_2530_topology_near_parity_corrective_smoke_2026-06-07/summary.json)

Observed signals:

- `diagnostic_status`: `diagnostic_complete`
- topology status counts: `ok=90`, `insufficient_hypotheses=70`
- route-selector selections: `primary_route=56`, non-primary hypotheses total `42`
- selected near-parity gate reasons: `selected_non_primary_near_parity=42`,
  `route_distance_exceeds_slack=25`, `static_clearance_floor_failed=17`,
  `eligible_near_parity_alternative=14`
- local command sources: `dynamic_window=121`, `topology_hypothesis=33`, `route_guide=4`,
  `path_follow_0.5m=2`
- topology-command influence: `primary_route=26`, `masked_cell_84_103=6`,
  `masked_cell_85_103=1`
- corrective-behavior decision: `revise`
- terminal outcome: `horizon_exhausted` at step 159, with no success and no pedestrian, obstacle,
  or robot collision

## Interpretation

The near-parity gate still moves the route selector beyond primary-route dominance and reaches
local-command arbitration. That is stronger than selection diversity alone: the trace includes 33
`topology_hypothesis` command-source selections and 7 non-primary topology-command influence steps.

However, the corrective-behavior bar is not met in this h160 slice. The episode exhausted the
horizon without success, and the largest per-rank route-progress delta was only `0.16812408921843236`
meters. This is a useful negative/diagnostic result: topology should stay in `revise`, not
`continue` or benchmark-promotion mode.

## Next Direction

The next smallest proof step is a paired or broadened diagnostic that distinguishes whether the
failure is mainly hypothesis availability, weak command arbitration, horizon/route-progress
accounting, or near-parity parameterization. The run should keep the diagnostic-only boundary and
compare against a gate-disabled or baseline-local-policy slice before treating any outcome as
planner improvement.

## Validation

Executed from branch `issue-2530-topology-near-parity-corrective-smoke`:

```bash
PYTHONPATH=. scripts/dev/run_worktree_shared_venv.sh -- \
  uv run pytest tests/validation/test_run_topology_hypothesis_diagnostics.py -q
PYTHONPATH=. scripts/dev/run_worktree_shared_venv.sh -- \
  uv run ruff check scripts/validation/run_topology_hypothesis_diagnostics.py \
    tests/validation/test_run_topology_hypothesis_diagnostics.py
PYTHONPATH=. scripts/dev/run_worktree_shared_venv.sh -- \
  uv run ruff format --check scripts/validation/run_topology_hypothesis_diagnostics.py \
    tests/validation/test_run_topology_hypothesis_diagnostics.py
```

The targeted test run passed with 13 tests. Ruff check passed and Ruff format reported both touched
Python files already formatted after formatting was applied.
