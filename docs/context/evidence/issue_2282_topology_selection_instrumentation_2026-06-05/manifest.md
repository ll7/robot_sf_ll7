# Issue #2282 Topology Selection Instrumentation Manifest 2026-06-05

This directory preserves compact durable evidence for the Issue #2282 topology-selection score and
rejection-reason instrumentation smoke.

## Tracked Files

- `summary.json`: machine-readable smoke summary, emitted fields, scenario/seed/horizon, topology
  status counts, selected-source counts, and one compact score/rejection example.

## Source Output

Raw generated output was written to a temporary local directory and summarized into `summary.json`.
The raw trace and Markdown report are not tracked and are not treated as durable evidence.

## Command

```bash
scripts/dev/run_worktree_shared_venv.sh \
  --venv /home/luttkule/git/robot_sf_ll7.worktrees/autopilot-research-cycle-20260605/.venv \
  -- uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir /tmp/robot_sf_issue2282_topology_selection_scores/classic_realworld_double_bottleneck_high_seed111_h160
```

## Claim Boundary

This is instrumentation-smoke evidence only. It proves the diagnostic path emits per-hypothesis
score components, score rank, margin to selected, selection outcome, and rejection reason on the
Issue #2258 topology slice. It is not planner mitigation evidence or benchmark-strength planner
performance evidence.
