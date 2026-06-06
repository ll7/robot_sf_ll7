# Issue #2258 Topology Primary-Route Audit Manifest 2026-06-05

Date: 2026-06-05

This directory preserves compact durable evidence for the #2258 primary-route audit. Raw regenerated
trace files remain under ignored `output/` and are not tracked.

## Tracked Files

- `summary.json`: compact diagnostic summary, selected-hypothesis counts, primary-vs-alternative
  route-distance dominance, interpretation, and recommendation.
- `topology_hypothesis_inventory.csv`: generated per-step inventory with status, hypothesis count,
  candidate IDs, selected topology-command hypothesis when present, primary-vs-best-alternative
  remaining-distance comparison, and explicit missing trace fields.

## Source Evidence

- `docs/context/issue_2223_topology_hypothesis_planning.md`
- `docs/context/evidence/issue_2223_topology_hypothesis_2026-06-04/summary.json`
- `scripts/validation/run_topology_hypothesis_diagnostics.py`
- `robot_sf/planner/topology_guided_local_policy.py`
- `configs/policy_search/candidates/topology_guided_hybrid_rule_v0.yaml`

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
  --output-dir output/diagnostics/issue2258_topology_audit/classic_realworld_double_bottleneck_high_seed111_h160
```

## Claim Boundary

This is diagnostic trace-smoke evidence for one scenario, seed, and horizon. It may guide the next
topology-hypothesis implementation, but it must not be cited as benchmark-strength planner
mitigation evidence.

The inventory records the trace fields available before scoring. Per-hypothesis selection scores and
candidate rejection reasons were not logged in the source trace, so score-margin and rejection-reason
analysis remains a follow-up instrumentation need.
