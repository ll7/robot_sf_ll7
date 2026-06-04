# Issue #2223 Topology-Hypothesis Evidence Manifest

Date: 2026-06-04

This directory preserves compact durable evidence for the #2223 topology-hypothesis bottleneck
diagnostic. Raw step traces remain local ignored artifacts and are not tracked.

## Tracked Files

- `summary.json`: compact baseline-vs-topology diagnostic summary, deltas, and interpretation.

## Commands

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2
```

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160
```

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160
```
