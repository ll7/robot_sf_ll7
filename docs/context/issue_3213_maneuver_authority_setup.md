# Issue #3213 Maneuver-Authority Sweep Setup

Status: setup/preflight only, not benchmark evidence.

Issue #3213 asks whether richer predictive-planner maneuver authority improves the hard-case seed
portfolio without safety regressions. The first bounded setup adds
`configs/benchmarks/predictive_hardcase_authority_grid_issue_3213.yaml`, a five-variant campaign
grid for `scripts/validation/run_predictive_success_campaign.py`.

## Grid Scope

The grid includes:

- `baseline`
- `high_angular`
- `dense_lattice`
- `nearfield_turn`
- `combined_max_authority`

This covers the required minimum of at least three authority settings while keeping the first local
campaign smaller than the full generated nine-config family under `configs/algos/hardcase_authority/`.

## Canonical Next Command

Use the hard-seed manifest and fixed planner checkpoint inputs required by the issue contract:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_predictive_success_campaign.py \
  --checkpoints <checkpoint-or-model-reference> \
  --hard-seed-manifest configs/benchmarks/predictive_hard_seeds_v1.yaml \
  --planner-grid configs/benchmarks/predictive_hardcase_authority_grid_issue_3213.yaml \
  --workers 1 \
  --output-dir output/benchmarks/issue_3213_maneuver_authority
```

Do not treat this setup note or the grid as evidence that any authority variant improves success.
Benchmark evidence starts only after the campaign writes valid result artifacts with provenance.

## Durable Evidence Packet

The closed-loop confirming-eval grid from this sweep is durably pinned (with full config + seed
provenance) at
[issue_3213_predictive_nontransfer_confirming_eval_2026-07-08](evidence/issue_3213_predictive_nontransfer_confirming_eval_2026-07-08/README.md)
(issue [#4879](https://github.com/ll7/robot_sf_ll7/issues/4879)). That packet archives the
checkpoint-eval plateau (closed-loop success `0.0667`-`0.1` across four populated prediction
checkpoints vs the `0.30` gate) behind the predictive-planner non-transfer finding.
