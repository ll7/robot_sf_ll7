# H500 Leader Clean Rerun

## 2026-05-06 Update

This rerun remains the clean provenance record for `scenario_adaptive_hybrid_orca_v1` and
`hybrid_rule_v3_fast_progress`, but its strict-gate recommendation is superseded by the later
collision-guard rerun: `scenario_adaptive_hybrid_orca_v2_collision_guard` reached `0.9028` success
and `0.0139` collision on `full_matrix_h500` and passed the strict `nominal_sanity` promotion gate.
Use `docs/context/policy_search/reports/2026-05-05_full_matrix_h500_analysis.md` for the current
promotion decision.

## Goal

Rerun the two h500 promotion candidates from a clean, pinned worktree so the result can be used as
stronger release-candidate evidence than the exploratory h500 sweep.

## Candidate Set

Use `configs/policy_search/candidate_sets/h500_leader_clean_rerun.txt`:

- `scenario_adaptive_hybrid_orca_v1`
- `hybrid_rule_v3_fast_progress`

`scenario_adaptive_hybrid_orca_v1` is the best observed h500 leader. At the time of this rerun,
`hybrid_rule_v3_fast_progress` was the best strict-gate-clean candidate from the same h500 evidence;
the 2026-05-06 collision-guard rerun later replaced it as the primary strict-gate promotion target.

## Submission Command

Run from a clean worktree on the SLURM host:

```bash
scripts/dev/sbatch_policy_search_sweep.sh \
  --stage full_matrix_h500 \
  --candidates-file configs/policy_search/candidate_sets/h500_leader_clean_rerun.txt \
  --run-id policy_search_full_matrix_h500_leaders_clean_$(date -u +%Y%m%d) \
  --throttle 1 \
  --workers 2 \
  --clean-pinned
```

`--clean-pinned` records the submitting `HEAD` and makes each array task fail if the checkout moves
or has non-ignored local changes when the task starts. This prevents the mutable-worktree provenance
problem observed in the earlier full-matrix sweep.

## Expected Evidence

Each candidate should emit:

- `output/policy_search/<candidate>/full_matrix_h500/<run-id>/summary.json`
- `output/policy_search/<candidate>/full_matrix_h500/<run-id>/*combined.jsonl`
- a tracked candidate report under `docs/context/policy_search/reports/`

After both jobs finish, refresh:

```bash
uv run python scripts/tools/summarize_policy_search_portfolio.py \
  --output-md docs/context/policy_search/portfolio_overview_2026-05-05.md \
  --output-json docs/context/policy_search/portfolio_overview_2026-05-05.json
```

Then run strict promotion checks with:

```bash
uv run python scripts/tools/promote_policy_search_candidate.py \
  output/policy_search/scenario_adaptive_hybrid_orca_v1/full_matrix_h500/<run-id>/summary.json \
  --gate-name nominal_sanity \
  --output docs/context/policy_search/reports/promotions/<run-id>_strict_gate

uv run python scripts/tools/promote_policy_search_candidate.py \
  output/policy_search/hybrid_rule_v3_fast_progress/full_matrix_h500/<run-id>/summary.json \
  --gate-name nominal_sanity \
  --output docs/context/policy_search/reports/promotions/<run-id>_strict_gate
```

## Promotion Boundary

If the rerun reproduces the current pattern, the original local decision was:

- `scenario_adaptive_hybrid_orca_v1` as the experimental h500 leader.
- `hybrid_rule_v3_fast_progress` as the strict-gate-clean h500 candidate.

The current decision keeps `scenario_adaptive_hybrid_orca_v1` as the experimental raw-success
leader but promotes `scenario_adaptive_hybrid_orca_v2_collision_guard` as the primary strict-gate
h500 candidate.

Do not use this rerun as camera-ready evidence unless the final report names the exact commit,
run id, scenario matrix, seed list, and horizon.
