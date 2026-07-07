# Issue #2224 Synthetic AMV Actuation Ranking Diagnostic

Date: 2026-06-04

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2224>

Status: diagnostic result; not paper-facing AMV evidence.

## Question

Do nominally useful local planners demand commands that look less feasible under the repository's
synthetic `amv-actuation-stress-v0` envelope than an actuation-aware local scorer?

## Evidence

I ran a matched one-seed `amv_actuation_smoke` slice on `classic_cross_trap_high` with:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_fast_progress --stage amv_actuation_smoke --output-dir output/policy_search/issue2224/hybrid_rule_v3_fast_progress/amv_actuation_smoke --workers 1
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_policy_search_candidate.py --candidate actuation_aware_hybrid_rule_v0 --stage amv_actuation_smoke --output-dir output/policy_search/issue2224/actuation_aware_hybrid_rule_v0/amv_actuation_smoke --workers 1
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/compare_policy_search_candidates.py output/policy_search/issue2224/hybrid_rule_v3_fast_progress/amv_actuation_smoke/summary.json output/policy_search/issue2224/actuation_aware_hybrid_rule_v0/amv_actuation_smoke/summary.json --output output/policy_search/issue2224/comparison
```

Compact evidence is preserved in
[issue_2224_amv_actuation_ranking_2026-06-04](evidence/issue_2224_amv_actuation_ranking_2026-06-04/README.md).

| Candidate | Episodes | Success | Collision | Failure mode | Command clip | Yaw saturation | Signed braking peak | Mean min distance |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `hybrid_rule_v3_fast_progress` | 1 | 0.0000 | 0.0000 | `timeout_low_progress: 1` | 0.2750 | 0.0000 | -2.5000 | 2.1571 |
| `actuation_aware_hybrid_rule_v0` | 1 | 0.0000 | 0.0000 | `timeout_low_progress: 1` | 0.1875 | 0.0000 | -2.5000 | 2.3627 |

## Interpretation

Observed evidence supports only a weak diagnostic direction: the actuation-aware scorer reduced
mean command clipping by 0.0875 absolute on this smoke row while preserving zero collision and zero
yaw-rate saturation. It did not solve the task: both candidates reached `max_steps` and are
classified as `timeout_low_progress`.

This is not a material planner-ranking result because the slice has one episode, no successes, and
no broader scenario-family coverage. It does, however, justify keeping actuation-aware scoring as a
useful diagnostic candidate when the next AMV actuation run has a broader matched slice.

## Claim Boundary

This is a synthetic software stress diagnostic under
[issue_2230_amv_actuation_evidence_ladder.md](issue_2230_amv_actuation_evidence_ladder.md). It does
not support calibrated AMV hardware claims, platform-class proxy claims, safety certification, or
paper-facing AMV planner superiority.

## Next Step

Run a broader matched `nominal_sanity` or purpose-built AMV actuation slice only after the report
surface can keep synthetic actuation metrics in the planner comparison. This PR updates
`scripts/tools/compare_policy_search_candidates.py` so future comparisons preserve command clip,
yaw saturation, and signed braking peak columns.
