# Issue #3637 Closure Audit

Plain-language summary: issue #3637 is not ready to close. Merged PRs delivered the
reactivity-vs-replay mechanism, preflight contract, frozen launch packet, fail-closed post-run
analyzer, and campaign-runner metadata hooks. The issue's central acceptance criterion still lacks
a durable >=3-planner, S20 campaign result plus seed-sufficiency analysis bundle, and that empirical
run was outside this closure-audit authorization.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/3637>
- Audit date: 2026-07-05
- Latest maintainer guidance reviewed: 2026-07-04 issue comments. They state PR #4462 and PR #4492
  merged the analyzer and runner metadata support, while the >=3-planner, 20-seed campaign run and
  analysis remain open.
- Merged PRs reviewed:
  - PR #3594, `4e8899e96295dd054895cb458e42ad869b56a87b`,
    <https://github.com/ll7/robot_sf_ll7/pull/3594>
  - PR #3612, `3134353784acd19d5d46c0976298607d396cf44e`,
    <https://github.com/ll7/robot_sf_ll7/pull/3612>
  - PR #3696, `69265c2296e3d703d748fa71315a18aad6970741`,
    <https://github.com/ll7/robot_sf_ll7/pull/3696>
  - PR #3853, `5e566f391f0f6b3605ea55792d3fb3d0b3f5ddef`,
    <https://github.com/ll7/robot_sf_ll7/pull/3853>
  - PR #3860, `bb3260145fcfd0877170c1c142b403449caa7579`,
    <https://github.com/ll7/robot_sf_ll7/pull/3860>
  - PR #3876, `dd82bd21df1990bff4224be69ced9ce46e054756`,
    <https://github.com/ll7/robot_sf_ll7/pull/3876>
  - PR #4150, `fb74c5091410581a6292b32062e117020ce48c4e`,
    <https://github.com/ll7/robot_sf_ll7/pull/4150>
  - PR #4462, `0e3a2096e96825106b25781b34603fcb7137a76a`,
    <https://github.com/ll7/robot_sf_ll7/pull/4462>
  - PR #4492, `c9a4584eb146757433840d5a149a430456fc9db6`,
    <https://github.com/ll7/robot_sf_ll7/pull/4492>

## Claim Boundary

This is a closure-audit and integration-status artifact only. It does not run the benchmark
campaign, submit Slurm or GPU work, promote local `output/` files, edit paper/dissertation claims,
or classify a reactivity-rank result as seed-sufficient.

The fragmentation guard applies: PR #4462 and PR #4492 both merged for issue #3637 within the last
24 hours. This artifact is therefore a consolidation slice, not another guardrail/checker refresh.

## Acceptance Mapping

| Acceptance criterion | Merged evidence | Audit status |
| --- | --- | --- |
| Reactivity-vs-replay run across >=3 planners with seed budget sufficient for rank stability. | PR #3594 added the pure quantifier in `robot_sf/benchmark/reactivity_ablation.py`. PR #3612 added paired reactive/replay runner mechanics in `scripts/benchmark/run_reactivity_ablation_campaign_issue_3573.py`. PR #3696, PR #3853, PR #3860, PR #3876, and PR #4150 froze the issue #3637 packet/preflight contract, scenario digest, exclusion rules, and rank-stability thresholds for `goal`, `orca`, `social_force`, seeds `101..120`. PR #4492 lets the #3573 runner stamp `study_issue=3637` and `evidence_tier=seed_sufficient_candidate` metadata. | Not met. The durable S20 campaign result files and post-run analysis bundle are absent. Existing artifacts are launch/preflight/analyzer support, not the run itself. |
| Rank stability / confidence reported for the reactivity effect. | PR #3860 and PR #4150 require the analysis contract and frozen thresholds. PR #4462 added `scripts/benchmark/analyze_reactivity_replay_rank_study_issue_3637.py`, which writes `analysis.json`, `frozen_gate_input.json`, `rank_bootstrap_summary.json`, and gate input from completed campaign rows. | Not met as issue evidence. The analyzer exists and is tested, but no completed campaign output has been analyzed into a durable evidence bundle. |
| `"replay = SFM force-off, not trajectory playback"` limitation stated in artifact docs note. | PR #3594 established `REPLAY_LIMITATION` and `REPLAY_IS_TRAJECTORY_PLAYBACK=false`. PR #3696 and PR #4150 carry the limitation through `configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml` and `docs/context/issue_3637_reactivity_replay_rank_study_preflight.md`. PR #4462 rejects contradictory post-run reports. | Met for mechanism, packet, preflight, and analyzer surfaces. Still must be copied into the final post-run evidence bundle when the campaign is executed. |
| Interpretation recorded conservatively in `docs/context`. | PR #3696 created the preflight context note with `plan-preflight only` status and explicit no-benchmark/no-rank-stability/no-paper-facing boundary. This audit note records the current closure decision. | Partially met. Conservative preflight and closure status exist; final interpretation cannot be written until the campaign result and seed-sufficiency gate exist. |
| Run reproduces same command and seed set. | `configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml` pins scenario set, planners, seeds `101..120`, horizon `300`, rank metric, bootstrap settings, and post-run gate command. | Met as a launch contract, not met as empirical evidence because the command has not produced a durable campaign bundle. |
| Paper-facing claim gated on rank-stability evidence and replay caveat. | Launch packet, preflight note, analyzer tests, and latest issue comments all keep paper-facing claims blocked until the post-run seed-sufficiency gate and claim-card review classify the bundle. | Met as a fail-closed policy guard. No paper-facing claim exists from #3637. |

## Closure Decision

Do not close issue #3637. The smallest remaining work is empirical, not another code guardrail:

1. Run the predeclared paired campaign for `goal`, `orca`, and `social_force` on seeds `101..120`.
2. Run the issue #3637 analyzer over the completed campaign output.
3. Promote compact, reviewable artifacts under `docs/context/evidence/`, including checksums or a
   durable raw-output pointer if raw episode files are too large.
4. Record the conservative context interpretation and seed-sufficiency decision as
   `paper_grade_candidate`, `benchmark_evidence_only`, `diagnostic_only`, `needs_more_seeds`, or
   `blocked`.

This closure-audit PR intentionally stops before those steps because the current authorization
forbids compute submission and asks for no full benchmark campaign, no Slurm/GPU submission, and no
paper/dissertation claim edits.

## Local Verification

Audit-time validation is docs/evidence only:

```bash
uv run python scripts/dev/check_docs_evidence_integrity.py
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/evidence/issue_3637_closure_audit_2026-07-05.md \
  --path docs/context/evidence/README.md \
  --path docs/context/INDEX.md
git diff --check
```
