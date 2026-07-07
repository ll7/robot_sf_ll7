# Issue #3637 Closure Audit

Plain-language summary: issue #3637 is not ready to close. The merged pull requests
through PR #4716 provide the reactivity-vs-replay mechanism, launch packet, preflight,
post-run analyzer, packet-backed campaign launcher, and machine-readable integration
handoff. The issue's central empirical acceptance criteria still lack the predeclared
three-planner, 20-seed campaign output and post-run rank-stability analysis bundle.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/3637>
- Original audit date: 2026-07-05
- Latest refresh: 2026-07-07 after PR #4707 and PR #4716 merged.
- Latest issue guidance reviewed: 2026-07-07 01:49 UTC comment, which keeps the
  issue open until the predeclared campaign run and analyzer bundle exist.
- Fragmentation guard: PR #4707 and PR #4716 both merged within 24 hours of this
  refresh. This document is therefore a consolidation audit of the existing contract
  and remaining blocker, not another guardrail/checker packet.

## Merged PR Evidence Reviewed

| PR | Evidence contribution |
| --- | --- |
| [#3594](https://github.com/ll7/robot_sf_ll7/pull/3594), `4e8899e96295dd054895cb458e42ad869b56a87b` | Added `assess_reactivity_ablation`, canonical replay limitation, reactive-minus-replay deltas, and rank-sensitivity quantifier. |
| [#3612](https://github.com/ll7/robot_sf_ll7/pull/3612), `3134353784acd19d5d46c0976298607d396cf44e` | Added paired reactive/replay campaign mechanics in the #3573 runner. |
| [#3696](https://github.com/ll7/robot_sf_ll7/pull/3696), `69265c2296e3d703d748fa71315a18aad6970741` | Added issue #3637 launch packet and plan-level preflight. |
| [#3853](https://github.com/ll7/robot_sf_ll7/pull/3853), `5e566f391f0f6b3605ea55792d3fb3d0b3f5ddef` | Enforced the predeclared scenario-set digest. |
| [#3860](https://github.com/ll7/robot_sf_ll7/pull/3860), `bb3260145fcfd0877170c1c142b403449caa7579` | Required the rank-analysis contract in the launch packet/preflight path. |
| [#3876](https://github.com/ll7/robot_sf_ll7/pull/3876), `dd82bd21df1990bff4224be69ced9ce46e054756` | Hardened degraded/fallback exclusion handling. |
| [#4150](https://github.com/ll7/robot_sf_ll7/pull/4150), `fb74c5091410581a6292b32062e117020ce48c4e` | Froze seed-sufficiency thresholds, including `target_ci_half_width: 0.10`. |
| [#4462](https://github.com/ll7/robot_sf_ll7/pull/4462), `4c5151860` | Added fail-closed post-run analyzer and seed-sufficiency artifact generation. |
| [#4492](https://github.com/ll7/robot_sf_ll7/pull/4492), `2559e7ade` | Added runner metadata overrides so #3637 runs can preserve study issue and evidence tier. |
| [#4590](https://github.com/ll7/robot_sf_ll7/pull/4590), `c3bcbdf4c` | Added the first closure-audit evidence note and kept the no-claim boundary explicit. |
| [#4707](https://github.com/ll7/robot_sf_ll7/pull/4707), `b7ae121d3` | Added packet-backed issue #3637 campaign launcher and tests proving packet values drive the #3573 runner. |
| [#4716](https://github.com/ll7/robot_sf_ll7/pull/4716), `a5e7dfa5e` | Added machine-readable `integration_report` to the campaign report with closure boundary `keep_open_until_analysis_artifacts_exist`. |

## Acceptance Criteria Mapping

| Issue #3637 criterion | Merged evidence | Closure status |
| --- | --- | --- |
| Reactivity-vs-replay run across at least three planners with seed budget sufficient for rank stability. | Launch packet pins `goal`, `orca`, `social_force`, seeds `101..120`, scenario set, horizon, and exclusion rules. PR #4707 adds the packet-backed launcher that runs those packet values through the existing campaign runner. PR #4716 records the required post-run integration handoff. | **Not met.** No durable completed campaign output exists for the predeclared three-planner, 20-seed matrix. |
| Rank stability / confidence reported for the reactivity effect. | PR #4462 adds the analyzer expected to emit `analysis.json`, `frozen_gate_input.json`, `rank_bootstrap_summary.json`, and `per_planner_condition_metrics.csv`. PR #4716 lists those as required post-run artifacts in the campaign report integration contract. | **Not met.** Analyzer output depends on completed campaign JSONL files that do not exist yet. |
| `"replay" = social-force model force-off, not trajectory playback` limitation stated in artifact docs note. | PR #3594 establishes the canonical limitation and `REPLAY_IS_TRAJECTORY_PLAYBACK=false`. PR #3696, PR #4150, PR #4462, and PR #4716 carry or enforce the limitation through packet, preflight, analyzer, and integration-report surfaces. | **Partially met.** Mechanism and planned artifact surfaces carry the caveat; the final empirical evidence bundle still must include it. |
| Interpretation recorded conservatively in `docs/context`. | PR #3696 records preflight-only status. PR #4590 records the first closure audit. This refresh records the post-#4716 closure boundary against the full merged PR chain. | **Partially met.** Conservative preflight and closure-audit interpretation exists, but final interpretation requires empirical analyzer outputs. |
| Run reproduces the same command and seed set. | The launch packet and PR #4707 launcher make the packet the source of truth for planners, seeds, scenario set, horizon, output root, and analyzer command. | **Launch contract met; empirical criterion not met.** Reproducibility cannot be claimed until the campaign has produced durable outputs. |
| Reactivity-rank claim supported by rank-stability evidence. | Code and config now define the intended gate and artifacts. | **Not met.** No rank-stability evidence exists. |

## Closure Decision

Keep issue #3637 open. The smallest remaining work that can satisfy the issue is empirical, not
another schema or guardrail slice:

1. Run the predeclared paired campaign for `goal`, `orca`, and `social_force` on seeds `101..120`.
2. Run `scripts/benchmark/analyze_reactivity_replay_rank_study_issue_3637.py` against the completed
   campaign output.
3. Promote compact, reviewable analyzer artifacts under `docs/context/evidence/`, with durable raw
   output pointers or checksums rather than raw episode JSONL in git.
4. Record a conservative context interpretation with one of the predeclared evidence outcomes:
   `paper_grade_candidate`, `benchmark_evidence_only`, `diagnostic_only`, `needs_more_seeds`, or
   `blocked`.

This refresh did not run the campaign and does not make a benchmark, rank-stability, paper-facing,
or dissertation claim. It also did not submit Slurm/GPU work.

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
