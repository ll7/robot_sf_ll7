# Context Notes Workflow

`docs/context/` is the repository's Markdown knowledge base for issue execution history, durable
agent handoff, and reusable reasoning that should not be trapped in chat or PR text.

Use this directory for non-trivial insights, decisions, tradeoffs, validation notes, and execution
context that future contributors or agents are likely to need again.

## When To Update An Existing Note

Prefer updating an existing note when:

- the same issue, planner family, workflow, or benchmark surface is already documented there,
- the new work changes or clarifies an existing conclusion,
- or splitting the note would make the decision trail harder to follow.

When you update a note, preserve the current source of truth and remove ambiguity:

- replace outdated statements when the old wording is no longer useful,
- add dated outcome updates when historical context still matters,
- and link to the validation commands, artifacts, or follow-up notes that justify the new state.

## When To Create A New Note

Create a new note when the subject is distinct enough that merging it into an existing document
would blur ownership or make the reasoning harder to locate.

Prefer these naming patterns:

- `issue_<number>_<topic>.md` for issue-scoped notes,
- `<topic>_<date>.md` using `YYYY-MM-DD` for cross-issue audits, release notes, or bounded
  investigations.

## Required Linking

Every durable context note should link to the smallest useful set of related surfaces:

- the GitHub issue or PR that motivated the work,
- canonical docs or configs that define the contract,
- validation commands, artifacts, or output paths that support the conclusion,
- predecessor or successor notes when a document is continued or superseded.

If the note changes repository guidance, also link it from a normal entry point such as
`docs/README.md`, `docs/dev_guide.md`, `AGENTS.md`, or `docs/ai/repo_overview.md`.

## Outdated And Superseded Content

Touched notes must not leave stale conclusions ambiguous.

If a note is still the canonical surface, update it in place.

If a note should remain for history but is no longer current, mark that clearly near the top:

```md
> Status: superseded by `docs/context/issue_999_new_note.md` on 2026-04-09.
> Keep this note only for historical context.
```

If the note is no longer useful even as history, remove the outdated statement instead of stacking
contradictory prose.

## Lightweight Structure

Use the smallest structure that keeps the note reusable. Most notes should include:

- the goal or decision,
- the assumptions made and why they matter,
- the key evidence or reasoning,
- the validation path,
- the current conclusion or follow-up boundary.

Avoid turning `docs/context/` into a scratchpad. Capture what future readers need to reuse the
knowledge, not every transient iteration detail.

## Skills And Entry Points

- Repository rule: [AGENTS.md](../../AGENTS.md)
- Contributor workflow: [docs/dev_guide.md](../dev_guide.md)
- Docs index entry: [docs/README.md](../README.md)
- AI-facing orientation: [docs/ai/repo_overview.md](../ai/repo_overview.md)
- Note-maintenance skill:
  [.agents/skills/context-note-maintainer/SKILL.md](../../.agents/skills/context-note-maintainer/SKILL.md)

## Example

- [docs/context/issue_796_agent_knowledge_capture_policy.md](issue_796_agent_knowledge_capture_policy.md)
- [docs/context/issue_805_teb_corridor_commitment_iteration.md](issue_805_teb_corridor_commitment_iteration.md)

## Feature Extractor Notes

- [Issue #193 Feature Extractor Evaluation](./issue_193_feature_extractor_evaluation.md)
  GPU throughput microbenchmark + 32 K PPO comparison of DynamicsExtractor vs MLP/CNN/Attention;
  recommends `mlp_small` as new default for fresh training runs.
- [Issue #193 Feature Extractor Optuna Study](./issue_193_feature_extractor_optuna_study.md)
  4 M-step SLURM sweep infrastructure, DB classification, and April 20 final pre-screen analysis;
  `feat_sweep_4m_array.db` is the current evidence surface, with longer 10 M+ validation still
  required before promotion.
- [Issue #835 Lightweight CNN Divergence Triage](./issue_835_lightweight_cnn_divergence.md)
  bounded 32 K rerun with PPO gradient and feature diagnostics; the issue-193 catastrophic
  `lightweight_cnn` final drop did not reproduce, so the extractor remains experimental without an
  immediate architecture change.
- [Issue #850 PPO Collision Failures](./issue_850_ppo_collision_failures.md)
  follow-up diagnostics for the issue-193 `dyn_large_med` hold-out collision failures and the
  config-first safety-reward mitigation candidate.
- [Issue #863 SVG/Model Log Spam](./issue_863_svg_model_log_spam.md)
  log dedupe and PPO evaluation phase-marker decision for issue-791 long-run triage.

## Performance Notes

- [Issue #513 High-Density Perf Gate Calibration](./issue_513_high_density_perf_gate.md)
  keeps `classic_cross_trap_high` advisory because no stable local trend-history window was
  available; documents the rerun evidence and non-blocking policy.
- [Issue #867 PPO Evaluation Reload Profile](./issue_867_ppo_eval_reload_profile.md)
  measurement-only issue-791 evaluation probe showing cached predictive-model reloads are small
  compared with shared cold startup and first-step overhead.
- [Issue #815 SAC Cold/Warm Performance Profile](./issue_815_sac_perf_cold_warm.md)
  cold/warm harness evidence showing the remaining issue-815 SAC simulator cost is localized to
  cold startup and lazy first-step initialization, not warm steady-state stepping.

## Planner Integration Notes

- [External Planner Reuse Checklist](./external_planner_reuse_checklist.md)
- [Issue #626 SoNIC Source Harness Probe](./issue_626_sonic_source_harness_probe.md)
- [Issue #627 SoNIC Wrapper Follow-up](./issue_627_sonic_wrapper_followup.md)
- [Policy Search Context](./policy_search/README.md) - file-based candidate registry, staged local evaluation funnel, and SLURM handoff notes for the current non-training policy-search workstream.

## Reasoning Notes

Design and decision rationale notes live in `docs/context/reasoning/` when the goal is to preserve
why a change was made rather than a full issue execution transcript.

- [Issue #589 Public Leaderboard MVP Boundary](./issue_589_public_leaderboard_mvp.md)
  records the no-implementation-now decision, future PR-based MVP boundary, and prerequisites for
  any public planner leaderboard work.

## Execution Workflow Notes

- [SLURM Multi-Worktree Branch Workflow](slurm_multi_worktree_branch_workflow.md) - branch-isolated
  SLURM submissions from a shared login node, including `local.machine.md` symlink guidance and
  virtualenv boundaries.
- [Issue #869 Adversarial Runner](issue_869_adversarial_runner.md) - programmable adversarial
  scenario search API, bundle contract, certification boundary, and deferred optimizer scope.
- [Issue 868 Scenario Certification](issue_868_scenario_certification.md) - `scenario_cert.v1`
  scope, public surfaces, validation path, and known limits.

## DreamerV3 Notes

- [DreamerV3 Program Full Handoff (2026-04-28)](dreamerv3_program_full_handoff_2026_04_28.md)
  Consolidated execution plan for issues #578, #608, #609, #782, and #789.
- [DreamerV3 BR-08 Full Progress (2026-04-29)](dreamerv3_br08_full_progress_2026_04_29.md)
  Run-level outcome and diagnostics summary for Slurm 12159.
- [DreamerV3 Program Close-Out (2026-04-30)](dreamerv3_program_close_out_2026_04_30.md)
  Program-level stop decision and closure rationale after the probe/gate/full sequence.
- [Issue 782: DreamerV3 world-model pretraining design](issue_782_dreamerv3_pretraining_design.md)
  Inventory of reusable rollout sources plus the recommended proof-first pretraining path.
- [Issue 789: DreamerV3 multimodal encoder stop note](issue_789_dreamer_multimodal_encoder.md)
  Fail-closed investigation result for mixed observation spaces on Ray 2.53.0 DreamerV3.
