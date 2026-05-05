# Context Notes Workflow

`docs/context/` is the repository's Markdown knowledge base for issue execution history, durable
agent handoff, and reusable reasoning that should not be trapped in chat or PR text.

Use this directory for non-trivial insights, decisions, tradeoffs, validation notes, and execution
context that future contributors or agents are likely to need again.

## When To Update An Existing Note

Prefer updating an existing note when:

* the same issue, planner family, workflow, or benchmark surface is already documented there, 
* the new work changes or clarifies an existing conclusion, 
* or splitting the note would make the decision trail harder to follow.

When you update a note, preserve the current source of truth and remove ambiguity:

* replace outdated statements when the old wording is no longer useful, 
* add dated outcome updates when historical context still matters, 
* and link to the validation commands, artifacts, or follow-up notes that justify the new state.

## When To Create A New Note

Create a new note when the subject is distinct enough that merging it into an existing document
would blur ownership or make the reasoning harder to locate.

Prefer these naming patterns:

* `issue_<number>_<topic>.md` for issue-scoped notes, 
* `<topic>_<date>.md` using `YYYY-MM-DD` for cross-issue audits, release notes, or bounded
  investigations.

## Required Linking

Every durable context note should link to the smallest useful set of related surfaces:

* the GitHub issue or PR that motivated the work, 
* canonical docs or configs that define the contract, 
* validation commands, artifacts, or output paths that support the conclusion, 
* predecessor or successor notes when a document is continued or superseded.

If the note changes repository guidance, also link it from a normal entry point such as
`docs/README.md` , `docs/dev_guide.md` , `AGENTS.md` , or `docs/ai/repo_overview.md` .

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

* the goal or decision, 
* the assumptions made and why they matter, 
* the key evidence or reasoning, 
* the validation path, 
* the current conclusion or follow-up boundary.

Avoid turning `docs/context/` into a scratchpad. Capture what future readers need to reuse the
knowledge, not every transient iteration detail.

## Skills And Entry Points

* Repository rule: [AGENTS.md](../../AGENTS.md)
* Contributor workflow: [docs/dev_guide.md](../dev_guide.md)
* Docs index entry: [docs/README.md](../README.md)
* AI-facing orientation: [docs/ai/repo_overview.md](../ai/repo_overview.md)
* Note-maintenance skill:
  [.agents/skills/context-note-maintainer/SKILL.md](../../.agents/skills/context-note-maintainer/SKILL.md)

## Example

* [docs/context/issue_796_agent_knowledge_capture_policy.md](issue_796_agent_knowledge_capture_policy.md)
* [docs/context/issue_805_teb_corridor_commitment_iteration.md](issue_805_teb_corridor_commitment_iteration.md)

## Feature Extractor Notes

* [Issue #193 Feature Extractor Evaluation](./issue_193_feature_extractor_evaluation.md)
  GPU throughput microbenchmark + 32 K PPO comparison of DynamicsExtractor vs MLP/CNN/Attention; 
  recommends `mlp_small` as new default for fresh training runs.
* [Issue #193 Feature Extractor Optuna Study](./issue_193_feature_extractor_optuna_study.md)
  4 M-step SLURM sweep infrastructure, DB classification, and April 20 final pre-screen analysis; 
`feat_sweep_4m_array.db` is the current evidence surface, with longer 10 M+ validation still
  required before promotion.
* [Issue #835 Lightweight CNN Divergence Triage](./issue_835_lightweight_cnn_divergence.md)
  bounded 32 K rerun with PPO gradient and feature diagnostics; the issue-193 catastrophic
`lightweight_cnn` final drop did not reproduce, so the extractor remains experimental without an
  immediate architecture change.
* [Issue #850 PPO Collision Failures](./issue_850_ppo_collision_failures.md)
  follow-up diagnostics for the issue-193 `dyn_large_med` hold-out collision failures and the
  config-first safety-reward mitigation candidate.
* [Issue #863 SVG/Model Log Spam](./issue_863_svg_model_log_spam.md)
  log dedupe and PPO evaluation phase-marker decision for issue-791 long-run triage.

## Performance Notes

* [Issue #513 High-Density Perf Gate Calibration](./issue_513_high_density_perf_gate.md)
  keeps `classic_cross_trap_high` advisory because no stable local trend-history window was
  available; documents the rerun evidence and non-blocking policy.
* [Issue #867 PPO Evaluation Reload Profile](./issue_867_ppo_eval_reload_profile.md)
  measurement-only issue-791 evaluation probe showing cached predictive-model reloads are small
  compared with shared cold startup and first-step overhead.
* [Issue #815 SAC Cold/Warm Performance Profile](./issue_815_sac_perf_cold_warm.md)
  cold/warm harness evidence showing the remaining issue-815 SAC simulator cost is localized to
  cold startup and lazy first-step initialization, not warm steady-state stepping.

## Planner Integration Notes

* [External Planner Reuse Checklist](./external_planner_reuse_checklist.md)
* [Issue #562 SocNavBench Re-Entry Gate](./issue_562_socnav_bench_reentry.md)
  defines the fail-fast focused probe and concrete keep-out/re-entry criteria for `socnav_bench`

  before it can be reconsidered for paper-facing benchmarks.
* [Issue #792 Social-Jym Source-Harness Reproduction](./issue_792_social_jym_source_harness.md)
  resolves the prior SSH-submodule blocker via HTTPS in an isolated side environment and proves a
  minimal upstream `SocialNav` reset plus `SARL` policy step, without claiming Robot SF benchmark
  integration.
* [Issue #905 Social-Jym Wrapper Spike](./issue_905_social_jym_wrapper_spike.md)
  proves a one-step Robot SF smoke loop through a thin random-action upstream `SARL` wrapper and
  records the remaining parity/provenance boundary.
* [Issue #907 Social-Jym SARL Wrapper Parity](./issue_907_social_jym_sarl_parity.md)
  proves controlled one-human SARL input parity for the wrapper and quantifies holonomic-to-unicycle
  projection loss, while keeping benchmark integration blocked on trained-policy provenance.
* [Issue #909 Social-Jym Trained-Policy Provenance](./issue_909_social_jym_policy_provenance.md)
  records the fail-closed finding that the pinned upstream checkout references SARL/SARL-PPO
  artifacts but does not include reproducible trained policy weights, so benchmark smoke remains
  unjustified.
* [Issue #626 SoNIC Source Harness Probe](./issue_626_sonic_source_harness_probe.md)
* [Issue #627 SoNIC Wrapper Follow-up](./issue_627_sonic_wrapper_followup.md)
* [Policy Search Context](./policy_search/README.md) - file-based candidate registry, staged local evaluation funnel, and SLURM handoff notes for the current non-training policy-search workstream.
* [Issue #926 Policy Stack V1 Contract](issue_926_policy_stack_v1_contract.md)
  defines the minimal `policy_stack_v1` portfolio-planner contract, diagnostics, and benchmark
  claim boundary before runtime implementation under #871.

## Map Coverage Notes

* [Issue #435 Map Coverage Flow](./issue_435_map_coverage_flow.md)
  parent flow state for real-world maps, SocNavBench import, and map-quality repair issues.
* [Issue #328 Real-World Map Parent Tracker](./issue_328_real_world_map_parent.md)
  parent/child split, current child issue state, and shared validation contract for real-world
  benchmark maps.

## Reasoning Notes

Design and decision rationale notes live in `docs/context/reasoning/` when the goal is to preserve
why a change was made rather than a full issue execution transcript.

* [Issue #592 Hybrid Obstacle-Context Predictor Design](./issue_592_hybrid_obstacle_predictor_design.md)
  scopes the obstacle-conditioned predictive-model idea into a feature-baseline-first experiment
  path with proof gates before any grid/CNN or obstacle-node graph prototype.
* [Issue #932 Hybrid Portfolio Diagnostics](./issue_932_hybrid_portfolio_diagnostics.md)
  records the first small policy-stack runtime diagnostics slice: selected-head counts, fallback
  counts, and last-decision metadata on `HybridPortfolioAdapter` .
* [Issue #938 Hybrid Portfolio Last Decision](./issue_938_hybrid_portfolio_last_decision.md)
  adds a step-level `HybridPortfolioAdapter.last_decision()` accessor consistent with nearby
  planner diagnostics APIs.
* [Issue #589 Public Leaderboard MVP Boundary](./issue_589_public_leaderboard_mvp.md)
  records the no-implementation-now decision, future PR-based MVP boundary, and prerequisites for
  any public planner leaderboard work.

## Execution Workflow Notes

* [SLURM Multi-Worktree Branch Workflow](slurm_multi_worktree_branch_workflow.md) - branch-isolated
  SLURM submissions from a shared login node, including `local.machine.md` symlink guidance and
  virtualenv boundaries.
* [Issue #845 Slurm Utilization Probe](issue_845_slurm_utilization_probe.md) - `sstat`/`sacct`/`seff`
  evidence collection path for diagnosing low CPU utilization without launching new jobs.
* [Issue #869 Adversarial Runner](issue_869_adversarial_runner.md) - programmable adversarial
  scenario search API, bundle contract, certification boundary, and deferred optimizer scope.
* [Issue #923 Multi-Ped Adversarial Candidate Schema](issue_923_multi_ped_adversarial_schema.md) -
  schema-only first slice under #870 for scripted multi-pedestrian adversarial candidates.
* [Issue #936 Multi-Ped Adversarial Overrides](issue_936_multi_ped_adversarial_overrides.md)
  records the pure-data materializer from `adversarial-multi-ped.v1` configs to scenario-loader
`single_pedestrians` override dictionaries, stacked on the issue #923 schema PR.
* [Issue #944 Multi-Ped Adversarial Scenario Payload](issue_944_multi_ped_adversarial_scenario_payload.md)
  adds a template-merging manifest payload materializer for `adversarial-multi-ped.v1` configs, 
  stacked on the issue #936 override materializer.
* [Issue 868 Scenario Certification](issue_868_scenario_certification.md) - `scenario_cert.v1`
  scope, public surfaces, validation path, and known limits.
* [Issue #930 CARLA T0 Neutral Export Schema](issue_930_carla_t0_export_schema.md)
  records the import-safe `robot_sf_carla_bridge` package, `carla-replay-export.v1` schema, and
  missing-CARLA `not-available` guard for future oracle replay work.
* [Issue #934 CARLA T0 Export Builder API](issue_934_carla_t0_export_builder.md)
  adds typed, schema-validated builder objects for `carla-replay-export.v1` payload construction, 
  stacked on the issue #930 bridge package.
* [Issue #940 CARLA T0 Export Read Helper](issue_940_carla_t0_export_read_helper.md)
  adds a schema-validating `read_export_payload(...)` boundary for exported replay JSON, stacked on
  the issue #934 builder API.
* [Issue #942 CARLA T0 MapDefinition Adapter](issue_942_carla_t0_map_definition_adapter.md)
  converts already-certified Robot-SF `MapDefinition` objects into schema-valid neutral export
  payloads, stacked on the issue #940 read helper.
* [Issue #946 CARLA T0 Scenario Entry Export](issue_946_carla_t0_scenario_entry_export.md)
  exports one scenario-loader entry through `scenario_cert.v1` into a neutral CARLA T0 payload, 
  stacked on the issue #942 map-definition adapter.
* [Issue #948 CARLA T0 Scenario File Export](issue_948_carla_t0_scenario_file_export.md)
  batch-loads scenario manifests into ordered neutral export payload records, stacked on the issue
  #946 scenario-entry helper.
* [Issue #950 CARLA T0 Export Record Writer](issue_950_carla_t0_export_record_writer.md)
  writes ordered neutral export records to deterministic JSON files plus a local manifest, stacked
  on the issue #948 scenario-file helper.
* [Issue #952 CARLA T0 Export CLI](issue_952_carla_t0_export_cli.md)
  adds a CARLA-free command-line wrapper over scenario-file export and deterministic record writing, 
  stacked on the issue #950 record writer.
* [Issue #954 CARLA T0 Export CLI Packaging](issue_954_carla_t0_export_cli_packaging.md)
  exposes the issue #952 CLI as an installable project script while keeping the package CARLA-free.
* [Issue #956 CARLA T0 Export Manifest Reader](issue_956_carla_t0_export_manifest_reader.md)
  adds a local manifest reader for issue #950 export records, stacked on the issue #954 CLI package
  surface.
* [Issue #958 CARLA T0 Manifest Validation CLI](issue_958_carla_t0_manifest_validation_cli.md)
  exposes the issue #956 manifest reader through a CARLA-free project script.
* [Issue #960 CARLA T0 Manifest Payload Paths](issue_960_carla_t0_manifest_payload_paths.md)
  resolves issue #950 manifest entries to local payload paths while rejecting unsafe path escapes.
* [Issue #928 CARLA T0/T1 Oracle Replay Contract](issue_928_carla_t0_t1_replay_contract.md)
  documents the first CARLA transfer boundary: neutral export first, optional oracle replay later, 
  and fail-closed `not-available` / `failed` statuses instead of fallback parity claims.
* [Issue #962 CARLA T0 Manifest Payload Loader](issue_962_carla_t0_manifest_payload_loader.md)
  loads and validates all payloads referenced by a local T0 export manifest, stacked on the issue
  #960 path resolver.
* [Issue #964 CARLA T0 Batch Validation CLI](issue_964_carla_t0_batch_validation_cli.md)
  exposes the issue #962 manifest payload loader through a CARLA-free batch validation project
  script.
- [Issue #966 CARLA T0 Batch Validation JSON Summary](issue_966_carla_t0_batch_validation_json_summary.md)
  adds deterministic machine-readable output to the CARLA-free batch validation CLI.
- [Issue #968 CARLA Runtime Availability Guard](issue_968_carla_runtime_availability_guard.md)
  adds a strict optional-CARLA import guard for future replay entry points.
- [Issue #970 CARLA Availability Check CLI](issue_970_carla_availability_check_cli.md)
  exposes CARLA bridge availability metadata through a CARLA-free project script.
- [Issue #972 CARLA Availability CLI Require Mode](issue_972_carla_availability_cli_require_mode.md)
  adds fail-closed availability checking for CARLA-dependent setup gates.
- [Issue #974 CARLA Availability Boolean Field](issue_974_carla_availability_boolean_field.md)
  adds an explicit boolean to CARLA availability metadata and CLI JSON output.
- [Issue #976 CARLA Availability Schema Version](issue_976_carla_availability_schema_version.md)
  adds a schema version to CARLA availability metadata for stable script consumption.
- [Issue #978 CARLA Availability JSON Schema](issue_978_carla_availability_json_schema.md)
  adds a JSON Schema for validating CARLA availability metadata.

## DreamerV3 Notes

* [DreamerV3 Program Full Handoff (2026-04-28)](dreamerv3_program_full_handoff_2026_04_28.md)
  Consolidated execution plan for issues #578, #608, #609, #782, and #789.
* [DreamerV3 BR-08 Full Progress (2026-04-29)](dreamerv3_br08_full_progress_2026_04_29.md)
  Run-level outcome and diagnostics summary for Slurm 12159.
* [DreamerV3 Program Close-Out (2026-04-30)](dreamerv3_program_close_out_2026_04_30.md)
  Program-level stop decision and closure rationale after the probe/gate/full sequence.
* [Issue 782: DreamerV3 world-model pretraining design](issue_782_dreamerv3_pretraining_design.md)
  Inventory of reusable rollout sources plus the recommended proof-first pretraining path.
* [Issue 789: DreamerV3 multimodal encoder stop note](issue_789_dreamer_multimodal_encoder.md)
  Fail-closed investigation result for mixed observation spaces on Ray 2.53.0 DreamerV3.
