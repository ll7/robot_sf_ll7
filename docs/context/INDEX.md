# Context Retrieval Index

Status: first-pass retrieval surface for GitHub issue #1714.

Use this file before browsing the full `docs/context/` tree. The directory has hundreds of
issue-scoped notes plus tracked evidence bundles; broad reads should start from the smallest
current surface below, then expand only when the task needs more detail.

This Markdown-first catalog preserves the retrieval boundary described in
[docs/ai/retrieval_deferral.md](../ai/retrieval_deferral.md): no database, vector store, or MCP
retrieval layer is introduced until repository evidence justifies that extra infrastructure.

## Retrieval Rules

1. Start with the task domain, not the newest note.
2. Prefer canonical or current notes over historical execution logs.
3. Open predecessor, successor, or evidence links only when a claim depends on them.
4. Treat `output/` paths mentioned in old notes as local-only unless a tracked manifest, release,
   W&B artifact, or `docs/context/evidence/` copy is linked.
5. If a touched note is stale or superseded, update the status line or point it at the current
   source of truth before adding more prose.

## Status Vocabulary

Use these status labels when adding or refreshing notes:

- `canonical`: the preferred current entry point for a domain or issue family.
- `current`: still accurate for its issue or bounded investigation.
- `historical`: useful background, but not the current decision surface.
- `superseded`: kept only for traceability; must link to the replacement note.
- `blocked`: work cannot proceed without named artifacts, hardware, data, or maintainer input.
- `evidence`: compact tracked proof under `docs/context/evidence/`.

## Domain Entry Points

| Domain | Start Here | Why |
| --- | --- | --- |
| Agent issue loops and GitHub workflow | [goal_driven_agent_loops_2026-05-13.md](goal_driven_agent_loops_2026-05-13.md), [issue_713_batch_first_issue_workflow.md](issue_713_batch_first_issue_workflow.md) | Canonical loop boundaries, queue policy, and API batching rules. |
| Open-issue execution state | [open_issue_execution_improvement_plan_2026-05-30.md](open_issue_execution_improvement_plan_2026-05-30.md), [open_issues_training_split_audit_2026-05-30.md](open_issues_training_split_audit_2026-05-30.md) | Current issue-queue audit and training/resource split context. |
| Context-note maintenance | [README.md](README.md) | Contribution rules for creating, updating, linking, and pruning notes. |
| Benchmark fallback and claim safety | [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md), [issue_1436_reproducibility_flaky_acceptance.md](issue_1436_reproducibility_flaky_acceptance.md) | Fail-closed fallback policy and validation rerun boundaries. |
| Observation tracks and learned policies | [issue_1612_observation_track_architecture.md](issue_1612_observation_track_architecture.md), [issue_1618_learned_policy_adapter_interface.md](issue_1618_learned_policy_adapter_interface.md), [issue_1685_dummy_learned_policy_adapter.md](issue_1685_dummy_learned_policy_adapter.md) | Current observation-track architecture, adapter boundary, and dummy fixture proof. |
| Model and artifact provenance | [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md), [../model_registry_publication.md](../model_registry_publication.md) | Shared language for exploratory outputs, durable evidence, releases, and promoted models. |
| Policy-search portfolio | [policy_search/README.md](policy_search/README.md), [policy_search/candidate_registry.yaml](policy_search/candidate_registry.yaml), [policy_search/learned_policy_registry.md](policy_search/learned_policy_registry.md) | Subtree-specific index and registries for policy candidates and learned policy evidence. |
| CI and validation runtime | [issue_1653_ci_runtime_slice.md](issue_1653_ci_runtime_slice.md) | Current timing baseline, implemented instrumentation, and next CI-runtime targets. |
| Root layout and repository hygiene | [issue_1690_root_layout_inventory.md](issue_1690_root_layout_inventory.md), [issue_1583_high_risk_root_boundaries.md](issue_1583_high_risk_root_boundaries.md) | Root layout inventory and high-risk path boundaries. |
| Simulation trace and visualization workbench | [issue_1689_simulation_trace_export_schema.md](issue_1689_simulation_trace_export_schema.md), [open_issue_execution_improvement_plan_2026-05-30.md](open_issue_execution_improvement_plan_2026-05-30.md) | Trace export schema plus current workbench sequencing and deferred viewer boundary. |
| AMV, CARLA, and external data blockers | [slurm_issue_batch_status_2026-05-21.md](slurm_issue_batch_status_2026-05-21.md), [issue_1584_socnav_unavailable_row_policy.md](issue_1584_socnav_unavailable_row_policy.md), [issue_1111_carla_setup_smoke.md](issue_1111_carla_setup_smoke.md) | Resource-gated issue status, unavailable-row policy, and CARLA setup boundary. |

## Evidence Bundles

Use [evidence/README.md](evidence/README.md) for small tracked evidence copies. Evidence bundles
are review aids, not replacements for canonical configs, commands, seeds, commit SHAs, or durable
artifact pointers.

Current high-signal bundles include:

- `evidence/issue_1608_seed_sensitivity_2026-05-30/`
- `evidence/issue_1674_topology_hypothesis_diagnostics_2026-05-30/`
- `evidence/issue_1692_topology_hypothesis_probe_2026-05-30/`

## Pruning And Refactoring Rules

- Do not delete historical notes just because they are old; mark them `historical` or `superseded`
  when they still explain a decision trail.
- Prefer one canonical note per active issue family, with older notes linking forward.
- Move repeated operational instructions into `AGENTS.md`, `docs/dev_guide.md`, `.agents/skills/`,
  or `docs/ai/` when they are no longer issue-specific.
- Keep large generated artifacts out of this tree. Promote only compact, reviewable summaries or
  manifests into `docs/context/evidence/`.
- When a note only points at unavailable local `output/` files and has no durable pointer, classify
  it as historical or blocked instead of treating it as reusable evidence.

## Adding A New Index Row

Add a row here when a note becomes a stable entry point for future agents. Include the domain, the
smallest useful starting file, and why it should be read before lower-level history.
