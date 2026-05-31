# Context Retrieval Index

Issue: [#1714](https://github.com/ll7/robot_sf_ll7/issues/1714)

This is the first stop for broad `docs/context/` retrieval. Use it to choose the smallest current
context surface before reading issue history in bulk.

## Retrieval Order

1. Start with this index when the task says "context", "history", "handoff", "what happened", or
   mentions a broad subsystem.
2. Read the domain entry points below, then follow only the linked notes that match the task.
3. Use [README.md](README.md) when creating or maintaining notes, because it defines the note
   workflow and full discoverability list.
4. Use [memory/MEMORY.md](../../memory/MEMORY.md) for stable cross-session facts that should outlive
   one issue or PR.
5. Treat `output/` as disposable local state. Durable evidence belongs in small tracked manifests
   under [evidence/](evidence/README.md) or in an external artifact store with a tracked pointer.

## Status Rules

- `Current`: use as an active source of truth.
- `Historical`: useful background, but check newer linked notes before relying on conclusions.
- `Superseded`: keep only for provenance; use the replacement note named near the top.
- `Evidence`: tracked compact proof or manifest; do not treat missing local `output/` files as
  durable dependencies.
- `Proposal`: planned or exploratory guidance; do not cite as completed implementation evidence.

When updating a note, mark stale or superseded claims in place instead of stacking contradictory
paragraphs.

Machine-readable sidecar: [catalog.yaml](catalog.yaml) records the curated entry points below with
`status` and `freshness` metadata. Validate it with
`uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml`
or the normal diff wrapper.

## Pruning And Refactoring Rules

- Prune, rewrite, rename, or delete stale notes more aggressively when the reason is unambiguous,
  the note no longer provides reusable value, and benchmark or paper-facing evidence pointers are
  preserved.
- If two notes cover the same decision, keep the newer canonical note current and mark the older
  note `Historical` or `Superseded` near the top with a pointer to the replacement.
- If a note is issue-local and no longer reusable, keep it linked from the relevant issue-specific
  entry only, or remove it from this index when the issue/PR record already preserves the needed
  provenance.
- If a note supports a benchmark or paper-facing claim, preserve the evidence pointer even when the
  prose is shortened.
- Add a note to this index only when it is a current domain entry point, a durable policy boundary,
  or a curated context-pack ingredient.

## Canonical Context Surfaces

| Area | Current entry points | Use for |
|---|---|---|
| Agent workflow | [goal_driven_agent_loops_2026-05-13.md](goal_driven_agent_loops_2026-05-13.md), [open_issue_execution_improvement_plan_2026-05-30.md](open_issue_execution_improvement_plan_2026-05-30.md), [issue_713_batch_first_issue_workflow.md](issue_713_batch_first_issue_workflow.md) | Issue-to-PR loops, queue exhaustion, batching, issue splitting, and GitHub workflow policy. |
| Context architecture | This file, [../ai/context_packing.md](../ai/context_packing.md), [../ai/retrieval_deferral.md](../ai/retrieval_deferral.md), [issue_728_coding_agents_compatibility.md](issue_728_coding_agents_compatibility.md) | Context-pack decisions, optional external tools, Markdown-first retrieval, and cross-agent compatibility. |
| Benchmark evidence policy | [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md), [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md), [issue_1436_reproducibility_flaky_acceptance.md](issue_1436_reproducibility_flaky_acceptance.md), [../code_review.md](../code_review.md) | Fail-closed fallback/degraded handling, artifact classes, reproducibility, and benchmark review traps. |
| Benchmark release and reports | [../benchmark_release_protocol.md](../benchmark_release_protocol.md), [../benchmark_camera_ready.md](../benchmark_camera_ready.md), [issue_1542_manuscript_claim_evidence_map.md](issue_1542_manuscript_claim_evidence_map.md), [issue_750_paper_results_handoff.md](issue_750_paper_results_handoff.md) | Camera-ready runs, paper-facing claims, release manifests, and results handoff. |
| Planner integration | [../ai/planner_zoo_context.md](../ai/planner_zoo_context.md), [../benchmark_planner_family_coverage.md](../benchmark_planner_family_coverage.md), [issue_1530_optional_preflight_audit.md](issue_1530_optional_preflight_audit.md), [issue_1360_external_teb_assessment.md](issue_1360_external_teb_assessment.md) | Planner-family coverage, optional planner preflights, adapter provenance, and benchmark readiness. |
| Learned-policy and training | [open_issues_training_split_audit_2026-05-30.md](open_issues_training_split_audit_2026-05-30.md), [policy_search/learned_policy_registry.md](policy_search/learned_policy_registry.md), [policy_search/contracts/learned_local_policy_eligibility.md](policy_search/contracts/learned_local_policy_eligibility.md), [issue_1618_learned_policy_adapter_interface.md](issue_1618_learned_policy_adapter_interface.md) | Learned local-policy eligibility, adapter contracts, training blockers, and durable checkpoint metadata. |
| CARLA and external simulators | [issue_1169_carla_live_replay.md](issue_1169_carla_live_replay.md), [carla-replay-parity skill](../../.agents/skills/carla-replay-parity/SKILL.md), [issue_1491](https://github.com/ll7/robot_sf_ll7/issues/1491) | CARLA live replay boundaries, host-dependent evidence, and parity caveats. |
| SLURM and long jobs | [slurm_issue_batch_status_2026-05-21.md](slurm_issue_batch_status_2026-05-21.md), [../dev/slurm_submission.md](../dev/slurm_submission.md), [../dev/slurm_resource_audit.md](../dev/slurm_resource_audit.md), [open_issues_training_split_audit_2026-05-30.md](open_issues_training_split_audit_2026-05-30.md) | Launch packets, campaign state, resource limits, and artifact preservation for long runs. |
| Root layout and cleanup | [issue_1690_root_layout_inventory.md](issue_1690_root_layout_inventory.md), [issue_1573_root_layout_inventory.md](issue_1573_root_layout_inventory.md), [issue_1583_high_risk_root_boundaries.md](issue_1583_high_risk_root_boundaries.md), [issue_1598_1599_root_compatibility_decisions.md](issue_1598_1599_root_compatibility_decisions.md) | Root hygiene, high-risk path boundaries, compatibility shims, and inventory-first cleanup. |
| Adversarial search | [issue_1457_adversarial_generation_protocol.md](issue_1457_adversarial_generation_protocol.md), [issue_1500_adversarial_manifest.md](issue_1500_adversarial_manifest.md), [issue_1571_adversarial_smoke_packet_sharpening.md](issue_1571_adversarial_smoke_packet_sharpening.md), [../ai/awesome_copilot_adaptation.md](../ai/awesome_copilot_adaptation.md) | Bounded adversarial generation, manifest freeze, smoke packets, and workflow adaptation. |
| Manual control and trace analysis | [issue_1151_manual_control_mvp_foundation.md](issue_1151_manual_control_mvp_foundation.md), [issue_1689_simulation_trace_export_schema.md](issue_1689_simulation_trace_export_schema.md), [../debug_visualization.md](../debug_visualization.md) | Recorder workflows, trace export shape, and debug visualization boundaries. |

## Context-Pack Manifests

Generated packs are temporary artifacts and should stay under ignored paths such as
`output/context_packs/`.

Use [../ai/context_packing.md](../ai/context_packing.md) for the Repomix decision and command
patterns. Start with these curated pack scopes:

| Pack | Include globs | Exclude notes |
|---|---|---|
| `learned_policy_integration` | `AGENTS.md`, `docs/context/INDEX.md`, `docs/context/policy_search/**`, `docs/context/issue_1618_learned_policy_adapter_interface.md`, `robot_sf/nav/**`, `configs/training/**`, `scripts/training/**` | Exclude checkpoints and raw run output. |
| `benchmark_campaign_evidence` | `AGENTS.md`, `docs/code_review.md`, `docs/context/INDEX.md`, `docs/context/issue_691_benchmark_fallback_policy.md`, `docs/context/artifact_evidence_vocabulary.md`, `docs/benchmark*.md`, `robot_sf/benchmark/**`, `configs/benchmarks/**`, `scripts/validation/**` | Exclude raw episode JSONL, videos, and coverage HTML. |
| `slurm_artifact_rescue` | `AGENTS.md`, `local.machine.md` when local-only use is safe, `docs/dev/slurm*.md`, `docs/context/slurm_issue_batch_status_2026-05-21.md`, `docs/context/open_issues_training_split_audit_2026-05-30.md`, `model/registry.md` | Do not publish machine-local secrets or raw Slurm logs. |
| `root_layout_cleanup` | `AGENTS.md`, `docs/context/issue_1690_root_layout_inventory.md`, `docs/context/issue_1583_high_risk_root_boundaries.md`, `docs/context/issue_1598_1599_root_compatibility_decisions.md`, `.github/**`, `scripts/dev/**` | Keep generated inventories under `output/` unless promoted as compact evidence. |
| `adversarial_search` | `AGENTS.md`, `.agents/skills/adversarial-search-campaign/**`, `docs/context/issue_1457_adversarial_generation_protocol.md`, `docs/context/issue_1500_adversarial_manifest.md`, `docs/context/issue_1571_adversarial_smoke_packet_sharpening.md`, `configs/**`, `scripts/**` | Exclude large campaign bundles unless replaced by tracked summaries. |

## Optional Tool Boundary

- `context-mode`: optional runtime-layer pilot for long sessions and large tool output. Keep caches
  local and do not make it a repository dependency without a completed evaluation issue.
- `Understand-Anything`: optional read-only graph/navigation evaluation. Pair any graph output with
  this index, [AGENTS.md](../../AGENTS.md), and benchmark fallback policy before drawing workflow
  conclusions.
- Repomix: recommended for static, reproducible context packs. Generated packs are disposable
  `output/` artifacts, not source-of-truth documentation.
