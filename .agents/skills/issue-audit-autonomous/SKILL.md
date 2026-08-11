---
name: issue-audit-autonomous
description: Autonomous open-issue audit that applies only evidence-supported label or completion repairs and emits a machine-readable pending-decision queue without asking questions.
category: github-issue
kind: atomic
phase: context
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# Autonomous Issue Audit

## Purpose

Use this entry point for unattended open-issue inventory and safe contract
cleanup. It may repair unambiguous labels, expose proven blockers, and close
issues only under the shared closure contract. It never enters a question loop
or chooses maintainer policy.

Read docs/context/issue_audit_contract.md before operating. The deterministic
classifier and mutation executor are in
scripts/dev/issue_audit_core.py. Both this skill and issue-audit consume
issue_audit_plan.v1.

## Workflow

1. Inventory all open issues, open and merged pull requests, existing labels,
   remote claims, worktrees, and visible jobs with bounded REST/local reads:

       uv run python scripts/dev/issue_audit_core.py plan \
         --mode autonomous --include-comments \
         --output output/issue_audit_plan.json

   Do not use Project #5 as a mutex or as a source of readiness evidence.
2. Inspect schema, counts, and truncation_or_errors. If any source is partial
   or failed, preserve affected issues and stop mutation application. A
   partial inventory is not a complete audit.
3. Apply only the mutations already present in the plan:

       uv run python scripts/dev/issue_audit_core.py apply \
         output/issue_audit_plan.json

   The executor refuses incomplete plans, enforces a bounded mutation budget,
   uses URI-safe REST label paths, and reads every touched issue back.
4. Build the pending-decision queue from pending_decisions. For each entry,
   include the exact issue body/comment source, blocking evidence, and the
   safe mutations actually confirmed by readback. The core helper
   build_pending_decision_queue(plan, applied_mutations=...) performs this
   merge. Use this shape:

       issue: "#123"
       decision_required: true
       question_source: "issue body/comments"
       blocking_evidence: "..."
       safe_mutations_applied: []

5. Report the applied mutation/readback result and queue. End the run after
   safe work is complete; do not ask a question or call the interactive skill
   inline. The queue is the handoff to issue-audit.

## Allowed autonomous repairs

- Normalize contradictory state labels only when current PR, claim,
  worktree, job, acceptance, or explicit blocker evidence selects one state.
- Add or remove only labels that already exist in the repository.
- Mirror a complete, valid issue Archetype Metadata block to one existing type
  label when no competing type label exists.
- Add an existing blocker label when provenance, rights, compute, or required
  external-input evidence proves the gate.
- Add state:ready only with concrete acceptance or validation evidence and no
  active or unresolved gate.
- Close only with a merged issue-linked PR plus the documented completion
  condition in docs/context/issue_audit_contract.md.

## Guardrails

- Never ask questions, select among options, or infer research priority.
- Never write Project #5 fields; gh-issue-sequencer owns queue ordering.
- Never create labels, follow-up issues, branches, claims, jobs, releases, or
  publications.
- Never submit compute or reinterpret benchmark, provenance, rights, or
  paper-facing evidence.
- Preserve state:running when no active record is observable; uncertain state
  is not readiness or completion.
- Preserve optional research unless it is duplicate, invalid, superseded, or
  complete.
- Fail closed on partial inventory, missing label inventory, unavailable
  required compute evidence, unsupported mutation, or failed readback.
- Use REST for ordinary GitHub issue operations and keep Markdown-heavy bodies
  out of inline shell arguments.

## Output

Return:

- the exact plan path and schema;
- inventory counts and truncation/error sources;
- mutations planned, applied, failed, and read back;
- issues closed, with merged-PR and completion evidence;
- the pending-decision queue; and
- explicit non-actions: no questions, no Project #5 writes, no new labels,
  no follow-up issues, no compute submission, and no evidence reinterpretation.

## When to use

Use this skill for unattended repository maintenance, scheduled issue audits,
and the deterministic cleanup phase of issue-contract-maintainer.
