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
         --max-closed-pr-pages 50 \
         --max-wall-seconds 420 \
         --output output/issue_audit_plan.json

   Do not use Project #5 as a mutex or as a source of readiness evidence.
2. Inspect schema, counts, classification_status, and truncation_or_errors.
   The command has one fail-closed aggregate wall-time budget across REST and
   local discovery, merged-PR indexing, and issue classification, in addition
   to each individual gh command timeout. Closed-PR history has an independent
   50-page budget so repository growth does not widen every other REST source.
   If either budget expires, preserve affected issues and stop mutation
   application; the emitted partial inventory is not a complete audit. A
   classification timeout retains `resume_from_issue` only as a diagnostic
   next-unclassified issue. It is not a resumable cursor:
   `resume_supported: false` and
   `resume_requires_fresh_full_inventory: true` require a fresh full inventory
   before any retry, and suffix-only continuation must be rejected. Timeout
   plans have empty top-level and per-issue mutation lists. POSIX main-thread
   discovery and classification work is interrupted at the deadline; other
   hosts invalidate late results cooperatively. Increase a bounded source
   budget only when current repository counts justify it.
3. Apply only the mutations already present in the plan:

       uv run python scripts/dev/issue_audit_core.py apply \
         output/issue_audit_plan.json

   The executor verifies plan provenance (`source_sha`, `classifier_digest`, producer
   identity), checks freshness against fetched `origin/main`, refuses incomplete or diagnostic
   plans, enforces a bounded mutation budget, uses URI-safe REST label paths, and reads every
   touched issue back. For read-only diagnostic runs without write intent, use
   `--read-only-diagnostic` during the `plan` step.
4. Build the pending-decision queue from pending_decisions. For each entry,
   include the issue title/URL, current classification and labels, bounded
   issue body/comment source metadata, blocking evidence, documented options,
   and the safe mutations actually confirmed by readback. The core helper
   build_pending_decision_queue(plan, applied_mutations=...) performs this
   merge. The interactive skill projects the first row into
   issue_decision_envelope.v1; do not answer or reorder the queue here. The
   minimum handoff shape remains:

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
- Add an existing blocker label only when provenance, rights, compute, or
  required external-input evidence proves the gate and the issue body or
  complete comment inventory records a `blocked-triage-v1` reason block or a
  `Blocked-by: #<number>` reference. Otherwise, decline the dispatch-suppressing
  label and route to the existing `needs-triage` label when available.
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
