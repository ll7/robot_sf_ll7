---
name: issue-audit
description: Interactive maintainer issue audit that consumes one decision queue entry, asks at most one focused question, applies the exact answer, and verifies the result.
category: github-issue
kind: atomic
phase: context
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# Interactive Issue Audit

## Purpose

Use this entry point when an issue needs a maintainer decision or when a
maintainer has supplied an exact answer that must be applied to GitHub. The
autonomous cleanup path is issue-audit-autonomous. Read the shared contract in
docs/context/issue_audit_contract.md and use the shared classifier in
scripts/dev/issue_audit_core.py; do not recreate its label or evidence rules
in this prompt.

## Workflow

1. Load the latest issue_audit_plan.v1 and its pending_decisions queue. If no
   queue exists, generate a fresh read-only plan in interactive mode with
   bounded comment reads. Refresh
   the selected issue and complete comments through the REST helper
   scripts/dev/gh_issue_rest.py before asking anything.
2. Select the first unresolved decision in the requested scope. Do not use
   Project #5 score or ordering as an implicit policy answer. Project #5
   ordering remains owned by gh-issue-sequencer.
3. Ask exactly one focused question in the current turn. Quote the relevant
   issue body/comment evidence and make the answer choices concrete. Never
   bundle scope, rights, compute, provenance, publication, and priority into
   one question.
4. On the next turn, treat only the user's exact answer as authorization for
   that stated decision. Apply the smallest required issue-body, label, or
   comment change. Record the answer source and preserve uncertainty outside
   the answered choice.
5. Project #5 writes are opt-in and separate. Perform them only when the user
   explicitly requested the field change, and read the field back. Do not
   infer a priority score or reorder research from a conversational hint.
6. Read back the issue through REST and verify the body marker or comment,
   labels, state, and any explicitly requested Project field. Rerun the shared
   classifier after the answer. If the marker, label, or readback is missing,
   report failure and do not ask a second question in the same turn.

## Guardrails

- Ask at most one question per turn and never ask for approval of ordinary
  deterministic cleanup.
- Do not guess a maintainer policy, research priority, benchmark
  interpretation, provenance status, rights grant, or compute authorization.
- Do not close an issue merely because a PR is merged; use the shared closure
  evidence contract.
- Do not create labels or follow-up issues in this loop unless the user
  explicitly changes the scope and the relevant skill is invoked.
- Keep optional research open unless it is duplicate, invalid, superseded, or
  complete.
- Use REST for ordinary issue reads and writes. Never pass multiline
  Markdown-heavy content inline through a shell string; use
  scripts/dev/gh_comment.sh or a body file.
- If the queue or inventory is partial, preserve the issue and report the
  missing evidence instead of asking a question that depends on it.

## Output

Return:

- the single question asked, or the exact answer applied;
- the evidence source and issue number;
- body, label, comment, state, and explicitly requested Project changes;
- REST readback and the rerun classifier result; and
- the next pending decision, if one remains.

The machine-readable handoff uses issue_audit_plan.v1 plus
pending_decisions. Never claim that an answer was applied without readback.

## When to use

Use this skill for maintainer decisions, ambiguity resolution, exact answer
application, and review of a pending queue emitted by
issue-audit-autonomous.
