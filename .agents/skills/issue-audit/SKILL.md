---
name: issue-audit
description: Interactive maintainer issue audit that presents one issue_decision_envelope.v1 at a time, applies only an exact answer, and verifies the result.
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
docs/context/issue_audit_contract.md and use the shared classifier and envelope
builder in scripts/dev/issue_audit_core.py; do not recreate its label or
evidence rules in this prompt.

## Workflow

1. Load the latest issue_audit_plan.v1 and its pending_decisions queue. If no
   queue exists, generate a fresh read-only plan in interactive mode with
   bounded comment reads. Use the shared core to emit the next
   issue_decision_envelope.v1, ordered by issue number or an explicit issue
   scope. Never use Project #5 ordering as an implicit policy answer.
2. Refresh the selected issue and complete comments through the REST helper
   scripts/dev/gh_issue_rest.py before presenting the envelope. Compare the
   live issue state and labels with the envelope and fail closed if they
   changed.
3. Present exactly one envelope and one focused question in the current turn.
   Quote only bounded issue body/comment evidence and use documented option
   tokens. Never bundle scope, rights, compute, provenance, publication, and
   priority into one question.
4. If the envelope status is needs_clarification, ask one clarification
   question and do not apply a decision answer. If it is
   blocked_incomplete_inventory, preserve the issue and report the missing
   evidence without asking a question that depends on it.
5. On the next turn, treat only the exact answer format
   `#<issue-number>: <option-token>` as authorization for that stated decision.
   Apply the smallest required issue-body, label, or comment change. Record the
   answer source and preserve uncertainty outside the answered choice.
6. Read back the issue through REST and verify the body marker or comment,
   labels, state, and any explicitly requested Project field. Rerun the shared
   classifier after the answer. Report the next queue item, but do not ask a
   second question in the same turn.

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
- Never fabricate an option list when the issue body and comments do not
  document the available choices.

## Output

Return:

- the single question asked, or the exact answer applied;
- the `issue_decision_envelope.v1` status, digest, and queue position;
- the evidence source and issue number;
- body, label, comment, state, and explicitly requested Project changes;
- REST readback and the rerun classifier result; and
- the next pending decision, if one remains.

The machine-readable handoff uses issue_audit_plan.v1,
issue_decision_envelope.v1, and pending_decisions. Never claim that an answer
was applied without readback.

## When to use

Use this skill for maintainer decisions, ambiguity resolution, exact answer
application, and review of a pending queue emitted by
issue-audit-autonomous.
