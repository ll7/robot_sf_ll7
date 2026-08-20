# Autonomous issue implementation admission

Autonomous implementation must use the repository label state and a complete issue contract. A Project field, title, priority, open state, or absence of a blocker is not sufficient.

## Admission rule

An implementation worker may acquire `agent-claims/issue-<number>` only when the issue:

- is open and labeled `state:ready`;
- is unassigned and has no existing atomic claim;
- is not a parent, epic, decision, review, active-work, compute, campaign, external-input, or blocked issue;
- contains non-empty sections for the objective, scope, inputs or affected surfaces, acceptance criteria, and verification.

The checker reports each missing field separately. It never converts a numeric completeness score into a pass.

## Commands

Check an issue without a write:

```bash
uv run python scripts/dev/issue_implementability.py <issue-number>
uv run python scripts/dev/goal_issue_admission.py <issue-number> --check-only
```

Acquire the existing atomic claim only after the live preflight passes:

```bash
uv run python scripts/dev/goal_issue_admission.py <issue-number>
```

Offline contract fixtures can use a body file and explicit labels:

```bash
uv run python scripts/dev/issue_implementability.py 1 \
  --body-file /tmp/issue.md \
  --label state:ready \
  --title "fixture issue"
```

## Outcomes

- `ready`: the generic implementation contract permits claim admission.
- `needs_ready_label`: a maintainer or authorized preparation phase has not marked the issue ready.
- `needs_spec`: one or more required contract fields are missing.
- `parent`: dispatch a bounded child, not the tracker.
- `human_decision`: obtain the named ruling before implementation.
- `needs_compute`: route through the compute owner, not the local implementation lane.
- `blocked`, `working`, `review`, `assigned`, or `already_claimed`: do not start another implementation worker.
- `error`: re-read the live state; do not infer readiness.

## Authority boundary

This is a generic implementation gate. Research campaigns still require `research_answerability.v1`. Result and claim interpretation still require their specialized evidence and review contracts. A successful claim does not imply validation, review, merge readiness, evidence admission, compute authorization, release authorization, or issue completion.
