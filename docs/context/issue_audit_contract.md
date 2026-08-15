# Issue-audit contract

This document is the shared policy boundary for the two issue-audit entry
points:

- issue-audit-autonomous performs evidence-supported cleanup and emits a
  pending-decision queue.
- issue-audit is the maintainer-facing loop that presents one decision envelope
  and applies one explicit answer at a time.

Both entry points consume the same issue_audit_plan.v1 produced by
scripts/dev/issue_audit_core.py. The interactive path projects the next queue
row into issue_decision_envelope.v1. The classifier, not either skill prompt,
owns the decision about whether a label repair is safe.

## Authority boundary

The autonomous path may read all open issues and related repository state. It
may apply only mutations that are directly supported by current repository
evidence:

- add or remove existing labels when the label family rule and evidence select
  one unambiguous result;
- expose a proven provenance, rights, compute, or external-input blocker;
- close an issue only when the closure conditions below are proven; and
- write a machine-readable pending-decision queue.

It must not ask questions, select among maintainer policy options, change
research priority, create labels or follow-up issues, write Project #5 fields,
submit compute, publish or release anything, or reinterpret benchmark
evidence. Project #5 ordering remains owned by gh-issue-sequencer.

The interactive path may consume the autonomous queue and ask at most one
focused question per turn. The answer is authoritative only for the stated
question. It must apply the exact answer to the issue body, labels, and/or
comment, then read back the issue state before asking another question. It
must rerun the shared classifier after each answer. It may not silently make a
different priority or benchmark-policy decision. Project #5 changes are
opt-in, must be explicitly requested, and must be read back; queue ordering
itself remains owned by gh-issue-sequencer.

## Evidence sources

The core inventories, with bounded pagination:

1. canonical open issues (pull-request rows from the issues endpoint are
   excluded);
2. open and merged pull requests, correlated by explicit issue references in
   title, body, or branch;
3. the existing repository label names;
4. remote agent-claim branches;
5. local worktrees and their branches; and
6. visible SLURM jobs when squeue is available.

An inventory page cap, failed read, unavailable SLURM query for a
resource:slurm issue, or failed readback is an uncertainty. The plan records it
and the apply path fails closed. A partial inventory is never evidence that an
issue is ready or complete.

When the bounded global closed-PR history is partial, the core may also read
each currently open issue's bounded REST timeline. A `cross-referenced` event
whose source contains a merged pull request contributes targeted closure
evidence with `coverage_source: targeted_issue_timeline`. The plan exposes this
under `inventory_coverage` and keeps the original global `closed_prs` truncation
metadata visible. Timeline coverage narrows the evidence for the currently open
issues; it does not make the repository-wide closed-PR history complete, and
mutation application remains fail-closed while any global or targeted source is
partial or unavailable.

Issue bodies and comments are evidence sources for decisions and gates. They
are not permission to infer missing provenance, rights, compute authorization,
or maintainer intent.

## Label rules

Canonical execution-state labels are mutually exclusive. The repository also
uses composable `state:*` qualifiers, including `state:review`,
`state:needs-artifact-promotion`, and `state:needs-interpretation`; these are
preserved and are not removed as contradictory execution states. The core may
repair canonical execution-state contradictions only when current evidence
selects a winner:

1. state:blocked-external-input for a proven missing external input;
2. state:blocked for a proven provenance, rights, compute, or generic blocker;
3. state:hold when an explicit hold is already the selected state;
4. state:running when an open linked PR, claim, worktree, or active job is
   observed; and
5. state:ready when explicit acceptance or validation evidence exists and no
   gate or active record remains.

`state:review` is a composable qualifier, not a dispatch state. When the
complete issue body/comment inventory contains an explicit report-status line
such as `Report status: diagnostic_ready_for_domain_review`, and no claim,
worktree, or active job remains, the classifier treats terminal review as
stronger than an open PR-only record. It removes stale `state:ready` or
`state:running`, adds the existing `state:review` label when available, and
surfaces `decision-required`. It does not infer terminality from a future
acceptance criterion, a hypothetical stop rule, or an unstructured mention of
the word "terminal". A live claim, worktree, job, blocker, or unavailable
SLURM inventory keeps the classifier fail-closed.

Before the autonomous core adds either dispatch-suppressing blocker label
(`state:blocked` or `state:blocked-external-input`), the issue body or complete
comment inventory must contain an explicit `blocked-triage-v1` reason block or
a `Blocked-by: #<issue-or-pr-number>` reference. The reason evidence is bound
into the planned mutation and the apply path rejects a blocked-label mutation
that omits it. Prose that merely looks like a blocker is not enough: when the
blocker is otherwise detectable but its reason is not recorded, the core
declines the blocked-label write and adds the existing `needs-triage` label
when available. The plan's `blocked_label_report` records the blocker evidence,
reason evidence, and whether the write was applied, declined, or already
present. An existing unexplained blocker label is reported for repair review;
absence of a reason is not treated as proof that the underlying blocker has
resolved.

A stale state:running label is preserved when no active record is observable;
absence of evidence is not evidence of completion. Multiple states without a
decisive signal become a decision gate.

Resource labels and evidence labels are composable. A resource label does not
by itself prove that work is blocked. A type label is normally singular. A
missing or conflicting type is preserved for review unless a complete,
valid Archetype Metadata block maps unambiguously to an existing type label.

The core never creates labels. It reads the repository label inventory before
planning an addition or removal. URI-sensitive label endpoints must encode the
whole label name: state:running is sent as state%3Arunning.

## Readiness and blocker rules

Readiness requires concrete issue-local evidence such as an acceptance,
definition-of-done, success-criteria, validation heading, complete checklist,
or explicit validation command. A title, a type label, Project #5 placement,
or a plausible implementation idea is not enough.

Terminal campaign results awaiting interpretation or domain review are not
dispatch-ready. The audit recognizes only explicit report-status evidence, so a
completed run can be routed to review without treating its metrics as a new
benchmark or paper-facing claim. Open implementation/report PRs may remain
active repository work, but they do not by themselves restore `state:ready`
after an explicit terminal-review status.

The following are fail-closed gates:

- provenance: missing or incompatible digest, seed, lineage, compatibility, or
  exact-source evidence;
- rights: missing license, permission, consent, redistribution, or release
  evidence;
- compute: missing authorization, allocation, quota, or required SLURM
  availability;
- external input: missing dataset, checkpoint, model weight, or other
  required external asset; and
- maintainer decision: explicit decision-required label or issue text asking a
  maintainer to choose, approve, confirm, or set policy.

When a gate is proven, the autonomous path makes the blocker visible only with
existing labels. It does not answer the gate. Optional research remains open
unless it is duplicate, invalid, superseded, or complete.

## Closure conditions

Autonomous closure requires an open issue, at least one merged PR explicitly
linked to the issue, and one of:

- an explicit body line such as
  "Completion condition: merged PR #..." or
  "Close condition: merged pull request"; or
- all issue acceptance checkboxes are complete and the merged PR is the
  issue-linked implementation evidence.

Parent, roadmap, epic, tracking, multi-slice, and umbrella issues require the
literal documented condition
"Parent close condition: all linked children closed", plus evidence that no
referenced child remains open. A merged child PR alone never closes a parent.

If merged work exists without a documented completion condition, the issue
remains open and the plan records a closure review finding.

## Shared plan schema

Every plan has schema issue_audit_plan.v1 and contains:

    {
      "schema": "issue_audit_plan.v1",
      "repo": "ll7/robot_sf_ll7",
      "mode": "autonomous",
      "project5": {"writes": false, "owner": "gh-issue-sequencer"},
      "inventory": {},
      "issues": [],
      "mutations": [],
      "inventory_coverage": {},
      "pending_decisions": [],
      "truncation_or_errors": [],
      "counts": {}
    }

Each pending decision is shaped so a background run can hand it to the
interactive skill without guessing:

    issue: "#123"
    number: 123
    title: "..."
    url: "https://github.com/ll7/robot_sf_ll7/issues/123"
    state: open
    labels: ["decision-required"]
    classification: decision-required
    decision_required: true
    question_source: "issue body/comments"
    blocking_evidence: "..."
    evidence_sources: []
    documented_options: []
    safe_mutations_applied: []

The plan is deterministic for a fixed inventory. It contains evidence and
reasons for every safe mutation. apply_mutations refuses incomplete plans,
enforces a mutation budget, uses REST for issue/label writes, and reads every
touched issue back. Project changes are intentionally absent.

## Decision envelope

The interactive path presents exactly one `issue_decision_envelope.v1` at a
time. The envelope is a factual projection of the pending queue, not a new
policy layer. It contains the plan digest, deterministic issue-number queue
position, live-snapshot state and labels, bounded body/comment source excerpts,
documented option tokens, confirmed autonomous mutations, an exact answer
format, and the verification contract.

The envelope is `ready` only when the inventory is complete and at least two
explicit options are documented by the issue body or comments. If the source
does not document a choice set, the envelope is `needs_clarification`; the
interactive skill may ask one focused clarification question but must not
invent a policy option or apply an answer. A truncated inventory, a relevant
unavailable SLURM inventory, a stale plan digest, or changed live issue state
is fail-closed.

The answer format is `#<issue-number>: <option-token>`. The token must be one
of the source-backed options in the envelope. Before applying it, the
interactive path refreshes the issue, compares state and labels with the
envelope, applies only the stated answer, reads the result back, and reruns
the shared classifier. Project #5 writes remain false unless separately and
explicitly requested under the existing Project contract.

## Routing

issue-contract-maintainer is the orchestrator for issue-contract work:

- deterministic template, label, and state cleanup routes to
  issue-audit-autonomous;
- ambiguity, exact maintainer decisions, and answer application route to
  issue-audit; and
- Project #5 ordering routes to gh-issue-sequencer.

The two audit skills may share this contract and the core, but they must not
share an implicit authority mode. The entry-point name is the interface.
