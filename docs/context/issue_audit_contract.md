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
issue is ready or complete. Ordinary REST collections use `--max-pages`; the
larger repository-wide closed-PR history uses the independent
`--max-closed-pr-pages` budget (default 50) so increasing history coverage does
not silently widen issue, open-PR, label, or timeline reads.

When the bounded global closed-PR history is partial, the core may also read
each currently open issue's bounded REST timeline. A `cross-referenced` event
whose source contains a merged pull request contributes targeted closure
evidence with `coverage_source: targeted_issue_timeline`. The plan exposes this
under `inventory_coverage` and keeps the original global `closed_prs` truncation
metadata visible. Timeline coverage narrows the evidence for the currently open
issues; it does not make the repository-wide closed-PR history complete, and
mutation application remains fail-closed while any global or targeted source is
partial or unavailable.

The core performs the REST core-quota preflight before reading issues, comments,
pull requests, timelines, or labels. It converts the remaining capacity above
the safety margin into one shared request budget, and every subsequent REST
page reserves one request from that budget. Optional comment enrichment stops
after an actual rate-limit response. Only explicit primary or secondary
rate-limit responses (including HTTP 429 and GitHub abuse/rate-limit markers)
are classified as quota exhaustion; an ordinary permission HTTP 403 remains an
incomplete source error. A low or spent request budget, partial/truncated
collection, failed source read, or uncertain quota result sets
`classification_status.mutations_suppressed` to `true` and clears both
top-level and per-issue mutations. A mid-run rate limit also records the core
reset time, retry-after timestamp, retry command, and a human-readable handoff
so the next run can resume only after a fresh inventory.

Issue bodies and comments are evidence sources for decisions and gates. They
are not permission to infer missing provenance, rights, compute authorization,
or maintainer intent.

The command applies one aggregate monotonic wall-time budget
(`--max-wall-seconds`, default 120 seconds) across REST discovery, local
discovery, merged-PR reference indexing, and issue classification, in addition
to the 60-second timeout on each individual `gh` subprocess. Each runner
receives the remaining budget, and a result that returns after the deadline is
converted to a timeout. The merged-PR index, classification loop, plan
finalization, and JSON rendering perform final deadline checks before a result
can be reported as complete; on POSIX main-thread execution, in-process
discovery, indexing, classification, and rendering are also interrupted at the
deadline. Embedding contexts that cannot install an interval timer retain
cooperative checks and invalidate late results. When the aggregate budget
expires, the core records a structured inventory or classification error, writes a partial plan
when an output path was requested, returns a non-zero status, and the apply
path refuses all mutations. Timeout conversion clears both the top-level
`mutations` list and every per-issue `issues[*].mutations` list.

A classification timeout retains `classification_status.resume_from_issue` for
diagnostics as the next unclassified issue; it is not a resumable cursor.
`resume_supported` is `false` and `resume_requires_fresh_full_inventory` is
`true`, so consumers must reject suffix-only continuation and rerun discovery
and classification against a fresh full inventory. A partial plan is never a
complete audit; callers may increase the budget only when the bounded
inventory scope justifies it.

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

`state:working` is a live downstream-work qualifier. It is preserved as a
qualifier, but the classifier must not promote an issue carrying it to
`state:ready` until an exact-head completion receipt has been independently
verified. The receipt is a delivery-integrity prerequisite, not scientific,
benchmark, release, licensing, or domain approval.

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

Canonical ruling comments use the exact unquoted form
`ll7/robot_sf_ll7#<issue>: <token>`. The classifier orders timestamped comments
chronologically before deciding whether a ruling follows an older prompt. If
any comment timestamp is missing or invalid, the ruling cannot suppress the
decision gate. Exact lines under an example, copied, quoted, historical, or
"do not apply" context are not treated as live rulings.

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

The documented closure condition is necessary but no longer sufficient for an
autonomous `close_issue` mutation. The issue-audit inventory may provide an
issue-number keyed `completion_receipts` map. Each entry must contain an
`issue_completion_receipt.v1` payload and the JSON result from
`scripts/dev/issue_completion_receipt.py verify`. The receipt binds the issue
contract digest, exact base and delivered head, branch, changed paths and
diffstat, validation commands and exit codes, validation inputs, durable
artifacts, one disposition per acceptance criterion, residual risks, producer,
independent verifier, and the post-review drift policy.

The standalone verifier checks that the named base and head exist, the branch
still points at the reviewed head, the exact base/head diff matches the receipt,
and any covering PR snapshot has the same head, base, and branch. It also
checks contract, artifact, and validation-input digests. Validation records
keep `passed`, `failed`, `skipped`, `unavailable`, and `not_applicable`
distinct; only a receipt with all validations passed, every criterion `met` or
`not_applicable`, an independent verifier status of `verified`, and a matching
Git-backed verification result can authorize autonomous closure or clear the
receipt prerequisite for downstream promotion. A missing, stale, producer-only,
failed, skipped, or unavailable receipt remains a fail-closed finding.

Build and verify a receipt with:

    uv run python scripts/dev/issue_completion_receipt.py build \
      --input receipt-payload.json --output completion-receipt.json
    uv run python scripts/dev/issue_completion_receipt.py verify \
      --receipt completion-receipt.json --repo-root .

The receipt does not replace PR review, maintainer decisions, domain or
scientific interpretation, benchmark admission, release checks, licensing
review, or specialized evidence packets. Raw logs remain out of the receipt;
they may be referenced through digested durable artifacts.

## Shared plan schema

Implementation admission may consume one canonical `issue_dependency_packet.v1` for exact
prerequisites. The packet is a predicate record attached to the issue contract, not a second issue
graph; mandatory unsatisfied, unavailable, conflicting, or invalid rows keep admission blocked.
See [Typed Issue Dependency Packets](../ai/issue_dependency_packets.md) for the row contract and
read-only resolver.

Every plan has schema issue_audit_plan.v1 and contains:

    {
      "schema": "issue_audit_plan.v1",
      "repo": "ll7/robot_sf_ll7",
      "mode": "autonomous",
      "project5": {"writes": false, "owner": "gh-issue-sequencer"},
      "inventory": {},
      "classification_status": {
        "status": "complete",
        "resume_from_issue": null,
        "resume_supported": false,
        "resume_requires_fresh_full_inventory": true,
        "remaining_issue_numbers": [],
        "mutations_suppressed": false
      },
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

Every planned mutation also carries the issue snapshot used by classification:

    expected_issue:
      state: open
      updated_at: "2026-08-23T00:00:00Z"

The plan digest binds this state/version snapshot. Apply rejects missing or inconsistent snapshots,
issue identifiers that are not exact positive JSON integers, and non-null `close_issue` values
before any REST read or write. Immediately before the first mutation for an issue, apply reads the
live issue once and compares both fields. A mismatch skips the complete per-issue mutation batch,
records a machine-readable `stale_state` disposition with expected and observed values, and does
not count the issue as successfully applied. After admitted writes, readback verifies the expected
issue state as well as requested label additions/removals; `close_issue` expects `closed`, while a
label-only batch must retain its planned state. Apply results report `stale_state_issues` separately
from `skipped_stale_mutations`, so one stale issue containing several planned writes is not
misrepresented as one skipped mutation; every early refusal returns the same count shape.

The plan is deterministic for a fixed inventory. It contains evidence and
reasons for every safe mutation. apply_mutations refuses incomplete plans,
enforces a mutation budget, uses REST for issue/label writes, and reads every
touched issue back. A repeated `remove_label` for a label that is already
absent is recorded in the additive `already_applied` result bucket with
`skipped_reason: already_absent`; it is not a failure, but the issue still
goes through readback. The apply result also exposes `counts` for planned,
applied, already-applied, and failed operations. Other 404s remain failures.
Project changes are intentionally absent.

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
