# Robot SF issue-label taxonomy

This is the compact label vocabulary used by the shared `decision-cockpit` workflow
when it classifies and records decisions in `ll7/robot_sf_ll7`. It describes the
meaning of labels; it is not a second dispatch implementation. GitHub's label list
is the live availability authority, and labels must not be created by hand.

## Authorities

- [Issue state and dispatch rules](../../CONTRIBUTING.md#issue-state-labels-and-dispatch)
  define the sole positive dispatch signal: `state:ready` with no contradictory
  execution state.
- [`issue_audit_core.py`](../../scripts/dev/issue_audit_core.py) defines the
  fail-closed classification, execution-state precedence, and composability rules
  used by issue-audit tooling.
- The shared `decision-cockpit` skill consumes this file as its repository-local
  taxonomy reference. Its path is `docs/ai/label-taxonomy.md` from a Robot SF
  checkout.

To inspect live labels before a mutation, use the REST endpoint and read the result
back:

```bash
gh api 'repos/ll7/robot_sf_ll7/labels?per_page=100' --paginate --jq '.[].name' | sort
```

## Decision flow

| Label | Meaning | Next action |
| --- | --- | --- |
| `decision-required` | The issue contains a maintainer or author judgment that has not been classified. | Prepare one decision envelope; do not treat the label as approval. |
| `decision-ready` | The alternatives and residual judgment have been prepared for the author. | Present the packet and wait for the exact ruling token. |
| `ruled` | A ruling was recorded and its downstream terminal state still needs verification. | Verify merged, parked, escalated, or external-blocked closure. |

Decision labels describe the decision workflow only. They do not authorize a
submission, merge, publication, release, or issue closure by themselves.

## Lifecycle and origin

These labels add lifecycle or provenance context and are composable with state,
resource, type, and evidence labels:

| Family / labels | Meaning |
| --- | --- |
| `blocked`, `blocked-external` | Work is blocked; `blocked-external` identifies an unavailable external input or action. Prefer the more specific `state:blocked` or `state:blocked-external-input` when the execution-state classifier is the relevant authority. |
| `parked`, `parked-revivable` | Work is intentionally not active; `parked-revivable` must retain a revival condition. |
| `deferred`, `wontfix` | Deferred or deliberately not pursued; neither is a dispatch signal. |
| `follow-up`, `friction`, `campaign` | Successor/residual work, process or tooling friction, and campaign/evidence context. |

## Execution state

The core execution states are mutually exclusive. The classifier gives the more
blocking state precedence when contradictory labels coexist:

| Label | Meaning |
| --- | --- |
| `state:ready` | Sole positive dispatch signal, and only when no contradictory execution state is present. |
| `state:running` | Work is already in progress; do not dispatch a second worker. |
| `state:blocked` | Internal prerequisite or decision blocks execution. |
| `state:blocked-external-input` | External data, access, license, or other outside input blocks execution. |
| `state:hold` | Explicit classifier-compatible hold. It is not in the current live label inventory; do not create it manually. |
| `state:done` | The tracked work is recorded as done; verify the issue/PR terminal state separately. |

These are composable state qualifiers rather than replacement execution states:
`state:review`, `state:needs-artifact-promotion`, and `state:needs-interpretation`.
An issue with no `state:*` label is undispatchable, not implicitly ready. A
`resource:*` label never promotes an issue to ready.

## Resource

Resource labels identify the lane or dependency needed by the work. They never
authorize that resource or dispatch work:

`resource:local`, `resource:slurm`, `resource:carla`, and
`resource:external-data`.

Compute submission still requires the private-ops queue and its submit-path
preflight; a resource label alone is not a submission contract.

## Type

Type labels describe the dominant work shape:

`type:analysis`, `type:benchmark`, `type:data`, `type:docs`,
`type:implementation`, `type:synthesis`, `type:training`, and `type:workflow`.

Type labels aid routing and reporting. They do not override state, evidence, or
domain-approval gates.

## Evidence

Evidence labels describe the evidence boundary or intended use, not a scientific
result:

`evidence:analysis-only`, `evidence:blocked`, `evidence:full-matrix`,
`evidence:launch-packet`, `evidence:nominal`, `evidence:preflight`,
`evidence:proposal`, `evidence:smoke`, `evidence:stress`, and
`evidence:synthesis`.

`evidence:proposal`, `evidence:smoke`, and `evidence:launch-packet` do not become
paper-grade or admitted benchmark evidence without their separate contracts and
domain-aware review. `evidence:blocked` is a blocker, not a degraded success.

## Mutation rules

1. Read the live labels and the issue body before changing metadata.
2. Preserve explicit blocked, running, review, resource, and evidence markers;
   do not infer readiness from their absence.
3. Use the shared issue-audit plan and its REST readback for label changes; do not
   hand-create labels or silently rename them.
4. After recording a decision, verify the terminal issue/PR state from GitHub.

This taxonomy is intentionally small and semantic. A newly introduced label needs
an owner, a documented meaning, and a corresponding classifier or workflow test
before it is treated as part of the shared vocabulary.
