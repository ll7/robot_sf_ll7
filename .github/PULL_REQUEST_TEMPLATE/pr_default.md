## Summary

Describe what changed and the user or maintainer value in one or two sentences.

## Linked Issues

<!-- Keep issue references plain so GitHub recognizes closing keywords after merge. -->
- Closes #<id>
- Relates to #<id>

## Stack / Dependency

- Base dependency: none, `PR #<id>`, or branch `<name>`.
- Required prior PRs and stack follow-up issues: none or issue/PR references.
- Safe to review independently: yes or no, with the dependency reason.

## What Changed

- Key implementation changes.
- Relevant tests, docs, or configuration updates.

## Why It Matters

- Expected impact and compatibility considerations.

## Research / Evidence Notes

For evidence-bearing changes, describe the targeted claim, comparator, mechanism activation,
failure mode, result route, and next empirical action. For support/tooling/docs-only changes, write
`Not applicable — no research claim.` The machine-readable evidence and approval fields are in the
v2 block below.

## Validation / Proof

- Commands run and the observed result.
- For evidence-bearing changes, name the durable/versioned input and the claim boundary.

## Risks / Rollback

- Failure modes, rollout constraints, and rollback plan.

## Docs / Provenance

- Updated docs, design/provenance notes, and assumptions that must remain stable.

## Downstream Propagation

- State parent issue, claim-map, benchmark-report, registry, catalog, context-note, or follow-up
  updates; write `Not applicable — support change.` when none apply.

## Follow-Up / Residual Scope

- State what remains, or explain why no deferred work remains.

## Reviewer Notes

- Anything a reviewer should verify closely.

<!-- Keep this marker and its keys unchanged. Update values to match the human narrative and
changed files. PRs without this marker remain on the v1 Markdown compatibility path. See
docs/context/issue_7665_pr_contract_v2.md. -->

<!-- pr-contract:v2
change_class: tooling
linked_issues:
  closes: []
  relates: []
deferred_work:
  status: none
  issues: []
  reason: ""
evidence:
  applicability: na
  tier: null
  result: na
domain_approval:
  required: false
  status: not_required
  domains: []
  note: "NA - support/tooling change; no experimental claim."
performance:
  claimed: false
-->
