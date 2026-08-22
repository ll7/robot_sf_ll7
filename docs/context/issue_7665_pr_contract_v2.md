# PR contract v2

Issue [#7665](https://github.com/ll7/robot_sf_ll7/issues/7665) introduces a compact machine-readable
contract for pull-request metadata. It separates stable machine checks from the human narrative
without changing the evidence, approval, follow-up, performance, or exact-head policy.

## Boundary

The contract is a workflow input, not evidence of implementation correctness or scientific validity.
A valid block only says that the declared metadata is internally consistent. Tests, review, hosted
checks, domain approval, and benchmark provenance remain independent gates.

The first migration slice keeps the v1 Markdown parser. A PR with no `pr-contract:v2` marker uses
the existing headings. A PR containing the marker must pass v2 validation; malformed v2 metadata
fails closed and never silently falls back to v1.

## Block shape

The marker is one HTML comment whose first line is exactly `<!-- pr-contract:v2`:

```markdown
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
```

The validator rejects unknown keys, duplicate YAML keys, duplicate issue references, invalid enum
values, wrong scalar types, contradictory fields, and missing class-specific values. Issue
references are positive integers. Evidence-bearing contracts require a canonical evidence tier,
result classification, approved or waived domain status, domains, a note, and all five validity
checklist fields. Performance claims require baseline/changed measurements, a reproducible command,
the hot-path anchor, cache disposition, and rollback criterion.

## Change classes

The supported values are `docs`, `tooling`, `runtime`, `benchmark_or_metric`, `paper_or_claim`, and
`performance`. Benchmark or paper/claim classes must declare evidence-bearing applicability. A
docs declaration cannot include substantive source changes. Any benchmark/metric/paper title or
changed-file signal must use evidence-bearing metadata rather than hiding behind a docs/tooling
declaration.

## Migration

Bots and agents should keep the human sections concise, update the v2 values in the same change,
and leave v1 headings in older open PRs until those PRs are migrated. Do not remove v1 parsing or
change branch-protection requirements in this first slice. Exact-head carriers may remain in the
human body; the optional `exact_head` field is available when a workflow chooses to carry that
value inside v2.

The shortened default template preserves the reusable v1 guidance through a field map:

| v1 surface | v2 or human destination |
| --- | --- |
| Summary, What Changed, Why It Matters | Human narrative sections; substantive changes still require a human-authored summary. |
| Linked Issues | Human issue references plus `linked_issues.closes` and `linked_issues.relates`. |
| Stack / Dependency | Human section; dependency metadata is intentionally not reinterpreted by v2. |
| Research Result Guidance | `evidence` and `domain_approval`, with claim/comparator detail in Research / Evidence Notes. |
| Falsification / Non-Transfer Check | Research / Evidence Notes for evidence-bearing work. |
| Next Empirical Action | Research / Evidence Notes and deferred-work fields when action remains. |
| Performance Evidence | `performance`; claimed measurements remain mandatory for `change_class: performance`. |
| Risks / Rollout, Docs / Provenance, Downstream Propagation | Human sections, including explicit not-applicable dispositions. |
| Follow-Up Issues | `deferred_work`; linked issues are checked for open state when requested. |
| Reviewer Notes and shared-helper migration guidance | Human Reviewer Notes plus this migration document. |
| Exact-head SHA carriers | Existing body carriers or the optional `exact_head` field. |

This map is a compatibility boundary: the v2 parser does not make a valid machine block a
substitute for implementation review, domain review, durable evidence, or downstream propagation.

## Validation

```bash
uv run pytest tests/dev/test_check_pr_followups.py -q
uv run pytest tests/test_ci_script_contract.py -q
uv run ruff check scripts/dev/pr_contract_v2.py scripts/dev/check_pr_followups.py
```
