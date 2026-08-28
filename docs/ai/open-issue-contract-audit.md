# Open issue contract audit

`audit_open_issue_contracts.py` is the report-only, repository-wide consumer of the generic issue
implementation-admission contracts owned by #7609.

## Authority boundary

The audit does not define readiness. It reuses:

- `issue_implementability.v1` for issue state, execution-contract fields, and generic blocker
  classification;
- `issue_dependency_packet.v1` when an issue contains an explicit dependency packet;
- the atomic claim owner for current claim state;
- `goal_issue_admission.py` as the only normal implementation-claim entry point.

The report therefore names a preparation action, not an approval. It does not create labels, edit
issues, assign work, create sub-issues, authorize compute, resolve decisions, admit evidence, or
make a merge/release/scientific judgment.

The JSON summary includes `admission_reason_histogram` and `not_admitted`, so an empty claimable
queue identifies the blocking reason (for example `wrong_owner_repo`, `external_input_missing`,
`parent_not_leaf`, `needs_spec`, or `covering_pr_open`) instead of collapsing every row into one
count. The report remains a candidate inventory until each proposed leaf passes a fresh
`goal_issue_admission.py --check-only` call.

## Live audit

Run from a current Robot SF checkout with authenticated `gh` read access and the canonical Git
remote available:

```bash
uv run python scripts/dev/audit_open_issue_contracts.py \
  --repo ll7/robot_sf_ll7 \
  --remote origin \
  --json-report output/open_issue_contract_audit.json \
  --format markdown \
  --output output/open_issue_contract_audit.md \
  --check
```

The command pages through the REST issues endpoint, excludes pull requests, and re-reads each issue
through the exact-item owner before calling the canonical implementability classifier. A full final
page at the page limit, duplicate identity, malformed row, exact-read failure, unavailable claim
state, or listing-to-exact-read drift makes the report non-applicable.

For an explicitly multi-repository contract, pass a fresh route artifact with
`--route-preflight-json`; absent or expired route evidence remains non-claimable. The artifact is
checked for a selected route, config digest, and timezone-qualified timestamp within the 30-minute
default TTL.

Exit codes are:

- `0`: complete and applicable report;
- `2`: with `--check`, report produced but fail-closed because pagination or an exact item is
  non-applicable;
- `1`: malformed input or an unexpected operational failure.

## Offline fixture

Use an offline fixture for deterministic tests and review:

```bash
uv run python scripts/dev/audit_open_issue_contracts.py \
  --fixture tests/fixtures/open_issue_contract_audit/example.json \
  --page-size 100 \
  --check
```

A fixture contains paginated listing rows, exact issue payloads, claim states, and optional canonical
dependency evaluations. Fixed fixture data produces byte-stable report content and a stable
`content_sha256`.

## Output use

The JSON report contains one bounded packet per issue: classification, missing contract fields,
reasons, dependency status, listing drift, next action, and responsible authority. The Markdown
view is deliberately capped and never copies full issue bodies.

Only issues classified as `ready`, without drift or operational error, appear in
`summary.executable_leaf_numbers`. Even these issues must pass a fresh
`goal_issue_admission.py` check immediately before atomic claim acquisition.
