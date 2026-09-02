# Issue #3098 Closed-State Label Hygiene 2026-06-18

Status: Current

Issue: [#3098](https://github.com/ll7/robot_sf_ll7/issues/3098)

## Summary

On 2026-06-18, a fresh run of `scripts/dev/closed_state_label_hygiene.py` found 134
closed issues that still carried live routing labels. The cleanup removed only these stale
`state:*` labels from closed issues through REST issue-label deletes. It did not reopen issues,
change Project #5 fields, change issue scores, or modify benchmark metadata.

## Before

Command:

```bash
uv run python scripts/dev/closed_state_label_hygiene.py --repo ll7/robot_sf_ll7 --limit 200
```

Result:

- stale closed issues: 134
- labels removed:
  - `state:ready`: 109
  - `state:running`: 19
  - `state:blocked`: 6

The private per-issue source report for this run is stored in the common Git-dir artifact
`codex-agent-runs/active/issue-3098/before.json`.

## Cleanup

Each issue-label removal used the REST endpoint:

```text
DELETE /repos/ll7/robot_sf_ll7/issues/{issue_number}/labels/{label_name}
```

Attempted removals: 134.

Failed removals: 0.

Skipped cases: none.

The private REST write result is stored in
`codex-agent-runs/active/issue-3098/remove_labels_result.json`.

## After

Command:

```bash
uv run python scripts/dev/closed_state_label_hygiene.py --repo ll7/robot_sf_ll7 --limit 200
```

Result:

- `ok`: true
- stale closed issues: 0

The private after-report is stored in `codex-agent-runs/active/issue-3098/after.json`.

## Spot Checks

Representative issues were checked after cleanup:

- Issue #1108: closed issue that previously had `state:blocked`; no live state label remains.
- Issue #2259: closed issue that previously had `state:ready`; no live state label remains.
- Issue #2382: closed issue that previously had `state:running`; no live state label remains.

## Recurrence and Automation Follow-up

The 2026-06-18 cleanup was not durable: a fresh run on 2026-06-23 found 66 closed issues that had
re-accumulated live `state:*` labels (`state:ready`: 52, `state:running`: 13, `state:blocked`: 1),
which were removed via the same read-then-write (verify `CLOSED` before stripping) process. About 66
stale labels reappeared in roughly 5 days, confirming that periodic manual scrubs do not hold.

The guard `scripts/dev/closed_state_label_hygiene.py` is a detector only; no automatic fixer exists.
The durable fix is tracked in [Issue #3456](https://github.com/ll7/robot_sf_ll7/issues/3456): a
GitHub Action on `issues: closed` that strips live `state:*` labels at every close path (manual,
duplicate/wontfix, PR-merge), reusing the guard's `LIVE_STATE_LABELS` as the single source of
truth. Until that lands, expect to re-run the manual cleanup periodically.

## Boundary

This note records routing-hygiene evidence only. It is not benchmark evidence, research evidence,
or a change to issue taxonomy semantics. Future cleanup should continue to treat live `state:*`
labels on closed issues as stale queue metadata unless a specific issue documents an exception.

## Durable Automation Follow-up (Issue #3456)

The manual cleanups above were one-shot reconciliations: #3098 removed stale state labels from 134
closed issues, and a 2026-06-23 follow-up removed them from a further 66. Because issues keep closing
through many paths, stale labels re-accumulate, so issue [#3456](https://github.com/ll7/robot_sf_ll7/issues/3456)
adds automation so closed issues are de-labeled at the moment of closure instead of in periodic
sweeps.

### GitHub Action: `.github/workflows/strip-closed-state-labels.yml`

- Trigger: `on: issues: types: [closed]`. The `issues.closed` event is the single choke point that
  covers every close path — manual close, duplicate/wontfix, and PR-merge auto-close.
- Permissions: `issues: write` only (least privilege), using the default `GITHUB_TOKEN`.
- Read-then-write: the job re-confirms the requested number, exact `CLOSED`/`OPEN` state, explicit
  pull-request discriminator, and canonical issue identity via the REST-backed
  `gh_issue_rest.py view --json number state url is_pull_request` command before any removal, as
  defense-in-depth even though the trigger implies closure. Unknown or inconsistent identity is
  an error, while a valid open issue or pull request is a no-op. This state-only read does not
  fetch comments; complete thread reads remain opt-in with `--comments`.
- Removal is guarded by label presence: it fetches the issue's current labels and removes only the
  live `state:*` labels that are actually present, so it is a no-op when none are stale and never
  fails the job for a missing label. Only the documented live state set is touched — no other label.
- Label inventory and removals use the verified REST helper: `gh_pr_label_rest.py list` reads the
  complete paginated label set, and `gh_pr_label_rest.py remove` verifies each deletion. The
  workflow validates each successful result envelope, and an allowlist-import or malformed-output
  failure is checked before it can enter a no-op path.
- Single source of truth: the live label set is read at runtime from
  `scripts/dev/closed_state_label_hygiene.py::LIVE_STATE_LABELS`; the workflow hard-codes no second
  copy of the label list.

### Detector `--fix` mode

`scripts/dev/closed_state_label_hygiene.py` now accepts `--fix`. Default (no flag) behavior is
unchanged: a read-only audit with the same exit codes. With `--fix` the detector strips the live
`state:*` labels (reusing `LIVE_STATE_LABELS`) from each closed issue it found, re-confirming every
issue is `CLOSED` before writing (read-then-write), and reports a per-issue `fix_actions` log. This
provides a manual/backfill path for sweeps; the Action handles steady-state closure events.

### Backstop

The periodic read-only detector remains the CI/backstop check: it still flags any closed issue that
slips through (for example, a closure event that predates the Action, or a label re-added later), so
the automation and the audit reinforce each other rather than replace one another.

## Open-Issue Stale-State Guard (Issue #7537)

Closed-issue cleanup does not catch the inverse queue failure: an issue can remain open with an active
`state:ready`, `state:running`, or `state:working` label after the exact implementation PR has merged.
Those rows are neither safe implementation supply nor trustworthy open work, so they need a separate
report-only audit.

Run the bounded guard with:

```bash
uv run python scripts/dev/open_state_label_hygiene.py --repo ll7/robot_sf_ll7
```

The guard reads open issues through REST, rechecks each candidate's current state and labels, follows
the issue timeline, and verifies each referenced PR's current merged state and `merge_commit_sha`.
Its `open_state_label_hygiene.v1` report includes the candidate issue, active labels, merged PR, merge
commit, timeline source, and `complete_for_open_issues` coverage flag. Any incomplete issue or timeline
inventory returns non-zero and must not be interpreted as a clean queue.

The guard never closes issues, removes labels, edits Project #5, or declares that a merged PR fully
satisfies the issue. Each candidate is `merged_reference_needs_exact_fix_review`; a maintainer or
issue-audit authority must verify the named-symbol/failing-signature boundary before closing or
relabeling it. This is routing metadata hygiene only, not benchmark, research, or publication evidence.

## Exact-fix review routing (Issue #7549)

The report can now be handed to a deterministic, no-write review queue:

```bash
uv run python scripts/dev/open_state_label_hygiene.py \
  --repo ll7/robot_sf_ll7 > output/open_state_label_hygiene.json
uv run python scripts/dev/route_exact_fix_audit.py \
  --report output/open_state_label_hygiene.json \
  --output output/exact_fix_review_queue.json
```

`route_exact_fix_audit.py` requires a complete `open_state_label_hygiene.v1` report and preserves
its digest, issue links, active labels, merged PRs, and merge commits in an
`exact_fix_review_queue.v1` packet. Every candidate is checked against the exact-fix checklist:
the named symbol, failure signature, failing file/line, regression proof, current-main SHA, and
the verified issue-timeline covering PR. Missing fields are classified as
`needs_exact_fix_evidence`; a packet with all fields is merely
`ready_for_manual_exact_fix_review`.

An optional evidence manifest can provide the five explicit fields for a later maintainer review:

```json
{
  "schema": "exact_fix_evidence.v1",
  "issues": [
    {
      "number": 123,
      "covering_pr": 456,
      "named_symbol": "scripts/dev/route_exact_fix_audit.py:build_review_queue",
      "failure_signature": "ValueError: stale label",
      "failing_file_line": "scripts/dev/route_exact_fix_audit.py:202",
      "regression_proof": "tests/dev/test_route_exact_fix_audit.py::test_build_review_queue_routes_without_authorizing_a_disposition",
      "current_main_sha": "<40-hex-main-sha>"
    }
  ]
}
```

The route never closes or relabels an issue and never treats an issue-number match or merged PR
alone as an exact fix. The resulting `pending_decisions` rows are handed to the existing
maintainer-facing issue-audit lane for one explicit disposition at a time.
