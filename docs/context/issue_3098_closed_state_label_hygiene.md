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

- #1108: closed issue that previously had `state:blocked`; no live state label remains.
- #2259: closed issue that previously had `state:ready`; no live state label remains.
- #2382: closed issue that previously had `state:running`; no live state label remains.

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
- Read-then-write: the job re-confirms the issue is `CLOSED` (and not a pull request) via
  `gh issue view` before any removal, as defense-in-depth even though the trigger implies closure.
- Removal is guarded by label presence: it fetches the issue's current labels and removes only the
  live `state:*` labels that are actually present, so it is a no-op when none are stale and never
  fails the job for a missing label. Only the documented live state set is touched — no other label.
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
