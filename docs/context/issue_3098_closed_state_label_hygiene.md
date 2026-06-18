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

## Boundary

This note records routing-hygiene evidence only. It is not benchmark evidence, research evidence,
or a change to issue taxonomy semantics. Future cleanup should continue to treat live `state:*`
labels on closed issues as stale queue metadata unless a specific issue documents an exception.
