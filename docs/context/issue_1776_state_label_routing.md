# Issue #1776 State Label Routing 2026-05-31

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1776>

## Purpose

This note records the conservative state-label mapping and bounded cleanup pass for open issue
routing. It does not introduce new labels. Use existing labels first, and treat issue-body reality,
open PR coverage, and blocker labels as stronger routing signals than `state:ready` alone.

## Canonical Mapping

| Queue reality | Existing label expression | Queue classification | Action |
| --- | --- | --- | --- |
| Clear, bounded local implementation with no active PR | `state:ready` + `resource:local` | `ready_local` | Implement next. |
| Clear issue currently covered by an open PR | `state:running` | `covered_by_pr` | Wait for PR outcome; do not reimplement. |
| Follow-up issue depends on an unmerged `source_pr` or explicit prerequisite PR | `state:blocked` or leave out of clean-main queue | `blocked_other` | Wait for prerequisite merge, or choose an explicit stacked-PR route. |
| Requires SLURM/Auxme execution | `state:blocked` + `resource:slurm` | `blocked_slurm` | Skip locally; update ledger or launch only from SLURM host. |
| Requires CARLA runtime | `state:blocked` + `resource:carla` | `blocked_external` | Skip locally unless CARLA host is available. |
| Requires unavailable dataset/model/artifact | `state:blocked` + `resource:external-data` or `state:needs-artifact-promotion` | `blocked_external` | Mark unblock condition; do not count as ready. |
| Maintainer decision needed | `decision-required` | `ambiguous` | Clarify before implementation. |
| Parent, epic, umbrella, or analysis-only issue | `epic`, `type:analysis`, `type:synthesis`, or body text naming a parent/synthesis role | `parent_or_epic` or `analysis_only` | Split, synthesize, or clarify; do not implement as a direct PR. |
| Closed issue with stale state label | Closed issue state wins | `covered_by_pr` or `blocked_other` | Remove stale `state:*` label when touched, or explicitly ignore closed issues in queue queries. |

`state:ready` is therefore necessary but not sufficient for implementation routing. Agents should
exclude open issues with active PR coverage, blocker/resource contradictions, parent/epic bodies,
analysis-only scope, or unmerged source-PR prerequisites before selecting a worktree branch. Before
branching from clean `origin/main`, inspect issue bodies/comments for `source_pr`, linked PRs,
`Closes`/`Refs`, and explicit prerequisite PRs, then verify the referenced implementation surface
exists on current `origin/main`.

## Bounded Audit On 2026-05-31

Baseline `state:ready` query before cleanup:

```text
#1784, #1783, #1782, #1781, #1776, #1775, #1774, #1773, #1772, #1771, #1653
```

Open PR coverage found for:

| Issue | PR | Action |
| --- | --- | --- |
| Issue #1653 | PR #1793 | Changed from `state:ready` to `state:running`. |
| Issue #1771 | PR #1790 | Changed from `state:ready` to `state:running`. |
| Issue #1772 | PR #1787 | Changed from `state:ready` to `state:running`. |
| Issue #1775 | PR #1786 | Changed from `state:ready` to `state:running`. |
| Issue #1781 | PR #1789 | Changed from `state:ready` to `state:running`. |
| Issue #1782 | PR #1788 | Changed from `state:ready` to `state:running`. |
| Issue #1783 | PR #1791 | Changed from `state:ready` to `state:running`. |
| Issue #1784 | PR #1792 | Changed from `state:ready` to `state:running`. |

Merged-PR stale ready issues:

| Issue | Covered by | Action |
| --- | --- | --- |
| Issue #1773 | PR #1779, commit `a11ecc02` | Closed with a corrective comment and removed `state:ready`. |
| Issue #1774 | PR #1778, commit `afd7a7fb` | Closed with a corrective comment and removed `state:ready`. |

Post-cleanup `state:ready` query:

```text
#1776
```

Post-cleanup `state:running` query:

```text
#1653, #1771, #1772, #1775, #1781, #1782, #1783, #1784
```

## Validation

Commands used:

```bash
gh issue list --state open --label state:ready --json number,title,labels,url --limit 200
gh issue list --state open --label state:running --json number,title,labels,url --limit 200
for n in 1773 1774; do gh issue view "$n" --json number,state,labels; done
for n in 1653 1771 1772 1775 1781 1782 1783 1784; do gh issue view "$n" --json number,labels; done
```

No Project #5 fields were changed in this pass. No new labels were created.

## Reusable Closed-Issue Guard

Issue #1833 added a read-only regression guard for the stale-label class found in this cleanup pass:

```bash
uv run python scripts/dev/closed_state_label_hygiene.py
```

The command checks closed issues carrying `state:ready`, `state:running`, or `state:blocked`,
prints a `closed_state_label_hygiene.v1` JSON report, and returns non-zero when stale labels are
present. It uses GitHub issue search only; it does not edit labels or write Project #5 metadata.
