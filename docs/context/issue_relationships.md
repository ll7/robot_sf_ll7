# Native GitHub issue relationships

[Back to Documentation Index](../README.md)

GitHub's native issue relationships are the source of truth for issue dependencies and hierarchy.
Do not duplicate those links in issue or pull request (PR) bodies. Issue bodies and comments may
explain why a relationship exists; they are evidence for deciding whether to create a native link,
not a second relationship store.

## Supported relationships

| Relationship | Meaning | Native workflow |
| --- | --- | --- |
| Parent | This issue is a child of the named issue | Supported; one parent at most. Use for clear epic/child hierarchy. |
| Blocked by | This issue cannot proceed until the named issue is resolved | Supported; use only for a current prerequisite. |
| Blocking | The named issue cannot proceed until this issue is resolved | Supported; usually add `Blocked by` on the dependent issue instead. |
| Relates to | Informational association without a dependency | Excluded from the CLI relationship workflow; `gh issue` has no supported write flag. |

GitHub may display issue references from prose, but a mention alone does not create a dependency.
Use body and comment context to establish intent. A blocker may be inferred when the discussion
clearly says the target must be resolved before the issue can proceed. Do not create blockers from
parallel work, optional inputs, downstream consumers, completed ordering predecessors, or vague
references. If evidence conflicts or the prerequisite is unclear, leave the relationship untouched
and record the issue for review.

Keep links within the same repository. Do not create an issue edge for an external artifact,
runtime, license, or decision. A parent must be unambiguous and must not create a hierarchy cycle.

## GitHub CLI

Direct CLI relationship operations require GitHub CLI (`gh`) **2.100.0 or newer**. Check the local
version and use an explicit repository context:

```bash
gh version
gh issue create --repo <owner>/<repository> --title "..." --body-file issue.md --parent 123
gh issue edit 999 --repo <owner>/<repository> --add-blocked-by 456
gh issue edit 999 --repo <owner>/<repository> --add-blocking 890
gh issue view 999 --repo <owner>/<repository> --json number,parent,blockedBy,blocking
```

`--parent`, `--add-sub-issue`, `--add-blocked-by`, and `--add-blocking` set native relationships.
The corresponding `--remove-*` flags remove them. Read native state back after every write; command
success output alone is not proof that GitHub changed the relationship. If a relationship field is
unsupported by the installed CLI, stop and use the supported GitHub Relationships panel or the
documented REST route. Never report an unsupported field as absent.

## Review and write boundary

For a relationship decision, inspect the current issue body, relevant comments, target issue, and
existing native state. Record the source issue and body/comment evidence in the review artifact so
the decision can be checked later. Refresh the issue immediately before writing if its discussion
changed after review.

Relationship writes are remote control-plane mutations. Make them from the owning writable
implementation/publication worktree after a fresh read. Review-only worktrees may inspect the graph
but never write links. Add only reviewed links; do not replace a different parent or remove an
existing edge without explicit evidence that it is wrong. Read back every changed issue and preserve
the exact issue numbers, evidence references, commands, and final native state in the handoff.

PRs continue to use `Closes` and `Refs` for closure and coverage semantics. Reviewers verify the
linked issue's current native Parent/Blocked by/Blocking state directly; PR text does not mirror the
graph. A closing reference is not itself a parent or dependency relationship.

## Auditing existing issues

Use the read-only relationship audit before a migration:

```bash
uv run python scripts/dev/audit_issue_relationships.py --state open --format json
```

The audit reads explicit legacy `## Relationships` blocks as candidate evidence when they remain in
old issue bodies. A missing block is normal and is not an audit failure. Legacy headings and
incidental mentions remain review-only; the audit does not infer or write links from them. Use its
explicit confirmation-gated apply mode only for reviewed, complete, unambiguous legacy declarations.
For the current native graph, read `parent`, `blockedBy`, and `blocking` through `gh issue view` or
`gh issue list`; native state takes precedence over any older body text.

For large reviews, capture one complete issue and comment inventory, verify pagination and comment
counts, and divide disjoint issue sets among read-only reviewers. Every proposed edge must carry its
source body/comment evidence, reason, and uncertainty. Re-read changed issues before applying links.
