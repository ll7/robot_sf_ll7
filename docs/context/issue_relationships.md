# Issue relationships

Native GitHub relationships are canonical; the `## Relationships` block in an
issue body mirrors intentional links so audits can diff the two. Use
`scripts/dev/audit_issue_relationships.py` to audit one issue, and to migrate
additive native links only after explicit confirmation.

## Canonical mirror block

```markdown
## Relationships

<!-- Native GitHub relationships are canonical; this block mirrors intentional links. -->
- Parent issue: #9293
- Blocked by: none
- Blocking: none
- Relates to: none
```

Rules: `Parent issue` takes at most one `#N` reference; `Blocked by` takes
same-repository `#N` references; `Blocking:` rows are owned by the other
issue's `Blocked by` row and are never written by the helper; `Relates to` is
informational and never written. Placeholder tokens (`TBD`, `unknown`),
multiple parents, and full URLs to other repositories are refused, never
guessed.

## Commands

Read-only audit (default, no writes):

```bash
uv run python scripts/dev/audit_issue_relationships.py --issue 123
uv run python scripts/dev/audit_issue_relationships.py --issue 123 --json
```

Guarded migration (additive only; verifies read-back and fails closed):

```bash
uv run python scripts/dev/audit_issue_relationships.py --issue 123 --apply --confirm RELATIONSHIP_MIGRATION
```

Reads: REST issue bodies, GraphQL `issue.parent` / `issue.subIssues`, and the
REST `issues/{n}/dependencies/blocked_by|blocking` lists (all bounded to 30
seconds). Writes: REST `issues/{parent}/sub-issues` and
`issues/{n}/dependencies/blocked_by`, each verified by re-read. Anything
requiring deletion, disambiguation, or a cross-repository link is refused with
a stable reason. The REST sub-issue *list* endpoint is unavailable on this
repository, so native children are read through GraphQL.
