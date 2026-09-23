# Explicit issue relationships

[Back to Documentation Index](../README.md)

Issue references in prose are useful context, but they are not a machine-readable issue graph.
For every issue, declare intentional relationships explicitly and mirror them in GitHub's
**Relationships** panel when the issue is created or updated.

## Canonical body block

Use one `## Relationships` block in every issue body:

```markdown
## Relationships

<!-- Native GitHub relationships are canonical. Mirror every intentional link here for review and
auditability. Use `none` when a relationship does not apply. Do not infer a relationship from a
mention in another section. -->
- Parent issue: none
- Blocked by: none
- Blocking: none
- Relates to: none
```

Replace `none` with one or more same-repository issue references (`#123` or a canonical issue URL)
when the relationship is intentional. Keep external dependencies as prose in the relevant
contract section; do not invent an issue number for an external artifact, runtime, license, or
decision.

Use references that remain unambiguous outside the immediate issue view:

- In human-facing issue and PR body mirrors, prefer `#123` for an issue in this repository.
- In automation input, audit output, or a cross-repository context, use the canonical URL
  `https://github.com/<owner>/<repository>/issues/123`.
- In a CLI command, an issue number is valid only with an explicit `--repo <owner>/<repository>`;
  use the canonical URL when the repository context is not fixed.
- Do not use a title, a bare number in prose, a PR URL, or an external issue URL as a relationship
  declaration. `none` means an explicit, reviewed absence—not an unknown or unsupported field.

The fields have these meanings:

| Field | Direction | Native GitHub support | Use |
| --- | --- | --- | --- |
| Parent issue | This issue is a child of the named issue | Yes | One parent at most; use for epics and extracted child work. |
| Blocked by | This issue cannot proceed until the named issue is resolved | Yes | Name only concrete issue blockers. Typed dependency packets remain the source for richer predicates. |
| Blocking | The named issue cannot proceed until this issue is resolved | Yes | Prefer adding the reciprocal `Blocked by` link on the blocked issue. |
| Relates to | Informational association | UI preview/manual | Keep the body mirror; do not assume a REST/CLI write path. |

“Security alert” in the GitHub menu is a Dependabot-specific association, not a general-purpose
relationship field for repository issues.

## GitHub CLI capability gate

The direct command-line relationship workflow requires GitHub CLI (`gh`) **2.100.0 or newer**.
This is the repository-tested floor (not a claim about the upstream feature's first release). Check
the installed version before using relationship flags:

```bash
gh version
gh issue view <issue> --repo <owner>/<repository> --json parent,blockedBy,blocking
```

If the version is older than 2.100.0, or the readback reports `Unknown JSON field`, stop using
`--parent`, `--blocked-by`, `--blocking`, and the `--add/remove-*` relationship flags. Upgrade
`gh`, use the GitHub Relationships panel, or use the REST-backed
`scripts/dev/audit_issue_relationships.py` route. Never turn an unsupported field into `none`.

With a supported CLI, use explicit repository context and read the native state back:

```bash
gh issue create --repo <owner>/<repository> --title "..." --body-file issue.md \
  --parent 123 --blocked-by 456,457 --blocking 890
gh issue edit 999 --repo <owner>/<repository> \
  --add-blocked-by 456 --add-blocking 890
gh issue view 999 --repo <owner>/<repository> --json number,parent,blockedBy,blocking
```

The CLI still does not expose `Relates to`; create that informational link in the panel and keep
the body mirror.

## PR and worktree handoff

Pull requests do not replace the issue graph. Keep the existing `Closes`/`Refs` entries in the PR
template for closure and coverage semantics, and add one explicit mirror for graph edges:

```markdown
## Issue Relationship Mirror

<!-- The linked issue's native relationships are canonical. Mirror only intentional same-repository
links here so a PR can be reviewed from its owning worktree. Use `none` when a field does not apply. -->
- Parent issue: none
- Blocked by: none
- Blocking: none
- Relates to: none
```

For a PR that changes relationship intent, update the issue body and native GitHub relationship
first, then mirror the resulting state in the PR body. A PR may use `Closes` or `Refs` for its
primary issue without treating that coverage reference as a parent, blocker, or informational
relationship. A PR with no issue relationship should say so explicitly with `none` and explain the
support-only scope in its summary.

Relationship writes are remote control-plane mutations. Perform them from the owning writable
implementation/publication worktree after a fresh issue read; never write them from a review-only
worktree. Review-only worktrees may inspect the issue graph and verify the PR mirror, but must not
push, edit bodies, or create native links. Preserve the exact issue, PR, branch, and head-SHA
evidence in the handoff so another worktree can re-read before publishing.

## Creation and migration workflow

1. Fill the relationship block before creating the issue. Use `none` explicitly when there is no
   relationship.
2. Create the issue from one of the repository templates or forms.
3. Set Parent, Blocked by, and Blocking in GitHub's Relationships panel or through the CLI gate
   above. The native links are the operational source of truth; the body block is a reviewable
   mirror.
4. Add Relates to links manually in the panel when useful, and keep their body mirrors until a
   supported API is available.
5. For existing issues, run the bounded, read-only audit first:

   ```bash
   uv run python scripts/dev/audit_issue_relationships.py --state open --format json
   ```

   Use `--state all` for a bounded review that includes closed issues; increase `--max-pages` only
   when the report remains complete.

   The audit considers only canonical `## Relationships` declarations safe for a migration
   proposal. Legacy headings such as `## Parent`, `## Related`, or free-form mentions are reported
   as review candidates, never silently converted. A migration is opt-in and fails closed on
   ambiguous, conflicting, cross-repository, or external references:

   ```bash
   uv run python scripts/dev/audit_issue_relationships.py \
     --issue 123 --apply --confirm RELATIONSHIP_MIGRATION
   ```

   Review the JSON report before applying a larger scope. The command never writes `Relates to`
   links and never replaces an existing different parent.

6. For PRs opened from a worktree, copy the reviewed native state into the PR's `## Issue
   Relationship Mirror` section. Re-read the linked issue and PR head immediately before publication;
   if either changed, stop and refresh the mirror and validation evidence.

The repository audit uses the REST endpoints when an explicit apply is requested so the workflow
remains usable with older CLI versions; authentication, permissions, and a fresh read-back are
required. Direct CLI writes remain gated at `gh >= 2.100.0` as described above.

Reference documentation: [`gh issue create`](https://cli.github.com/manual/gh_issue_create),
[`gh issue edit` flags](https://cli.github.com/manual/gh_help_reference), and
[GitHub issue dependencies](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-issue-dependencies).

This contract complements, rather than replaces, typed dependency packets. A relationship says
which issue is connected; a dependency packet records the predicate, evidence, freshness, and
unblock condition needed to decide whether work may proceed.
