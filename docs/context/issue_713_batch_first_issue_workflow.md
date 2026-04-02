# Issue #713 Batch-First GitHub Workflow

This note defines the repository-local workflow for batching GitHub issue and Project #5 updates
without wasting GitHub API quota.

## Rule

- Prefer GitHub MCP / GitHub app tools for interactive issue, PR, and project inspection when
  available.
- Do issue cleanup first: body rewrites, labels, comments, and title fixes.
- Do Project #5 routing second: status, priority, duration, and review metadata.
- Do derived score sync last: run the score helper once after the batch, not after each issue.
- Cache the project ID and field IDs once per shell session and reuse them for all edits in that batch.
- Keep `gh` as the deterministic fallback for scripted project writes, score sync, and
  auth/debugging.

## Recommended Sequence

1. Collect the issue set you want to touch.
2. Finish all issue text and label edits in one pass.
3. Resolve the Project #5 IDs once:
   - `PROJECT_ID`
   - `Status` field ID and option IDs
   - `Priority` field ID and option IDs
   - any number/date field IDs you need
4. Apply Project #5 updates in a separate pass.
5. Run `scripts/tools/project_priority_score.py sync` once at the end.

## Why This Matters

- It keeps issue cleanup and derived metadata from being mixed together.
- It reduces repeated `gh project item-list` and `gh project field-list` calls.
- It lowers the chance that GraphQL quota exhaustion interrupts the middle of a batch.

## Operational Notes

- If GraphQL quota is low, stop project writes and keep working on issue text first.
- MCP-first does not mean MCP-only: issue cleanup can use MCP, while score sync and some batch
  project writes may still be best done through `gh`.
- Use `scripts/dev/gh_comment.sh` for multiline comments instead of ad hoc `gh` heredocs.
- Keep the batch small enough that a retry does not make the project state ambiguous.

## Diagnostic Boundary

- This workflow is about GitHub issue/project hygiene.
- It does not change Project #5 semantics or score math.
- It is guidance for batching and ordering, not a new automation service.
