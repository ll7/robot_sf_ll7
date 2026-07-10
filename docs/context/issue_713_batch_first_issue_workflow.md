# Issue #713 Batch-First GitHub Workflow

This note defines the repository-local workflow for batching GitHub issue and Project #5 updates
without wasting GitHub API quota. It is the canonical instruction surface for GitHub API efficiency
in agentic workflows.

## Rule

- Prefer REST-backed reads and writes for ordinary GitHub objects: issues, labels, PRs, branches,
  commits, workflow runs, and repository metadata.
- Reserve GraphQL for GitHub Projects v2, review-thread operations, and genuinely cheaper nested
  bulk reads.
- Prefer GitHub MCP / GitHub app tools for interactive issue, PR, and project inspection when
  available. MCP means Model Context Protocol. Interactive inspection means ad-hoc review or triage
  work such as opening a PR, checking review context, commenting, or linking issues; batch scripts,
  CI updates, and read-only analytics should prefer REST or local `git` when those are cheaper.
  Switch to REST when GraphQL quota is low or when MCP abstracts a simple REST operation through a
  costly GraphQL path.
- Prefer local `git` for repository state that is already available locally: current branch,
  changed files, commit hashes, merge bases, and diffs.
- Do issue cleanup first: body rewrites, labels, comments, and title fixes.
- Do Project #5 routing second: status, priority, duration, and review metadata.
- Do derived score sync last: run the score helper once after the batch, not after each issue.
- Cache Project #5 IDs once per shell session, and prefer a local cache file for long-running or
  multi-agent work.
- Keep `gh` as the deterministic fallback for scripted project writes, score sync, REST reads, and
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

## API Selection

Use the cheapest source of truth for the question:

| Need | Preferred source | Notes |
| --- | --- | --- |
| Working tree, branch, changed files, merge base, commits | local `git` | Do not ask GitHub for state already present locally. |
| Interactive issue, PR, and project inspection | GitHub MCP / GitHub app tools | Prefer this for ad-hoc triage, review context, commenting, and linking when authorized. Fall back to REST, local `git`, or narrow GraphQL when MCP/app tools are unavailable or would hide a costly GraphQL path. |
| Complete issue thread before implementation | `scripts/dev/gh_issue_rest.py thread <n>` | Tries concise `gh issue view --comments`, then falls back to paginated REST only for the known `repository.issue.projectCards` failure (issue #5092). |
| Structured issue fields or explicit REST reads | `scripts/dev/gh_issue_rest.py view <n> --comments`, or REST via `gh api repos/ll7/robot_sf_ll7/issues/...` directly | The `view` interface remains REST-only and normalizes fields for machine consumers (issue #5021). |
| PR metadata, branch refs, commits, workflow runs | REST via `gh api` | Poll sparingly; use event/check URLs from known PRs when available. |
| Project #5 item status, priority, duration, reviewed date | GraphQL / `gh project item-*` | Projects v2 is GraphQL-only; batch and cache aggressively. |
| Review-thread resolution or nested review data | GraphQL | Keep queries narrow and use known node IDs. |

Do not use a broad GraphQL query for ordinary issue or PR cleanup when a REST endpoint or local
Git command can answer the same question.

### Issue-with-comments helper

`gh issue view <number> --comments` fails on some GitHub CLI versions because it requests the
deprecated classic-Projects GraphQL field `repository.issue.projectCards` (issue #5021). Autonomous
workflows that must read an issue together with its comment thread should use the shared helper:

```bash
# preferred complete read: native CLI first, targeted REST fallback
uv run python scripts/dev/gh_issue_rest.py thread <number> --repo ll7/robot_sf_ll7

# explicit REST and normalized JSON fields for machine consumers
uv run python scripts/dev/gh_issue_rest.py view <number> --json number title state url labels

# library use for Python callers
from scripts.dev.gh_issue_rest import fetch_issue_with_comments
payload = fetch_issue_with_comments(<number>)
```

The `thread` command preserves successful native output. It falls back only when stderr names the
known `repository.issue.projectCards` field; authentication, authorization, and other failures stay
visible instead of being masked. The REST path preserves API comment order, paginates comments, and
fails closed when a thread exceeds the page budget. The `view` command remains REST-only and
normalizes `state`/`url` for `gh issue view --json` consumers.

## Project #5 Cache

For repeated Project #5 work, create a local, git-ignored cache at `.github/cache/project5.json`
using `docs/templates/github.project5-cache.example.json` as the shape. The cache may store:

- project ID,
- item IDs for issues touched in the current batch,
- field IDs,
- single-select option IDs,
- the timestamp/source command used to refresh the cache.

Rules:

- Treat cache values as hints, not permanent truth. Refresh before destructive or broad writes, or
  when a mutation fails because an ID is stale.
- Do not commit `.github/cache/` files; they are local state and may expose workflow assumptions.
- In a single-agent batch, resolving IDs once per shell session is enough.
- In multi-agent or long-running work, use the local cache to avoid each agent rediscovering the
  same project, field, and option IDs.

## Why This Matters

- It keeps issue cleanup and derived metadata from being mixed together.
- It reduces repeated `gh project item-list` and `gh project field-list` calls.
- It lowers the chance that GraphQL quota exhaustion interrupts the middle of a batch.
- It prevents non-project operations from burning the Projects v2 GraphQL budget.
- It makes failed project writes easier to resume because the batch has known item and field IDs.

## Operational Notes

- Check `gh api rate_limit` before a large GitHub batch and whenever GitHub calls start failing.
  Pay attention to both `resources.core.remaining` and `resources.graphql.remaining`.
- If GraphQL quota is low, stop Project #5 writes and keep working on issue text through REST.
- If REST core quota is low, stop nonessential GitHub reads and use local repository state.
- If `Retry-After` or reset headers indicate throttling, do not spin in a retry loop; leave a
  handoff with the pending mutation and reset time.
- MCP-first does not mean MCP-only: issue cleanup can use MCP, while score sync and some batch
  project writes may still be best done through `gh`.
- Use `scripts/dev/gh_comment.sh` for multiline comments instead of ad hoc `gh` heredocs.
- Keep the batch small enough that a retry does not make the project state ambiguous.
- Project #5 mutations should happen only on meaningful state transitions, such as `Tracked`,
  `Ready`, `In progress`, `Hold`, and `Done`; do not model transient agent phases as project
  statuses.
- When a Project #5 write is blocked by GraphQL exhaustion, finish the REST issue/body/label work,
  report the exact pending project mutation, and resume the project write after reset.

## Multi-Agent Coordination

Avoid running several agents that independently poll GitHub with the same token. For broad issue or
PR work:

- appoint one main agent as the GitHub writer,
- let side agents use local repository state or already-fetched issue bodies where possible,
- deduplicate issue/PR/project reads before dispatching parallel work,
- batch project writes through one process,
- separate background IDE or automation tokens from the token used for Codex work when possible.

For larger automation, prefer a GitHub App or brokered local queue over several PAT-backed agents
sharing one rate-limit bucket. The broker should cache reads, throttle writes, prioritize project
mutations, and degrade to local/REST-only mode when GraphQL quota is low.

### Cross-Machine Issue Claims

When two `goal-autopilot` or `goal-issue-implementation` runs may execute on different PCs, use a
stable remote Git ref as the implementation mutex before creating a worktree or branch:

```bash
uv run python scripts/dev/issue_claim.py acquire <issue-number>
```

The helper creates `refs/heads/agent-claims/issue-<issue-number>` through GitHub's create-ref API,
which fails when the ref already exists. That makes acquisition atomic enough for this workflow: the
first PC succeeds, and later PCs fail instead of doing the same implementation work. A failed
acquisition means "skip this issue now", not "retry until it wins". Inspect the current claim with:

```bash
uv run python scripts/dev/issue_claim.py status <issue-number>
```

After acquiring a claim, make it visible to humans and other agents by moving the issue to
`In progress`/`state:running`, assigning the actor when practical, and adding a short comment with:

- claim ref: `agent-claims/issue-<issue-number>`;
- machine/thread identity;
- intended implementation branch or worktree;
- stale-claim cleanup condition.

After a PR is open and duplicate-PR checks can see that the issue is covered, release the transient
claim:

```bash
uv run python scripts/dev/issue_claim.py release <issue-number>
```

Only release another run's claim after checking that there is no open PR, no recent claimant comment,
and the claim is clearly stale or abandoned. Do not use Project #5 status or labels as the mutex;
they are useful visibility and routing metadata, but they are not atomic across two PCs.

## Diagnostic Boundary

- This workflow is about GitHub issue/project hygiene.
- It does not change Project #5 semantics or score math.
- It is guidance for batching, API selection, caching, and rate-limit behavior; it does not require
  a new automation service before ordinary issue work can proceed.
