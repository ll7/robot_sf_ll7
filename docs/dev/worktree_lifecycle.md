# Linked Worktree Lifecycle

This is the canonical task guide for isolated contributor work. Keep the main checkout untouched
when it contains user changes; create a sibling linked worktree from the current `origin/main`.

## Create and bootstrap

```bash
MAIN_REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKTREE_PARENT="$(dirname "$MAIN_REPO_ROOT")/$(basename "$MAIN_REPO_ROOT").worktrees"
git fetch origin main
scripts/dev/create_worktree.sh \
  --branch issue-123-short-description \
  --path "$WORKTREE_PARENT/issue-123-short-description" \
  --base origin/main
cd "$WORKTREE_PARENT/issue-123-short-description"
scripts/dev/bootstrap_worktree.sh
```

The capacity-guarded helper must create the worktree before editing, running PR validation,
pushing, or publishing. Use `--exec` when the first command must be bound to the new directory:

```bash
scripts/dev/create_worktree.sh \
  --branch issue-123-short-description \
  --path "$WORKTREE_PARENT/issue-123-short-description" \
  --base origin/main \
  --exec git rev-parse --show-toplevel
```

New branches are created without automatic upstream tracking. This avoids concurrent workers
contending on the shared repository configuration while they create linked worktrees. Configure a
remote explicitly when publishing a branch, for example with `git push -u origin <branch>`.

## Protected review worktrees

Review and synthetic-integration worktrees must opt into the protected mode explicitly:

```bash
scripts/dev/create_worktree.sh \
  --branch review/pr-123 \
  --path "$WORKTREE_PARENT/review-pr-123" \
  --base origin/main \
  --mode review
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/dev/review_worktree_guard.py integrate \
  --worktree "$WORKTREE_PARENT/review-pr-123" \
  --source-ref origin/main \
  --remote origin
```

The creator writes the worktree-local `robot-sf.worktree-mode=review` marker and installs the
tracked pre-push guard. Configured remote names also receive inert worktree-local push destinations
and a nonexistent worktree-local receive-pack command. An all-URL worktree-local rewrite (plus
exact push-URL rules) routes remote URLs to an inert path, covering inherited `pushurl` values,
equivalent local-path spellings, explicit destination refspecs, and `--no-verify`. Review mode also
denies Git's known and unknown transport protocols in the worktree-local config, so a longer
common-config URL alias cannot win URL-rewrite precedence and a remote added after activation stays
blocked. It therefore intentionally blocks direct fetch and `ls-remote` commands too; refresh refs
before entering the mode or use the integration helper, which reads its comparison through the
common Git config. No configured remote can be mutated from the protected worktree through ordinary
Git invocation paths. This is a Git-level workflow guard, not an operating-system sandbox; a
deliberate per-command Git configuration override can bypass it.

If the selected base predates the guard files, `create_worktree.sh --mode review` keeps the target
clean and temporarily points its worktree-local hooks path at the invoking checkout's tracked
guard and hook. Keep that invoking checkout available until the review worktree is restored or
removed; once the guard is present in the base, the target uses its own tracked files. In this
fallback, invoke the integration helper through the invoking checkout's wrapper, for example
`"$MAIN_REPO_ROOT/scripts/dev/run_worktree_shared_venv.sh" --standalone -- python
"$MAIN_REPO_ROOT/scripts/dev/review_worktree_guard.py" integrate --worktree <review-worktree>
--source-ref origin/main --remote origin`; the target does not contain the helper yet.
The integration helper snapshots every ref from `git ls-remote --refs`, runs
`git merge --no-commit --no-ff`, always attempts `git merge --abort`, restores the pre-probe
`ORIG_HEAD` pseudo-ref, and exits nonzero unless the worktree is clean and the before/after remote
snapshots are identical.

Ordinary implementation worktrees retain the default pushable behavior. To deliberately restore a
previously protected worktree, run `scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/dev/review_worktree_guard.py configure
--worktree <path> --mode implementation`; the helper restores the worktree-local hook and push
configuration captured when review mode was enabled. Re-fetch `origin/main` before creating a new
review worktree so its source ref is explicit and current.

## Delegated-worker isolation receipt

Repository-owned delegated workers should opt into an immutable, credential-free receipt and bind
their first command to the new worktree:

```bash
scripts/dev/create_worktree.sh \
  --branch issue-123-short-description \
  --path "$WORKTREE_PARENT/issue-123-short-description" \
  --base origin/main \
  --receipt "/path/to/private/issue-123.receipt.json" \
  --task-id issue-123 \
  --exec <worker-command>
```

Creation writes the receipt atomically after the linked worktree exists. The `--exec` command is
guarded before it starts; the read-only guard exits nonzero with one JSON result when the current
working directory, top-level, shared Git directory, branch/ref, or base ancestry differs. Workers
started separately must run the equivalent check from inside the assigned worktree with
`scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/dev/worktree_receipt.py check`.
Ordinary callers retain the existing behavior when receipt options are omitted.

Bootstrap symlinks the local machine context and creates a worktree-local `.venv`. Do not run a
bare `uv run ...` first in a fresh worktree: it can materialize a partial local environment, which
then shadows the shared environment selected by later commands. For a cheap targeted check, route
the command through the shared environment wrapper instead:

```bash
scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/dev/check_worktree_optional_deps.py --profile all-extras
```

If a worktree-local environment is intentional, create and sync it with
`scripts/dev/bootstrap_worktree.sh` before using it. If a bare invocation has already created an
accidental partial `.venv`, stop using that environment and follow the bootstrap/wrapper path after
confirming it contains no worktree-local state that needs preserving.

If the shared wrapper reports stale `fast-pysf`, use its explicit linked-worktree recovery option:
`scripts/dev/run_worktree_shared_venv.sh --recover-stale-fast-pysf -- <command>`. It creates or
refreshes only the current worktree's `.venv`, applies the capacity and repository recovery-lock
gates, rejects nested environment links that could redirect package writes outside the worktree, and
never repairs the main checkout implicitly. See the [local CI recovery contract](local_ci.md#recover-stale-fast-pysf-explicitly).

Never edit `.venv` by hand; manage dependencies through `pyproject.toml` and `uv sync`. Never use a bare
`git stash pop` in a linked worktree because all worktrees share one stash namespace. Prefer a
temporary commit or `scripts/dev/safe_stash_pop.sh`.

## Preserve and retire

Before retirement, inspect the exact worktree and enumerate ignored outputs:

```bash
git worktree list --porcelain
git -C "$WORKTREE_PATH" status --short --branch
uv run python scripts/dev/worktree_hygiene_snapshot.py \
  --repo-status --retirement-plan --json
```

Preserve tracked changes, unpushed commits, and ignored-but-important evidence before removal.
Classify `output/` as temporary scratch, durable evidence, or handoff-needed; worktree-local output
is not durable storage. Remove only a clean, no-longer-needed worktree:

```bash
git worktree remove "$WORKTREE_PATH"
git worktree prune
```

Do not remove a dirty worktree, an unpushed branch, or a durable artifact without an explicit
preservation record.
