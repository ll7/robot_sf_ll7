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
python scripts/dev/review_worktree_guard.py integrate \
  --worktree "$WORKTREE_PARENT/review-pr-123" \
  --source-ref origin/main \
  --remote origin
```

The creator writes the worktree-local `robot-sf.worktree-mode=review` marker and installs the
tracked pre-push guard. Configured remote names also receive inert worktree-local push destinations
and a nonexistent worktree-local receive-pack command. An all-URL worktree-local rewrite (plus
exact push-URL rules) routes remote URLs to an inert path, covering inherited `pushurl` values,
equivalent local-path spellings, explicit destination refspecs, and `--no-verify`. Review mode
therefore intentionally blocks direct fetch and `ls-remote` commands too; refresh refs before
entering the mode or use the integration helper, which reads its comparison through the common Git
config. No configured remote can be mutated from the protected worktree through ordinary Git
invocation paths. This is a Git-level workflow guard, not an operating-system sandbox; a deliberate
per-command Git configuration override can bypass it.
The integration helper snapshots every ref from `git ls-remote --refs`, runs
`git merge --no-commit --no-ff`, always attempts `git merge --abort`, and exits nonzero unless the
worktree is clean and the before/after remote snapshots are identical.

Ordinary implementation worktrees retain the default pushable behavior. To deliberately restore a
previously protected worktree, run `python scripts/dev/review_worktree_guard.py configure
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
`python scripts/dev/worktree_receipt.py check`.
Ordinary callers retain the existing behavior when receipt options are omitted.

Bootstrap symlinks the local machine context and creates a worktree-local `.venv`. For a cheap
targeted check, use the shared environment wrapper instead:

```bash
scripts/dev/run_worktree_shared_venv.sh -- \
  python scripts/dev/check_worktree_optional_deps.py --profile all-extras
```

Never edit `.venv`; manage dependencies through `pyproject.toml` and `uv sync`. Never use a bare
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
