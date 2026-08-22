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
