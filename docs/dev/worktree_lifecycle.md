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
  --base origin/main \
  --task-id issue-123
cd "$WORKTREE_PARENT/issue-123-short-description"
scripts/dev/bootstrap_worktree.sh
```

The capacity-guarded helper must create the worktree before editing, running PR validation,
pushing, or publishing. Active task-owned worktrees should pass `--task-id`; the creator writes a
path-scoped ownership lease before it releases the repository worktree mutation lock. This makes
creation and repository-owned cleanup atomic with respect to task ownership. Short-lived human
scratch worktrees may omit the task lease when no autonomous worker can race their teardown.

Use `--exec` when the first command must be bound to the new directory:

```bash
scripts/dev/create_worktree.sh \
  --branch issue-123-short-description \
  --path "$WORKTREE_PARENT/issue-123-short-description" \
  --base origin/main \
  --task-id issue-123 \
  --exec git rev-parse --show-toplevel
```

## Cleanup safety invariant

Repository-owned automatic cleanup is fail-closed: a linked worktree is not removable when its
tracked or untracked state is dirty, ignored content is present, an open pull request (PR) covers
its branch, push state cannot be proven safe, or an active task/gate lease exists. Use the reaper's
JSON output as the decision record; refused candidates include `reason_codes`, the path/branch/HEAD
identity, lease owner fields when available, and a `recovery` object that requires preservation and
review before reclaim.

```bash
uv run python scripts/dev/stale_worktree_reaper.py \
  --path "$WORKTREE_PATH" \
  --apply \
  --json
```

The apply path repeats every preservation read while holding the shared worktree lifecycle lock.
This closes the gap between a dry-run plan and removal: a new dirty file, ignored output, open PR,
unpushed state, unreadable check, or lease acquired after planning becomes a structured refusal and
the worktree remains present. `--skip-pr-check` is therefore also a refusal signal, not permission
to delete offline.

The incident's separate external/app cleanup caller has not been located in this repository. This
change does not claim to repair or intercept that caller. Until it is identified and routed through
this contract, direct cleanup outside the reaper is unsupported and must be treated as unsafe; the
repository can only guarantee the behavior of its supported reaper and lifecycle helpers.

For `--mode review`, the creator launches that optional first command through the Linux Landlock
process boundary. Keep later review commands as descendants of it, or invoke the guard's `run -- ...`
form explicitly; the mode marker and Git hook do not attach an OS policy to future raw processes.

New branches are created without automatic upstream tracking. This avoids concurrent workers
contending on the shared repository configuration while they create linked worktrees. Configure a
remote explicitly when publishing a branch, for example with `git push -u origin <branch>`.

The creation helper clears any `config.worktree` copied from the invoking checkout before applying
the requested mode. Review-only push barriers therefore cannot leak into a newly created
implementation worktree, and implementation worktrees do not inherit arbitrary per-worktree
settings from a protected review checkout.

## Protected review worktrees

Review and synthetic-integration worktrees must opt into the protected mode explicitly:

```bash
scripts/dev/create_worktree.sh \
  --branch review/pr-123 \
  --path "$WORKTREE_PARENT/review-pr-123" \
  --base origin/main \
  --mode review \
  --task-id review-pr-123
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/dev/review_worktree_guard.py integrate \
  --worktree "$WORKTREE_PARENT/review-pr-123" \
  --source-ref origin/main \
  --remote origin
```

The creator writes the worktree-local `robot-sf.worktree-mode=review` marker and installs the
tracked pre-push guard. Configured remote names also receive inert worktree-local push destinations
and a nonexistent worktree-local receive-pack command. Push-specific URL rewrites (plus exact
configured-URL rules) route push URLs to an inert path, covering direct pushes, equivalent local-path
spellings, explicit destination refspecs, and `--no-verify` for configured remotes. Push rejection is
separated from read-only URL resolution: direct `git fetch` and `git ls-remote` commands remain
operational for inspection and ref verification. A remote added after activation with an explicit
`remote.<name>.pushurl` is still protected on the ordinary hook path, but Git does not apply
`pushInsteadOf` to that explicit value when `--no-verify` bypasses the hook. That deliberate
configuration/command-line override belongs to the stronger process boundary described below. This
is a Git-level workflow guard, not an operating-system sandbox; a deliberate per-command Git
configuration override can bypass it.

Review setup never masks `url.*.pushInsteadOf` entries by editing the shared Git config. If an
effective repository, global, system, or pre-existing worktree `url.*.pushInsteadOf` alias could
outrank the worktree barrier, setup fails closed before enabling review mode; remove or relocate
the alias and retry. This refusal is required because a guard-specific lock would not serialize
arbitrary Git processes in other linked worktrees. Generic `url.*.insteadOf` entries remain intact
for read URL resolution.

The stronger adversarial process boundary is explicit and must wrap any command that may reach a
local remote or invoke an alternate receive-pack:

```bash
python "$MAIN_REPO_ROOT/scripts/dev/review_worktree_guard.py" run \
  --worktree "$WORKTREE_PARENT/review-pr-123" -- \
  git -c url.<actual-file-url>.insteadOf=<blocked-file-url> push \
  --no-verify --receive-pack=git-receive-pack origin HEAD:refs/heads/example
```

On Linux with Landlock application binary interface (ABI) 4 or newer, `run` is a real
descendant-inherited operating-system boundary: it allows reads and execution throughout the host,
permits filesystem mutation only below the review worktree and its linked Git admin directory,
closes inherited file descriptors,
and denies TCP bind/connect. It therefore protects a temporary bare remote outside those writable
roots even when Git URL configuration, `--receive-pack`, and `--no-verify` are supplied on the
command line. Direct alternate receive-pack descendants inherit the same policy. The command's
output and exit status are preserved.

`run` fails closed on non-Linux hosts, unsupported architectures, kernels without the required
Landlock ABI, unavailable `/proc/self/fd` inspection, or any policy-installation error. Landlock
is Linux-specific, so this is not a portable all-host guarantee. The process boundary is not
attached to a directory: a new terminal or a raw `/usr/bin/git` invocation launched outside `run`
is outside the contract. Use `run -- ... bash` for a bounded multi-command session. The strict
boundary denies TCP, so stage network refs before entering it; Unix-domain/existing privileged
helper channels, remotes placed inside the two writable roots, and privileged host escape remain
outside this bounded threat model. This paragraph describes the OS policy; the Git configuration
above remains only a defense-in-depth workflow guard.

For a local bare-remote or otherwise network-free synthetic integration probe, the integration
helper can itself be made a descendant of the boundary:

```bash
python "$MAIN_REPO_ROOT/scripts/dev/review_worktree_guard.py" run \
  --worktree "$WORKTREE_PARENT/review-pr-123" -- \
  python "$MAIN_REPO_ROOT/scripts/dev/review_worktree_guard.py" integrate \
  --worktree "$WORKTREE_PARENT/review-pr-123" --source-ref origin/main --remote origin
```

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

## Active-worktree lease

`--task-id` is the owner-visible lifecycle marker for an autonomous task. The lease lives in the
shared Git common directory, not inside the linked worktree, so it remains inspectable if the
worktree directory disappears. The compatibility schema is still `pr_gate_lease.v1`; for ordinary
task-owned worktrees the creator records the task id in the existing gate/owner fields instead of
introducing a second lease format.

The default lease lifetime is two hours. Long-running controllers should heartbeat at phase or
handoff boundaries before the current lease expires:

```bash
uv run python scripts/dev/pr_gate_lease.py heartbeat \
  --worktree "$WORKTREE_PATH" \
  --extend-hours 2
```

Run heartbeat/release commands from any surviving checkout of the same repository. Do not depend on
the leased worktree as the command's current directory: the recovery path exists specifically for a
missing worktree directory. When the task has completed or emitted a durable handoff, release its
lease explicitly:

```bash
uv run python scripts/dev/pr_gate_lease.py release --worktree "$WORKTREE_PATH"
```

The script is also directly executable (`scripts/dev/pr_gate_lease.py release --worktree "$WORKTREE_PATH"`)
and accepts `--worktree` either before or after the subcommand (`scripts/dev/pr_gate_lease.py --worktree "$WORKTREE_PATH" release`).

Lease creation, heartbeat, release, and guarded cleanup all serialize on the existing
`robot-sf-create-worktree.lock` identity. Cleanup therefore cannot observe "no lease" and then
remove the worktree while a task concurrently claims it. Expired leases are deliberately non-live:
a crashed owner that never releases eventually stops blocking safe cleanup after its TTL.
Malformed or unreadable lease state fails closed.

## Missing-path recovery

If an active task reports that its worktree path disappeared while its branch ref survived, do not
continue commands from the deleted current working directory. Move to a surviving checkout and
inspect the leased path:

```bash
uv run python scripts/dev/gate_worktree_guard.py verify \
  --path "$WORKTREE_PATH" \
  --json
```

The JSON health result is the compact recovery handoff: for a live lease it names the owner/task,
PR/gate identifiers when present, expiry, path, branch, and HEAD. A missing result always includes
a recovery loss boundary: branch recreation can restore the checkout only, while dirty, untracked,
and ignored local state cannot be recovered by this guard. To recreate the checkout from the
persisted branch/commit metadata, use:

```bash
uv run python scripts/dev/gate_worktree_guard.py ensure \
  --path "$WORKTREE_PATH" \
  --json
```

The `recovery.loss_boundary` value is `dirty_untracked_ignored_state_not_recoverable`, and
`local_state_restored` remains false even when recreation succeeds. Recover that state from a
durable handoff or backup before continuing. The guard never chooses a replacement branch or
transport policy.

## Delegated-worker isolation receipt

Repository-owned delegated workers can additionally opt into an immutable, credential-free receipt
and bind their first command to the new worktree:

```bash
scripts/dev/create_worktree.sh \
  --branch issue-123-short-description \
  --path "$WORKTREE_PARENT/issue-123-short-description" \
  --base origin/main \
  --receipt "/path/to/private/issue-123.receipt.json" \
  --task-id issue-123 \
  --exec <worker-command>
```

`--task-id` creates the lifecycle lease regardless of whether `--receipt` is supplied. When a
receipt is requested, creation writes it atomically after the linked worktree exists. The `--exec`
command is guarded before it starts; the read-only receipt check exits nonzero with one JSON result
when the current working directory, top-level, shared Git directory, branch/ref, or base ancestry
differs. Workers started separately must run the equivalent check from inside the assigned worktree
with `scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/dev/worktree_receipt.py check`.
The receipt proves assignment identity; the lease protects the active path from repository-owned
cleanup. They are deliberately separate contracts.

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
rejects broken or owning-checkout `bin/python*` aliases, and fails closed if a nested environment
subtree cannot be inspected. It never repairs the main checkout implicitly. See the [local CI
recovery contract](local_ci.md#recover-stale-fast-pysf-explicitly).

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
is not durable storage. Once the owning task has completed, release its lease and use the targeted
reaper path rather than a bare `git worktree remove`:

```bash
uv run python scripts/dev/pr_gate_lease.py release --worktree "$WORKTREE_PATH"
uv run python scripts/dev/stale_worktree_reaper.py \
  --path "$WORKTREE_PATH" \
  --apply \
  --json
git worktree prune
```

The targeted reaper retains the existing dirty/unpushed/open-PR/ignored-output checks. Immediately
before removal it reacquires the shared lifecycle lock and re-reads the lease. If another task
claimed the worktree after planning, cleanup refuses it and the audit log names the live task/owner.
An unreadable lease also refuses removal. An expired lease does not block cleanup, providing the
bounded stale-owner recovery path.

Do not remove a dirty worktree, an unpushed branch, a live-leased worktree, or a durable artifact
without an explicit preservation record.

## Verified merged-tree retirement

When a completed pull request (PR) has had its remote branch deleted, an explicitly supplied
proof bundle can authorize retirement of that one clean worktree without recreating the remote
reference. The default reaper remains conservative: a missing upstream is still refused unless
all verified inputs are supplied. GitHub's pull-request REST metadata does not provide an
authoritative historical merge-method field, so this mode deliberately proves exact merged-tree
identity and ancestry without claiming that the PR used squash, rebase, or regular merge.

Run the verified mode from the registered `main` worktree and name the target's canonical,
non-symlink path, surviving local branch, full head commit, and merged PR:

```bash
uv run python scripts/dev/stale_worktree_reaper.py \
  --path "$WORKTREE_PATH" \
  --verified-merged-pr 8658 \
  --verified-branch feature/example \
  --verified-head-sha <40-character-head-sha> \
  --json
```

The read-only proof requires the target path to be an exact registered non-current worktree with a
surviving local branch. That branch must retain `branch.<name>.remote=origin` and the matching
`branch.<name>.merge` configuration while its `refs/remotes/origin/<name>` object is absent;
never-published branches and branches with a still-resolvable upstream remain refused. Fresh
authoritative PR metadata must identify this repository, the exact branch and head, a merged PR
to `main`, and a full merge commit that resolves locally. The checked-out `main`, local
`origin/main`, authoritative `origin/main` from a fresh `git ls-remote --heads` read, target head,
PR merge commit, and all three complete Git tree IDs must agree as required by the plan, and the
merge commit must be an ancestor of current `main`. The same authoritative read must show the
candidate branch absent; a stale local remote-tracking ref is not treated as proof of deletion.
The resulting
verification evidence records `merge_method_scope=method_agnostic_exact_tree`; it is a content
and ancestry proof, not a merge-method classifier.

The command performs only bounded GitHub reads (`gh pr list` and `gh api`) and a read-only
authoritative `git ls-remote`; it never posts, edits, fetches, mutates local refs, recreates refs,
changes configuration, releases leases, or removes another worktree. The
JSON candidate's `verification` object and audit log expose the exact repository, PR, branch,
head, merge, main, tracking-ref, and tree identities plus the narrowly discharged
`missing origin upstream only` risk. Existing dirty/untracked/ignored-content, open-PR,
current-worktree, lease, and lifecycle-lock gates still apply. A verified candidate is safe to
remove only through the canonical apply path:

```bash
uv run python scripts/dev/stale_worktree_reaper.py \
  --path "$WORKTREE_PATH" \
  --verified-merged-pr 8658 \
  --verified-branch feature/example \
  --verified-head-sha <40-character-head-sha> \
  --apply --json
```

Immediately before the normal, non-force `git worktree remove`, the reaper reacquires the shared
lifecycle lock and repeats registration, path, branch, head, ref, tree, main, authoritative remote,
PR, cleanliness, ignored-state, open-PR, upstream-absence, and lease reads. It then performs a
final registration/path, branch, HEAD/ref-identity, cleanliness, and ignored-state preservation
recheck after the last remote/PR read and immediately before removal, closing the remaining window
in which an ignored artifact or target identity could drift during verification. Any deletion,
symlink/path alias, identity drift, new content, lease, lookup error, or lock failure refuses
removal. Normal
worktree removal preserves the local branch and its commits; artifact preservation remains the
owner's responsibility before retirement.
