# Local CI and PR Readiness

This is the canonical task guide for dependency-aware local validation. Match the proof to the
change risk; use the full readiness lane when the change affects scripts, runtime, schemas,
benchmark semantics, provenance, or publication behavior.

## Dependency profiles

`run_tests_parallel.sh` checks dependencies before resolving workers or starting pytest:

- `core` lane → `core` dependency profile;
- optional and `all` lanes → `all-extras` profile.

An incomplete current-worktree environment fails closed with the missing imports and an actionable
`uv sync --all-extras` repair command. It is setup evidence, not a changed-code failure. Use the
shared wrapper for a fresh worktree when only a focused check is needed:

```bash
scripts/dev/run_worktree_shared_venv.sh -- \
  uv run pytest tests/test_ci_script_contract.py -q
```

## Proportional checks

```bash
# Docs and links
python scripts/dev/check_docs_evidence_integrity.py --full
bash scripts/dev/check_context_notes.sh

# Focused workflow/runtime proof
uv run pytest <focused-test> -q
uv run ruff check <changed-files>
uv run ruff format --check <changed-files>

# Hermetic Git-identity lane (Git-backed tests only)
scripts/dev/run_hermetic_git_tests.sh
```

## Hermetic Git-identity lane

Git-backed tests that create commits or commit trees must not depend on ambient
developer or CI-runner Git identity/configuration. To reproduce a clean-runner
Git failure locally, run the hermetic lane:

```bash
scripts/dev/run_hermetic_git_tests.sh
```

The wrapper unsets `GIT_AUTHOR_NAME`/`GIT_AUTHOR_EMAIL`/
`GIT_COMMITTER_NAME`/`GIT_COMMITTER_EMAIL`, points `GIT_CONFIG_GLOBAL` at
`/dev/null`, and sets `GIT_CONFIG_NOSYSTEM=1` before running the Git-backed test
modules under `tests/dev/`, `tests/tools/`, `tests/validation/`,
`tests/unit/`, and `tests/integration/`.

Temporary Git fixtures configure their own deterministic identity via the
shared helpers in `tests/support/environment_guards.py`:

- `git_identity_environment()` returns a hermetic env dict for subprocess calls
  (sets author/committer identity and disables global/system config).
- `configure_git_identity(repo)` runs repository-local `git config` for
  `user.name`/`user.email`.

A fixture that omits both fails closed in the lane with git's
"Author identity unknown" error instead of silently passing on a developer
machine.

# Final PR proof when the change crosses the escalation boundary
BASE_REF=origin/main PR_READY_MODE=final scripts/dev/pr_ready_check.sh
```

Use the core lane by default. Opt into `ROBOT_SF_TEST_LANE=optional` only when optional paths are
part of the change. Do not treat fallback or degraded execution as benchmark success evidence.

## Worktree-scoped readiness lock

The local readiness entry point prevents duplicate expensive runs in one linked worktree while
allowing readiness to run concurrently in independent worktrees. It derives the lock identity from
the canonical absolute worktree path, not the shared Git directory or the process's `TMPDIR`.

If another run is active, the command exits without waiting or terminating that process and prints
the active worktree plus a safe retry command. Wait for the active run to finish, then rerun the
same command. Lock anchors may remain under the host-local lock root after exit; their file
presence is not used as ownership, so an interrupted run does not create a stale held lock. The
default root is `/tmp/robot-sf-pr-ready-locks`; deterministic test harnesses may set
`PR_READY_LOCK_DIR` to an isolated absolute directory.

The lock uses the host Python implementation's kernel-backed `fcntl` primitive on supported Unix
hosts. If that primitive cannot be initialized, readiness fails closed instead of running without
the worktree lock.

When readiness is terminated, it writes a private bounded receipt to
`output/validation/pr_ready/` before returning the conventional signal status. The receipt records
the active phase and lane, last progress, process-group cleanup verification, and a small host/cgroup
resource snapshot; it deliberately omits command lines and the environment. Set
`PR_READY_TERMINATION_RECEIPT` to choose an absolute or worktree-relative output path.

On a host where the shared NVIDIA CUDA (Compute Unified Device Architecture) probe reports a usable
graphics processing unit (GPU), the optional and `all` lanes default to one in-process worker
because some optional subprocess tests share GPU memory. This is a local readiness safety policy,
not benchmark evidence. Central processing unit (CPU)-only hosts retain the automatic xdist default,
and an explicit `PYTEST_NUM_WORKERS=<int>|auto` override remains visible and takes precedence over
the CUDA serial policy (subject to the existing platform and low-CPU caps). The readiness output
records the CUDA status, selected lane, worker count, and override or default reason. An uncertain
CUDA probe uses the serial safe default; an unavailable or unusable runtime keeps the CPU parallel
path and CUDA-gated tests use their explicit unavailable receipt.

## Local CI-equivalent path

Use `scripts/dev/run_ci_local.sh` when the repository's complete local CI contract is required. Run
it from a clean linked worktree after the dependency profile is ready. The helper may publish
advisory local statuses, but final readiness still requires a clean tree and exact `origin/main`
base.

For bounded polling of hosted PR checks, use:

```bash
uv run python scripts/dev/check_pr_ci_status.py <pr-number> \
  --poll-attempts 20 --poll-interval 30
```

Record command, base/head SHA, profile, and whether the result was native, adapter, fallback, or
degraded. Read [`docs/maintainer_values.md`](../maintainer_values.md) and
[`docs/code_review.md`](../code_review.md) before making benchmark, metric, provenance, or
paper-facing claims.
