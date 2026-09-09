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

## Recover stale fast-pysf explicitly

If the shared wrapper reports a stale installed `fast-pysf` package, recover from the linked
worktree by rerunning the same command with the explicit recovery option:

```bash
scripts/dev/run_worktree_shared_venv.sh --recover-stale-fast-pysf -- \
  uv run pytest tests/test_ci_script_contract.py -q
```

This route creates or refreshes only the current linked worktree's ignored `.venv`. It refuses the
main checkout and dirty dependency inputs, checks `ROBOT_SF_WORKTREE_MIN_FREE_BYTES` (2 GiB by
default) with `check_worktree_capacity.py`, serializes recovery per repository with a kernel-backed
lock, and verifies `fast-pysf` before starting the command. It runs the frozen, auditable operation
`uv sync --all-extras --reinstall-package robot-sf --frozen`; an existing coherent local environment
skips the sync. Environment ownership checks reject nested links that would redirect package writes
outside the worktree, while allowing valid standard `bin/python*` links to the host interpreter and
rejecting broken aliases or links into the owning checkout. The recursive scan also fails closed if
any environment subtree cannot be inspected. Capacity or lock contention fails closed without
starting the wrapped command.

Do not combine recovery with `--venv`, `--standalone`, or a freshness bypass. Repair an explicitly
owned environment manually with `uv sync --all-extras --reinstall-package robot-sf` in that
checkout only when that ownership is intentional. The standalone helper and its ownership boundary
are documented in `scripts/dev/recover_fast_pysf_worktree.sh`.

## Validate with an isolated Ruff version

Use the active checkout's exact Ruff version without changing a shared or local project
environment. This is useful when the owning checkout intentionally pins an older version:
re-syncing that owner would reinstall its older pin, not satisfy the newer worktree.

```bash
scripts/dev/run_worktree_shared_venv.sh --isolated-ruff -- \
  ruff check tests/test_ci_script_contract.py
scripts/dev/run_worktree_shared_venv.sh --isolated-ruff -- \
  ruff format --check tests/test_ci_script_contract.py
```

The opt-in mode requires host Python 3.11+ and uv. Host Python runs with `-I -S -B`, without
project or site imports, and requires one exact development dependency pin matching
`tool.ruff.required-version`. It verifies the provisioned Ruff version before validation. The
default shared-environment freshness refusal remains unchanged; isolated success is focused proof,
not final readiness or permission to bypass a stale environment.

Only bare `ruff check` and `ruff format --check` are supported. File paths, spaces, a `--` filename
separator, and the caller's working directory are preserved. Safe selectors include `--select`,
`--extend-select`, `--ignore`, `--extend-ignore`, per-file ignores, exclusions, target version,
line length, and output format. Unknown options, stdin, mutation flags, configuration overrides,
cache/output-file redirects, and combinations with other wrapper options fail closed.

This route explicitly uses the active root's `pyproject.toml`; **nested and user Ruff configurations
are ignored**. Root configuration cannot enable `fix` or `fix-only`, extend another configuration,
or redirect cache/output. Ruff's cache and fixing are additionally disabled at the command line.

Provisioning may access the network to download the exact pinned Ruff into a fresh task-owned
directory under the resolved system `/tmp`. No shared/user cache is reused or deleted. Both uv's
cache and temporary state stay there, outside the owning/worktree checkout and project environments;
symlink-resolved cache/temp aliases into those locations are rejected. Inherited uv, Python, Ruff,
and XDG redirects are cleared for the child. `UV_OFFLINE=1` is the sole inherited uv control; with
the deliberately empty isolated cache, an unavailable package fails closed rather than falling
back to another tool. The wrapper removes its own temporary directory on success, failure, and
catchable `INT`, `TERM`, or `HUP`, terminating the child process group and allowing at most two
seconds before forced termination. As with other processes, `SIGKILL` or host loss cannot execute
cleanup. Ruff's normal validation exit code is preserved; admission/provisioning failure exits 2,
and caught interruption returns the conventional `128 + signal` status.

For full PR proof, prepare an intentionally local environment through the existing bootstrap route
and run final readiness. This mode does not sync dependencies or replace the readiness formatter.

## Early evidence-registry check in final readiness

Final readiness checks the evidence registry before formatting or starting any test lane when the
committed base-to-head changes touch a hosted evidence-registry input. This includes
`docs/context/evidence/`, release and citation metadata, and the checker, linter, baseline,
review policy, focused tests, and workflow paths listed in
[the hosted workflow](../../.github/workflows/evidence-registry-ratchet.yml).
Deletions and both sides of renames count; whitespace or newlines in a filename do not change
the path boundary. Unrelated changes skip this early check.

The entry point runs the canonical check exactly once for that relevant scope. In final mode it
binds the check to the exact committed head and resolved base, for example:

```bash
candidate_head="$(git rev-parse --verify HEAD^{commit})"
frozen_base="$(git rev-parse --verify origin/main^{commit})"
uv run python scripts/dev/evidence_registry_ratchet.py --check \
  --candidate-head "$candidate_head" --frozen-base "$frozen_base" \
  --report-output /tmp/evidence-registry-ratchet-report.json
```

The projection reads the complete candidate evidence tree and uses only the frozen base for
producer reachability. Its report records the candidate/base/tree identities, exact receipt and
producer bindings, and projected per-path/per-code deltas. This is read-only: it neither repairs
the registry nor refreshes its baseline. A checker failure preserves its diagnostic and exit
status, including statuses 1 and 2, and prevents formatting, tests, and a success stamp. Missing,
stale, shallow, or incomplete identities/history fail closed. An explicitly requested final-mode
`BASE_REF` that remains unresolved after the existing best-effort fetch cannot fall back to `HEAD`.

Hosted pull-request and merge-group runs use the same explicit projection. The direct/native merge
gate consumers only consume the named exact-head check status; they do not parse the projection a
second time. Because GraphQL status rollups omit check-run head identities, source-PR snapshots
rebind a present evidence check through the exact REST check-run endpoint before admission consumes
it. The current source-PR gate has no complete changed-file applicability proof for the
path-filtered workflow, so an absent source-PR evidence check remains an explicit boundary rather
than a claimed universal block. The native merge-group workflow supplies the synthetic exact head
and frozen base to the canonical evidence job and evaluates the resulting proof against that
synthetic head, while retaining the source PR head as a separate queue-ref identity. The
single-account receipt inherits this gate audit.

`PR_READY_SKIP_PREFLIGHT=1` does not disable this integrity check. Interim mode retains its existing
behavior, including base fallback, and does not run the new early check. A successful check is
only an early rejection filter: all later readiness gates, core registry invariants, hosted
checks, and final freshness requirements still apply. No success is cached between runs.

## Readiness count selectors

When reporting readiness counts, name the exact selector so another contributor can reproduce the
same scope. The readiness contract selector is:

```bash
uv run pytest -q tests/test_ci_script_contract.py
```

The combined readiness reliability selector is:

```bash
uv run pytest -q \
  tests/dev/test_pr_ready_preflight.py \
  tests/dev/test_pr_ready_termination.py \
  tests/test_ci_script_contract.py
```

Use `--collect-only -q` with either selector to inspect its collected-test count without running
the tests. Do not label a partial or historical count as the readiness suite without its selector.

## Proportional checks

```bash
# Docs and links
uv run python scripts/dev/check_docs_evidence_integrity.py --full
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

## Final PR proof when the change crosses the escalation boundary

```bash
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
Group-cleanup verification requires `child_registration_state: registered`, both registered child
identifiers, and a negative process-group existence probe. On foreground-only hosts, a registered
direct child may still report direct-process cleanup without a process-group identifier; that status
does not claim descendant cleanup. Unknown or contradictory registration state remains unverified.

On a host where the shared NVIDIA CUDA (Compute Unified Device Architecture) probe reports a usable
graphics processing unit (GPU), the optional and `all` lanes default to one in-process worker
because some optional subprocess tests share GPU memory. This is a local readiness safety policy,
not benchmark evidence. Central processing unit (CPU)-only hosts retain the automatic xdist default,
and an explicit `PYTEST_NUM_WORKERS=<int>|auto` override remains visible and takes precedence over
the CUDA serial policy (subject to the existing platform and low-CPU caps). The readiness output
records the CUDA status, selected lane, worker count, and override or default reason. An uncertain
CUDA probe uses the serial safe default; an unavailable or unusable runtime keeps the CPU parallel
path and CUDA-gated tests use their explicit unavailable receipt.

## Parallel timeout diagnostics

The parallel pytest wrapper captures every non-zero run and invokes
`scripts/dev/diagnose_xdist_crash.py` with the selected worker count, distribution mode, execution
mode, and pytest exit code. The reporter classifies pytest-timeout, subprocess, and silent
process-timeout signatures and includes a bounded runtime fingerprint. A timeout remains an
incomplete, fail-closed readiness result; an opt-in serial rerun can classify load-sensitive
behavior but cannot promote the parallel lane to success evidence. Record the exact timed-out
tests, worker count, runtime/dependency versions, cache location, and process-cleanup result when
triaging parallel-load friction (issue #8469).

The high-concurrency `run_xdist_race_validation.sh` route also preserves the compact-validation
summary. It invokes the reporter only when that summary's `timed_out` field is true, rather than
assuming that every exit code 124 came from the outer process boundary. The route reads the actual
pytest execution-mode marker from the captured log, so true in-process serial execution is not
reported as parallel xdist. If the summary, log path, or unique execution-mode marker is missing,
the route reports the diagnostic as unavailable and remains failed; it never turns an incomplete
or ambiguous timeout into a pass.

## Local CI-equivalent path

Use `scripts/dev/run_ci_local.sh` when the repository's complete local CI contract is required. Run
it from a clean linked worktree after the dependency profile is ready. The helper may publish
advisory local statuses, but final readiness still requires a clean tree and exact `origin/main`
base.

For bounded polling of hosted PR checks, use:

```bash
uv run python scripts/dev/check_pr_ci_status.py <pr-number> \
  --poll-attempts 20 --poll-interval 30 --max-wall-seconds 600
```

The local wall cap also bounds nested `gh` reads. If a read reaches the cap, the monitor emits a
machine-readable fail-closed error and stops; it never cancels a remote GitHub check. On POSIX
hosts, timed-out `gh` process groups are terminated together with their local descendants.

Record command, base/head SHA, profile, and whether the result was native, adapter, fallback, or
degraded. Read [`docs/maintainer_values.md`](../maintainer_values.md) and
[`docs/code_review.md`](../code_review.md) before making benchmark, metric, provenance, or
paper-facing claims.
