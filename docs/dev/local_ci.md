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

# Final PR proof when the change crosses the escalation boundary
BASE_REF=origin/main PR_READY_MODE=final scripts/dev/pr_ready_check.sh
```

Use the core lane by default. Opt into `ROBOT_SF_TEST_LANE=optional` only when optional paths are
part of the change. Do not treat fallback or degraded execution as benchmark success evidence.

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
