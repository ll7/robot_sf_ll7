---
name: clean-up
description: Clean up the current branch in the Robot SF repo by following docs/dev_guide.md and reusable
  scripts/dev commands; use when asked to tidy a branch, run Ruff format/fix, or run parallel pytest before
  sharing changes.
category: validation
kind: atomic
phase: verification
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# Clean Up

## Purpose

Run the repo-standard cleanup pipeline for the current branch: formatting, automated checks,
parallel tests, and diff-based quality gates.

## Workflow

1. Confirm context:
   - Check for repo markers (`.git`, `docs/dev_guide.md`, `.specify/memory/constitution.md`).
   - Read `docs/dev_guide.md`, `.specify/memory/constitution.md`, and `.github/copilot-instructions.md`.
2. Prepare environment:
   - If `.venv` is present, use it.
   - If missing, run `uv sync --all-extras`, `source .venv/bin/activate`, and `uv run pre-commit install`.
3. Format and lint:
   - `scripts/dev/ruff_fix_format.sh`
   - Re-run until the script is clean.
4. Run tests:
   - `scripts/dev/run_tests_parallel.sh`
   - Optional: `--new-first` or `--no-fast-fail` as needed.
5. Run diff quality gates:
   - `BASE_REF=origin/main scripts/dev/check_changed_coverage.sh`
   - `BASE_REF=origin/main scripts/dev/check_docstring_todos_diff.sh`
6. Report:
   - Summarize passed/failed commands and any residual risk blocks (flaky or deferred tests).

## Guardrails

- Fix failures with intent; do not silence value-heavy tests without explicit direction.
- If touched tests were added, verify those new tests locally.
- For rendering or benchmark-affecting work, keep GUI/benchmarks in follow-up if required and noted.

## Output

- Commands executed and pass/fail status.
- List of remaining failures and the decision (fix, defer, investigate).
- Remaining follow-up tasks before merge-ready handoff.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
