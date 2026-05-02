---
name: clean-up
description: "Clean up the current branch in the Robot SF repo by following docs/dev_guide.md and reusable scripts/dev commands; use when asked to tidy a branch, run Ruff format/fix, or run parallel pytest before sharing changes."
---

# Clean Up

## Overview

Clean and validate the current branch for the Robot SF repo by applying the dev guide
workflow, running Ruff format/fix, and running parallel tests.

## Cleanup Workflow

1. Confirm repo and guidance
   - Verify the working directory looks like the Robot SF repo (e.g., `.git`,
     `docs/dev_guide.md`, `.specify/memory/constitution.md`, and
     `.github/copilot-instructions.md` exist).
   - Read `docs/dev_guide.md`, `.specify/memory/constitution.md`, and
     `.github/copilot-instructions.md` to align with required rules.

2. Ensure environment
   - If `VIRTUAL_ENV` is empty and `.venv/bin/activate` exists, run
     `source .venv/bin/activate`.
   - If `.venv/bin/activate` is missing, follow dev guide setup:
     `uv sync --all-extras`, `source .venv/bin/activate`, and
     `uv run pre-commit install`.

3. Run formatting and fixes first
   - Use the shared script:
     `scripts/dev/ruff_fix_format.sh`
   - If Ruff reports issues, fix them and rerun until clean.

4. Run tests in parallel
   - Use the shared script:
     `scripts/dev/run_tests_parallel.sh`
   - Default behavior is fail-fast + failed-first ordering:
     `pytest -n auto -x --failed-first`
   - Optional ordering toggle:
     `scripts/dev/run_tests_parallel.sh --new-first`
   - To disable fail-fast when you need a full failure set:
     `scripts/dev/run_tests_parallel.sh --no-fast-fail`
   - If tests fail, evaluate test value first (Constitution Principle XIII /
     dev guide testing strategy). Classify failures and decide whether to fix,
     defer, or ask for direction before removing or relaxing tests.

5. Run diff-based quality gates and fix them.
   - Run changed-files coverage check:
     `BASE_REF=origin/main scripts/dev/check_changed_coverage.sh`
     - If you changed any test files, run these new tests locally.
   - Run touched-definition TODO docstring check:
     `BASE_REF=origin/main scripts/dev/check_docstring_todos_diff.sh`

6. Report and follow-ups
   - Summarize commands run and results.
   - Note remaining failures, flaky tests, or follow-up tasks (for example,
     GUI tests if rendering changes were made, or CHANGELOG updates for
     user-facing changes).
  - Suggest commit batches and messages for any uncommited changes.
