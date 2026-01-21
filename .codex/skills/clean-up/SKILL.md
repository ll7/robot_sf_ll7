---
name: clean-up
description: "Clean up the current branch in the Robot SF repo by following docs/dev_guide.md and the repo VS Code tasks; use when asked to tidy a branch, run Ruff format/fix, or run parallel pytest before sharing changes."
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
   - Use VS Code task "Ruff: Format and Fix":
     `uv run ruff check --fix . --output-format concise; uv run ruff format .; uv run ruff check . --statistics`
   - If Ruff reports issues, fix them and rerun until clean.

4. Run tests in parallel
   - Use VS Code task "Run Tests in parallel":
     `uv run pytest -n auto`
   - If tests fail, evaluate test value first (Constitution Principle XIII /
     dev guide testing strategy). Classify failures and decide whether to fix,
     defer, or ask for direction before removing or relaxing tests.

5. Report and follow-ups
   - Summarize commands run and results.
   - Note remaining failures, flaky tests, or follow-up tasks (for example,
     GUI tests if rendering changes were made, or CHANGELOG updates for
     user-facing changes).
