---
name: pr-ready-check
description: "Run the repository PR readiness pipeline using shared scripts/dev entry points (ruff fix/format, parallel tests, changed-files coverage, and docstring TODO checks)."
---

# PR Ready Check

## Overview

Run the same PR validation pipeline used by local VS Code tasks, but via reusable
`scripts/dev` commands that are stable for Codex automation.

## Workflow

1. Confirm repo + guidance
   - Ensure you are in the Robot SF repo root.
   - Review `docs/dev_guide.md`, `.specify/memory/constitution.md`, and
     `.github/copilot-instructions.md` when context is needed.

2. Ensure environment
   - Prefer running from repo root with `.venv` present.
   - Scripts source `.venv/bin/activate` when available.

3. Run PR readiness checks
   - Default command:
     `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
   - Optional overrides:
     `BASE_REF=<ref> MIN_COVERAGE=80 GOAL_COVERAGE=100 scripts/dev/pr_ready_check.sh`

4. If checks fail
   - Fix issues directly in code/tests/docs.
   - Re-run the same command until green.
   - Apply test-value evaluation (Constitution Principle XIII) for failing tests.

5. Report result
   - Summarize which checks ran and whether each passed:
     - Ruff fix/format/lint stats
     - Parallel pytest (`-n auto -x --failed-first` by default via `scripts/dev/run_tests_parallel.sh`)
     - Changed-files coverage gate
     - Touched-definition docstring TODO gate

6. Commit and PR
   - Once all checks pass, commit your changes with a clear message.
   - Verify the issue addressed is acutally resolved by the changes.
   - Push to your branch and create/update the PR referencing related issues.
