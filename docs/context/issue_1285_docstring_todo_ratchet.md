# Issue #1285 TODO-Docstring Backlog Ratchet

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1285>

## Goal

Make existing `TODO docstring` placeholder debt visible and increase-only without turning the full
legacy backlog into an immediate cleanup blocker.

## Implementation

`scripts/validation/check_docstring_todos.py` keeps its default diff-only behavior for touched
definitions. New backlog modes add:

* `--mode report` for JSON counts by top-level area and file,
* `--mode write-baseline` for refreshing the tracked baseline after intentional cleanup,
* `--mode ratchet` for failing only when file-level placeholder counts increase.

The tracked baseline is `scripts/validation/docstring_todo_baseline.json`. The PR-ready wrapper now
runs both the existing diff-only touched-definition check and the backlog ratchet.

## Boundary

This does not replace all placeholder docstrings. It only prevents new or increased placeholder
debt relative to the tracked baseline. Baseline updates should happen after intentional cleanup or
explicit maintainer approval for known backlog changes.

## Validation

Planned validation:

```bash
./.venv/bin/python -m pytest tests/validation/test_check_docstring_todos.py -q
./.venv/bin/python scripts/validation/check_docstring_todos.py --mode ratchet
scripts/dev/check_docstring_todos_ratchet.sh
BASE_REF=origin/main scripts/dev/check_docstring_todos_diff.sh
git diff --check origin/main...HEAD
```
