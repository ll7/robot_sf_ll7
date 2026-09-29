---
name: test-value-gate
description: Decide whether a pytest test is worth adding, changing, or keeping, and prove that bug tests fail for the right reason.
category: validation
kind: atomic
phase: verification
requires_write: false
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# Test Value Gate

Idea source: the `test-audit` skill in the openclaw repository. Its license is unstated, so this
file is written independently and copies none of its text.

## When to use

Use this skill for any task that writes, changes, or reviews tests: new pytest tests, test edits
in a bug fix, and PR reviews that touch `tests/`. A green test run says little on its own. This
gate asks whether the test can ever catch something.

## Four questions before adding a test

Answer each in one line (in the PR body or review comment). If you cannot, do not add the test.

1. Which behavior does it protect? Name the behavior, not the function.
2. Which credible regression makes it fail? Describe a plausible future edit, not a typo.
3. Why does existing coverage miss it? Point at the nearest existing test and say what it skips.
4. Does it need a test-only seam in production code? If yes, justify the seam or choose another
   way to observe the behavior.

## Bug and correctness tests must fail first

A test added for a bug or a correctness claim has to be shown failing on the code before the fix,
and failing for the reason the fix addresses.

1. Check out the pre-fix code (or revert only the fix) and run the new test by explicit path:
   `uv run pytest <test file>::<test name> -q`.
2. Confirm the failure message points at the bug, not at an import error, a fixture problem, or a
   missing file.
3. Paste the command and the failing output into the PR body. Then show the same command passing
   on the fixed code.

If the pre-fix code cannot be run, say so and mark the test as unproven.

## Junk patterns

Reject or rewrite a test that matches one of these. Examples come from robot_sf reviews on
2026-09-29.

- Oracle from the code under test: the expected value is computed by calling the same function or
  helper that the test checks. It can only agree with itself. Use a value derived independently
  (a hand calculation, a fixed literal with its source, or an external reference).
- Template-only admission test: a release-admission test checks that a template or example file is
  well formed, while real admission (the path that accepts or rejects an actual release) has no
  test. Test the real admission path.
- Negative control that passes for the wrong reason: a "must reject" check or a skip that fires
  because an unrelated input is missing, for example a missing receipt causing a skip. Make the
  control fail on the exact rule it names, and assert on that reason.
- Promise larger than proof: a test name, docstring, or parametrized table claims more cases than
  the body exercises. Rename it or make the body cover what the name says.
- Hand-built oracle that passes on the old binding: a fixture built by hand matches both the
  old and the new wiring, so it cannot tell them apart. Run it on the pre-fix code; if it
  passes there, it does not test the change.

## Repo specifics

- `tests/conftest.py` defines `pytest_ignore_collect`, which overrides `--ignore`. Select test
  paths explicitly (`pytest <test file>`) and never rely on `--ignore` to narrow a run.
- Do not run a planner step on seeds 111-140. Those are held out. Use dev seeds 1001-1030 for
  development and tests.
- Prefer a small deterministic fixture over a full benchmark run. Follow `AGENTS.md` for
  validation depth and for what may run locally.

## Guardrails

- Do not delete a test only because it looks low value; report it and let the maintainer decide,
  unless the task is an explicit test cleanup.
- Do not add a production seam just to make a weak test possible.
- Do not report a test as verifying a fix without the pre-fix failure evidence.
- Do not count a skipped test as passing evidence.

## Output

- Four-question answers per new or changed test.
- For bug tests: the pre-fix command, the failing output, and the post-fix pass.
- Junk-pattern findings with file and test name, and the proposed rewrite or removal.
- Tests that stay unproven, with the reason.
