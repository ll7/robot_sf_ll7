# Test Suite Orientation

This directory holds the pytest suites. The canonical owners are
[`docs/qa_test_strategy.md`](../docs/qa_test_strategy.md) for test taxonomy, the command matrix, and
failure classification, and [`docs/dev_guide.md`](../docs/dev_guide.md) for contributor workflow.
This page maps the tree so a newcomer can find the right suite quickly.

| Path | What lives there | Typical command |
| --- | --- | --- |
| `tests/` (top level) | Unit, contract, integration, scenario, compatibility, and acceptance tests. | `uv run pytest -m "not slow" tests` |
| `tests/dev/` | Repository-automation and tooling contract checks (skills, instruction graph, worktrees, CI helpers). | `uv run pytest tests/dev` |
| `tests/validation/` | Validation-gate and evidence-integrity checks. | `uv run pytest tests/validation` |
| `tests/benchmark/`, `tests/benchmark_full/`, `tests/unit/benchmark/` | Benchmark contract, statistics, resume, and report-format tests. | `uv run pytest tests/benchmark tests/benchmark_full tests/unit/benchmark` |
| `tests/pygame/` | GUI rendering and playback regressions. | `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/pygame` |
| `fast-pysf/tests/` | SocialForce (`pysocialforce`) backend tests. | `uv run python -m pytest fast-pysf/tests -v` |
| `tests/conftest.py` | Shared markers, fixtures, and slow-test classification. | n/a |

Conventions:

- Name new tests `test_<feature>.py` and put shared fixtures in `conftest.py`.
- Replace `TODO docstring` placeholders in any test file you touch; the placeholder baseline is an
  increase-only ratchet. Check it with
  `uv run python scripts/validation/check_docstring_todos.py --mode report`.
- Use the command matrix in [`docs/qa_test_strategy.md`](../docs/qa_test_strategy.md) for validation
  lanes, failure classification, and rerun boundaries instead of ad-hoc commands.
