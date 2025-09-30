# Tasks for Feature 139 — Extract reusable helpers (Phase A)

Feature: Extract frame/recording, plotting/overlay, and formatting/table helpers from `examples/classic_interactions_pygame.py` into `robot_sf/benchmark/visualization.py` and `robot_sf/benchmark/utils.py`.
Branch: `139-extract-reusable-helpers`

Order: TDD-first, small commits, preserve example behavior. Use lazy imports for optional deps (moviepy, SimulationView). Run Ruff & pytest locally after each commit.

Setup tasks

T001. (Setup) Ensure local environment and dev tools are ready.
- Files/commands:
  - Run: `git submodule update --init --recursive`
  - Run: `uv sync && source .venv/bin/activate`
  - Run: `uv run ruff check --fix . && uv run ruff format .`
- Notes: Do this before running tests.

Contract test tasks (TDD) — one failing contract test per contract file [P]

T010. (Test [P]) Create contract test for `frame_shape_from_map`. [X]
- File: `tests/unit/benchmark/test_contract_frame_shape.py`
- Create a pytest test that:
  - Imports the contract `specs/139-extract-reusable-helpers/contracts/visualization_helper_contract.md` (as guidance) and asserts that calling `frame_shape_from_map` with a missing path raises FileNotFoundError and with a simple SVG string returns a positive (w,h) tuple.
  - This test should import the future function from `robot_sf.benchmark.visualization` but initially mark import as xfail or mock the function to keep test failing until implementation.
- Parallelizable: Yes [P]

T011. (Test [P]) Create contract test for `overlay_text`. [X]
- File: `tests/unit/benchmark/test_contract_overlay_text.py`
- Test behavior:
  - Create a minimal duck-typed `canvas` object with `draw_text` method; assert `overlay_text(canvas, 'hi', (1,2))` mutates or calls `draw_text`.
  - Assert TypeError raised when canvas lacks `draw_text`.
- Parallelizable: Yes [P]

T012. (Test [P]) Create contract test for `format_summary_table`. [X]
- File: `tests/unit/benchmark/test_contract_format_summary_table.py`
- Test behavior:
  - Call with empty dict → expect ValueError.
  - Call with small metrics dict → expect a non-empty Markdown table string.
- Parallelizable: Yes [P]

Core implementation tasks (after tests created — TDD order)

T020. (Core) Implement `robot_sf/benchmark/visualization.py` with minimal helpers: [X]
- File: `robot_sf/benchmark/visualization.py`
- Implementations:
  - `def frame_shape_from_map(map_svg_path: str) -> tuple[int,int]` — parse simple SVG width/height attributes using `xml.etree.ElementTree` and return ints; raise FileNotFoundError or ValueError per contract.
  - `def overlay_text(canvas, text: str, pos: tuple[int,int], font: Optional[str]=None) -> None` — call `canvas.draw_text(text, pos, font)` and raise TypeError if method missing.
- Tests: The T010/T011 tests should now import the real functions and pass.
- Notes: Use lazy import for optional heavy deps; add docstrings per Constitution XI.
- Parallelizable: no (same file sequential edits allowed)

T021. (Core) Implement `robot_sf/benchmark/utils.py` with formatting helpers: [X]
- File: `robot_sf/benchmark/utils.py`
- Implementations:
  - `def format_summary_table(metrics: dict[str,float]) -> str` — return a Markdown table string; raise ValueError on empty input.
  - Small helper `def _latex_safe(s: str) -> str` if needed.
- Tests: T012 should pass after this implementation.
- Parallelizable: yes [P]

Integration & example update tasks

T030. (Integration) Update `examples/classic_interactions_pygame.py` to import helpers from new modules. [X]
- File: `examples/classic_interactions_pygame.py`
- Replace local helper definitions with imports:
  - `from robot_sf.benchmark.visualization import frame_shape_from_map, overlay_text`
  - `from robot_sf.benchmark.utils import format_summary_table`
- Add an integration smoke test run in a new test `tests/unit/benchmark/test_example_dry_run.py` that runs the example with `--dry-run` and expects no exceptions.

  - Add an integration smoke test run in a new test `tests/unit/benchmark/test_example_dry_run.py` that runs the example with `dry_run=True` and expects no exceptions. The test should be lightweight and not require heavy deps.
- Notes: Keep behavior identical; if behavior differs, revert and split the change into smaller commits.
- Parallelizable: no

Testing, linting, and CI tasks

T040. (Test) Add unit tests scaffolding and run local tests.
- Files:
  - `tests/unit/benchmark/test_contract_frame_shape.py` (from T010)
  - `tests/unit/benchmark/test_contract_overlay_text.py` (from T011)
  - `tests/unit/benchmark/test_contract_format_summary_table.py` (from T012)
  - `tests/unit/benchmark/test_example_dry_run.py` (from T030)
- Commands:
  - `uv run pytest tests/unit/benchmark -q`
- Expected: All unit tests pass locally
 - Status: [X] Unit tests and full test suite executed; focused benchmark unit tests and full pytest run passed locally (714 passed, 5 skipped).

T041. (Lint) Run Ruff and format to clean style after edits.
- Commands:
  - `uv run ruff check --fix . && uv run ruff format .`
 - Status: [X] Ruff autofix and format executed; repository is ruff-clean.

Polish & docs

T050. (Docs) Add docstrings to moved helpers and update `docs/dev/issues/classic-interactions-refactor/design.md` with a short migration note.
- Files:
  - `robot_sf/benchmark/visualization.py` (docstrings)
  - `robot_sf/benchmark/utils.py` (docstrings)
  - `docs/dev/issues/classic-interactions-refactor/design.md` (new file describing rationale and migration steps)
- Parallelizable: yes [P]

T051. (CHANGELOG) Add a one-line changelog entry in `CHANGELOG.md` under Unreleased.

T052. (Docs Index) Add link to the new design doc in the central docs index near benchmark documentation.
- File to edit: `docs/README.md`
- Action:
  - Insert a bullet/link near the "Benchmark" or "benchmarking" section that points to `docs/dev/issues/classic-interactions-refactor/design.md` and a short one-line description: "Design note: Extract visualization & formatting helpers (feature 139)".
  - If `docs/README.md` uses a table of contents, add the entry in the appropriate place and run any docs-generator if applicable.
- Suggested edit command (example):
  - `git apply -p0 <<'PATCH'
  --- a/docs/README.md
  +++ b/docs/README.md
  @@
   - Benchmarks
     - ...existing entries...
  +  - Design note: Extract visualization & formatting helpers (feature 139) — `docs/dev/issues/classic-interactions-refactor/design.md`
  PATCH`
- Notes: This task satisfies Constitution VIII (Documentation as an API Surface). Mark as [P] since it's independent.
- Parallelizable: yes [P]

Final checks & PR

T070. (PR) Run full test suite and open PR from `139-extract-reusable-helpers` to `main` with the description and linking the design doc and spec.
- Commands:
  - `uv run ruff check . && uv run pytest tests`
  - Create PR with the branch and link to `specs/139-extract-reusable-helpers/spec.md` and `docs/dev/issues/classic-interactions-refactor/design.md`.
 - Status: [READY] Full test suite green; branch ready for PR. I can open the PR if you want me to.

Parallel execution guidance
- Can run in parallel [P]: T010, T011, T012, T021, T050
- Must run sequentially: T020 → T030 → T040 → T070

Notes & dependency map
- T010/T011/T012 (contract tests) must exist before T020/T021 implementation to follow TDD.
- T020 implements visualization functions; T021 implements formatting helpers.
- T030 updates the example to import moved helpers and requires T020+T021 to be merged locally.
- T040/T041 ensure quality gates before PR.

Created-by: automated /tasks generator
