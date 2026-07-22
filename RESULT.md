# PR #6087 Review Blockers

## Fixed comments

- `tests/baselines/test_sicnav_planner.py:141-143`: mocked `Path.expanduser` directly and compared platform-aware `Path` values, removing the Windows-dependent `HOME` string assertion.
- `tests/baselines/test_sicnav_planner.py:87-88`: added an autouse fixture that saves and restores the process-global `random` and NumPy RNG states around every test.

## Validation

- `scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/baselines/test_sicnav_planner.py -q` — 68 passed.
- `scripts/dev/run_worktree_shared_venv.sh -- uv run ruff check tests/baselines/test_sicnav_planner.py` — passed.
- `git diff --check` — passed.

## Delivery

- Commit: `17fa7b46a5a96557f478fd90cd11cc73efc3225d`
- Pushed to PR branch `cheap/cheap-issue-6077-b5a9ba703747`.
- Post-push head: `17fa7b46a5a96557f478fd90cd11cc73efc3225d`.

## Review state

- `PRRT_kwDOLRSZdc6SqrJp` — resolved; now outdated after the path-test edit.
- `PRRT_kwDOLRSZdc6SqrJy` — resolved; current-head RNG fixture comment addressed.
- Unresolved review threads: none.
- Blockers: none within the authorized scope.
