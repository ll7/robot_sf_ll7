# Quickstart: Docstring Enforcement

1. **Activate environment**
   ```bash
   uv sync && source .venv/bin/activate
   ```
2. **Run Ruff with docstring rules**
   ```bash
   uv run ruff check
   ```
3. **Fix violations**
   - Add module/class/function docstrings with a one-line summary + blank line + detail section.
   - Include `Args:` and `Returns:` sections for callables; document `Raises:` when exceptions are part of the contract.
   - Reference Loguru logging side effects when applicable to satisfy Principle XI.
   - Use the grouped report to focus on one file at a time: `uv run python scripts/tools/docstring_report.py robot_sf/benchmark/` writes `output/issues/docstrings_summary.json` with per-file counts.
4. **Verify**
   - Re-run `uv run ruff check` until status code 0.
   - Run `uv run pytest tests` to ensure docstring edits did not break imports or fixtures.
5. **CI expectations**
   - `Ruff: Format and Fix` VS Code task and `lint` workflow run the same command; merges are blocked if any docstring rule fails.
