# Quickstart: Clean Root Output Directories

This guide summarizes the changes introduced by feature `243-clean-output-dirs` and how collaborators can adapt their local workflows.

## 1. Prepare Environment
1. Pull the latest changes on branch `243-clean-output-dirs`.
2. Ensure dependencies are up to date: `uv sync --all-extras`.

## 2. Run the Migration Helper
1. Execute the forthcoming helper script (placeholder path shown):
   ```bash
   uv run python scripts/tools/migrate_artifacts.py
   ```
2. Review the migration report printed to the console (and saved under `output/migration-report.json`).
3. Verify that legacy directories (`results/`, `recordings/`, `tmp/`, `htmlcov/`, `wandb/`) no longer reside at the repository root.

## 3. Validate Guard Checks
1. Run the guard script (will be added during implementation):
   ```bash
   uv run python scripts/tools/check_artifact_root.py
   ```
2. Ensure the script reports `0` violations.
3. If violations occur, inspect the offending script, update its output directory, and rerun the guard.

## 4. Execute Quality Gates
1. `uv run ruff check .`
2. `uv run pytest tests`
3. Confirm coverage artifacts land under `output/coverage/`.

## 5. Override Artifact Location (Optional)
1. Set `ROBOT_SF_ARTIFACT_ROOT` to the desired path:
   ```bash
   export ROBOT_SF_ARTIFACT_ROOT="$HOME/robot_sf_artifacts"
   ```
2. Rerun the guard script and quality gates to confirm artifacts flow to the new location.

## 6. Cleanup Guidance
- To remove all generated artifacts, delete the `output/` directory (or the override path) and rerun the guard to confirm a clean state.
- Avoid deleting individual subdirectories unless you understand their retention purpose.

## 7. Documentation Touchpoints
- Expect updates to `README.md` and `docs/dev_guide.md` describing the artifact policy, migration script, and guard checks.
