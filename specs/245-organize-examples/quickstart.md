# Quickstart: Implementing Example Reorganization (Issue 245)

1. **Inventory & tagging**
   - Run `ls examples/*.py` and record existing scripts.
   - Draft `examples/examples_manifest.yaml` with categories, summaries, and `ci_enabled` flags (defaults to `true`).
   - Identify duplicates or obsolete scripts and mark candidates for `_archived/` with a `ci_reason`.

2. **Create tiered directories**
   - Add `examples/quickstart/`, `examples/advanced/`, `examples/benchmarks/`, `examples/plotting/`, and `examples/_archived/`.
   - Move scripts according to manifest assignments; keep history via `git mv` to preserve blame.
   - Update intra-example imports to use package factories after relocation.

3. **Docstring overhaul**
   - For each active script, add a module docstring containing: purpose, how to run (`uv run python examples/...`), prerequisites, expected output, and limitations.
   - Ensure the first sentence matches the manifest `summary` for consistency.

4. **Documentation updates**
   - Generate `examples/README.md` from the manifest: include decision tree, category descriptions, and index table (can be scripted later; initial version can be manual but must stay in sync).
   - Update `docs/README.md` and `docs/dev_guide.md` to link to the new README and describe the tiered structure.
   - Add `_archived/README.md` explaining archival policy and mapping to replacements.

5. **Automation**
   - Implement `tests/examples/test_examples_run.py` that reads the manifest, filters out `ci_enabled: false` entries, and executes each script headlessly via `subprocess.run`.
   - Provide fixtures to set `DISPLAY=`, `MPLBACKEND=Agg`, and `SDL_VIDEODRIVER=dummy` for graphical demos.
   - Record durations per script and fail if any exit code is non-zero; print skip reasons for archived/manual examples.

6. **Validation & CI**
   - Run `uv run pytest tests/examples/test_examples_run.py -k "not slow"` locally to ensure harness passes.
   - Execute existing validation scripts (`scripts/validation/test_basic_environment.sh`, etc.) to confirm no regressions.
   - Update CI workflow to include the new pytest module (if not covered by default test discovery).

7. **Final checks**
   - Verify manifest, README index, and directory contents remain in sync (consider a lightweight check inside the pytest module).
   - Confirm `_archived/` entries document replacements and reasons.
   - Update `CHANGELOG.md` summarizing the example reorganization.
