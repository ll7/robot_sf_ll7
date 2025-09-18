# Figures Output Naming — Design Note

Purpose: Define a canonical naming scheme and migration plan for figure output folders under `docs/figures/` to improve traceability, reproducibility, and LaTeX integration.

## Goals
- Deterministic, conflict-free folder names derived from inputs and repository state.
- Easy to reference in LaTeX with stable paths; avoid ad-hoc names like `figures_long_fix1`.
- Allow a moving alias for “current best” figures without breaking historical artifacts.

## Proposed Scheme
- Base directory: `docs/figures/`
- Canonical folder name pattern:
  - `<episodes-stem>__<gitsha7>__v<schema>`
    - `episodes-stem`: stem of the JSONL input path (e.g., `episodes_sf_long_fix1`).
    - `gitsha7`: short 7-char commit SHA at generation time.
    - `v<schema>`: integer schema version of the episode JSON (read from schema file / code constant).
- Example:
  - `docs/figures/episodes_sf_long_fix1__a1b2c3d__v1/`

Optional suffixes:
- Add `__ci` if generated with bootstrap confidence intervals.
- Add `__kde` if KDE overlays are enabled for distributions.

## Orchestrator Changes
- Add `--auto-out-dir` flag to `scripts/generate_figures.py`:
  - When set, compute the output folder name using the above pattern.
  - Schema version is inferred from the episodes JSONL (`schema_version` top-level or `_metadata.schema_version`) and falls back to `1` if not present.
  - Create the folder if missing; write a `meta.json` with:
    - input episodes path, git sha, schema version, timestamp, generator script version, CLI args.
- Keep `--out-dir` to override explicitly.
- Add `--set-latest` to update `docs/figures/_latest.txt` to point at the generated folder.

## Latest Alias
- Maintain a text file alias `docs/figures/_latest.txt` containing the relative path to the preferred folder for the current draft (one line).
- Optionally add a convenience symlink `_latest -> <folder>` on platforms that support it, but keep `_latest.txt` as the source of truth for portability.

## Migration Plan
1. Implement `--auto-out-dir` and `meta.json` writing.
2. Update `docs/README.md` with the naming scheme and how to include figures.
3. Regenerate current figures to a canonical folder and update LaTeX snippets.
4. Leave existing ad-hoc folders in place for now; add a cleanup task to prune once the paper draft stabilizes.

## Risks & Mitigations
- Risk: Paths change mid-draft. Mitigation: use `_latest.txt` and avoid hardcoding paths in LaTeX.
- Risk: Multiple teams generating figures on different commits. Mitigation: SHA in folder names ensures uniqueness; aggregates can be merged by copying folders.

## Acceptance Criteria
- New runs with `--auto-out-dir` produce uniquely named folders with a `meta.json` file.
- Docs show the pattern and provide a LaTeX include snippet using `_latest.txt`-resolved path.
