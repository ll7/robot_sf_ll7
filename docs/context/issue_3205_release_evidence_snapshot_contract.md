# Issue #3205 Release Evidence Snapshot Contract

Issue: [#3205](https://github.com/ll7/robot_sf_ll7/issues/3205)
Status: diagnostic-only.

This is the first dry-runable contract slice; it is not benchmark evidence by itself.

## Purpose

`scripts/tools/release_evidence_snapshot.py` builds a JSON manifest over tracked release evidence
inputs without cutting a release, running SLURM, uploading artifacts, or depending on worktree-local
`output/` files. It is a practical fresh-clone gate shape for tagged research releases: every
included file is read from Git at the requested source ref and recorded with a SHA-256 checksum.

Canonical dry-run command:

```bash
uv run python scripts/tools/release_evidence_snapshot.py \
  --output-json output/release_evidence_snapshot.json
```

Use `--tag <tag>` or `--source-ref <ref>` to snapshot a tag or other Git ref. Use
`--require-input <path>` for release inputs that must exist; missing required paths return exit code
2 and set `status: fail_closed`.

## Manifest Contract

The manifest schema is `release_evidence_snapshot.v0.1` and records:

- source ref, resolved source commit, dirty-worktree state for `HEAD`, generation time, and command;
- DOI-ready metadata fields: release id, title, DOI or placeholder, repository URL, license, and
  citation file;
- file list with Git-ref SHA-256 checksums and byte sizes;
- config and seed identifiers discovered from included YAML/JSON inputs;
- tracked `artifact_catalog` summaries with principal table/figure outputs and catalog checksum
  validation;
- fallback, degraded, unavailable, and failed-row exclusions copied from artifact claim boundaries;
- fail-closed missing-input records;
- reference-only links to future #3075 artifact-backend and #3071 researcher-guide integration,
  plus compatibility with the #3076 result-store vocabulary.

## Current Dry-Run Result

Focused run on this branch:

```bash
uv run python scripts/tools/release_evidence_snapshot.py \
  --output-json output/release_evidence_snapshot_issue3205.json
```

The command found the tracked artifact catalog at
`docs/context/evidence/issue_2037_artifact_compiler_smoke_2026-06-01/artifact_catalog.yaml`,
included the promoted campaign table and planner-status figure files from that catalog, and
validated their checksums against the Git tree. The manifest status was `valid`, but the evidence
classification remained `diagnostic-only` because that artifact catalog explicitly says it is not
standalone benchmark evidence.

## Boundary

This command is a release-evidence manifest gate, not a release executor. It does not:

- run benchmark campaigns or regenerate paper metrics;
- upload, hydrate, or verify external artifact-backend objects;
- certify that #3075 or #3071 are complete;
- turn diagnostic artifact catalogs into benchmark or paper-grade evidence.

Future release work should wire this manifest to the artifact backend/hydration path and the
researcher guide once those contracts are implemented, then use `--require-input` for every durable
release asset that must be present in a fresh clone.

## Validation

Focused proof for this slice:

```bash
uv run python scripts/tools/release_evidence_snapshot.py --help
uv run pytest tests/tools/test_release_evidence_snapshot.py
uv run python scripts/tools/release_evidence_snapshot.py \
  --output-json output/release_evidence_snapshot_issue3205.json
```
