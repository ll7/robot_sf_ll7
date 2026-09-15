# Review package (SREV-04)

Command-line glossary: SREV is the scenario-review portfolio; a review package
(`review-package.v1`) is a relocatable directory of verified payloads plus a
manifest; a component request (`component-request.v1`) is the fixture envelope
that invokes one workbench component; shared contract shapes live in
`robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.analysis_workbench.review_package` relocates the payloads
referenced by a `review-bundle.v1` index into a verified package. Every payload
is hash-verified against its declared digest before staging (tampered or missing
payloads fail that entry, never the whole run silently); remote, absolute,
traversal, and symlink-escaping references are refused; video payloads are never
staged; `--dry-run` lists the preservation operations without writing anything.

## Command

```bash
uv run python -m robot_sf.analysis_workbench.review_package \
  --input tests/fixtures/scenario_review/review_package/request.json \
  --config tests/fixtures/scenario_review/review_package/config.json \
  --output output/scenario_review/srev-04-smoke
```

## Output

- `package/`: verified payload bytes under their bundle-relative layout.
- `manifest.json`: per-entry artifact id, declared source URI/format/schema,
  source commit, config identity, units, coordinate frame, relocated path,
  observed SHA-256, and size, plus sorted operations and all declared source
  provenance (including video references that were not staged).
- `verification-report.json`: verified-entry count, diagnostics, and the
  declared provenance of video references that were not staged.
- A non-dry-run `complete` result advertises both JSON outputs in its
  `artifacts` list, with SHA-256 values matching the written bytes. Dry runs
  perform no writes and therefore advertise no output artifacts. `partial`,
  `unavailable`, and `failed` results never claim complete outputs.
- The printed result envelope carries `complete`, `partial`, `unavailable`, or
  `failed` with stable reason codes (`tampered_payload`, `source_unreadable`,
  `symlink_refused`, `video_not_staged`, `reference_not_local`,
  `staging_failed`, `publication_failed`, ...). Local source, bundle, output,
  and package paths must remain within the real base/output/package roots;
  absolute or traversal package names are rejected.

## Unavailable reasons and limits

- A missing, corrupt, tampered, or unsafe reference makes the result `partial`,
  never silently complete; unknown source formats are reported, not staged.
- Video classification uses the bundle reference's declared `format`, not its
  URI suffix. Video references are recorded as diagnostics/provenance and are
  never copied into the package.
- Evidence boundary: the manifest proves the package mirrors its declared
  sources byte-for-byte; it does not make the sources scientifically admissible.
  No benchmark, safety, or paper-facing claim follows from packaging.
