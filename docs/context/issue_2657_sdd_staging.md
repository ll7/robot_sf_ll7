# Issue #2657 Stanford Drone Dataset (SDD) Staging (2026-06-23)

This note makes SDD scenario-prior staging reproducible and connects staging state to
scenario-prior generation, so dataset-backed claims cannot be implied without dataset-backed
input. It follows [`docs/templates/external_data_audit.md`](../templates/external_data_audit.md).

> **Consolidated under Issue #3473.** The SDD staging behavior now lives in the canonical
> external-data subsystem `scripts/tools/manage_external_data.py` (the `sdd` `AssetSpec` references
> the manifest `configs/data/sdd_staging_manifest.yaml`; canonical functions are
> `load_sdd_staging_spec`, `validate_sdd_staging`, `resolve_sdd_scenario_prior_mode`,
> `run_sdd_download`, with CLI `sdd-plan|sdd-status|sdd-validate|sdd-mode|sdd-download`).
> `scripts/data/stage_sdd_dataset_issue_2657.py` is preserved as a thin wrapper that delegates to
> the canonical subsystem, so the commands below still work unchanged.

## Summary

- Dataset or asset name: Stanford Drone Dataset (SDD) annotations
- Upstream source URL: <https://cvgl.stanford.edu/projects/uav_data/>
- Upstream version, tag, or release: `sdd-v1.0-uav-data` (manifest `version_tag`)
- Related issue: `Issue #2657` (also #1497, #1126 importer lineage; #3161 real-world staging)
- Intended Robot SF use: provide original SDD annotation text files to the SDD importer
  (`scripts/tools/import_sdd_scenarios.py`) so scenario-prior generation can run in
  `dataset_backed_prior` mode.

## License And Access

- Observed license: Creative Commons Attribution-NonCommercial-ShareAlike 3.0
- License URL: <https://creativecommons.org/licenses/by-nc-sa/3.0/>
- Access restrictions: license-gated / manual acquisition; **no repository-approved direct
  download URL is encoded**.
- Citation requirement: cite Robicquet et al., "Learning Social Etiquette: Human Trajectory
  Understanding in Crowded Scenes," ECCV 2016.
- Redistribution: non-commercial, share-alike; Robot SF does **not** redistribute raw SDD.
- License compatibility decision: reference-only for now; raw data stays local and git-ignored.

## Download And Raw-File Policy (no-auto-download safety contract)

The staging tool is `scripts/data/stage_sdd_dataset_issue_2657.py`, driven by the manifest
`configs/data/sdd_staging_manifest.yaml`. Its safety contract is enforced in code:

- **Never auto-downloads.** The default invocation only PLANS/REPORTS (what would be downloaded,
  where, expected size, disk check) and exits without touching the network.
- A network fetch requires an explicit `--confirm-download` flag **and** an interactive `y/N`
  prompt; the prompt is skipped only when `--yes` is also given for non-interactive confirmation.
- **Disk-space check before any fetch:** free space at the staging location is checked against the
  manifest's `expected_total_size_bytes`; the tool fails closed (refuses) if insufficient,
  reporting required vs available.
- **License-gated fail-closed:** even after confirmation and a passing disk check, the tool refuses
  because no approved download URL is encoded. A maintainer must add a license-approved URL to the
  manifest and wire the fetch before the download path can run.
- Raw data is staged into the **git-ignored** subfolder `output/external_data/sdd`
  (`output/` is already covered by `.gitignore`).
- Expected raw files: at least one `**/annotations.txt` (the SDD importer's input format).
- Fail-closed when raw files are missing: availability is reported `missing` and scenario-prior
  generation is forced to `proxy_schema_smoke`.

Acquisition steps (user action, outside this tooling):

1. Obtain the SDD annotation archive from the official project page under its license.
2. Place the extracted annotation files under `output/external_data/sdd/` (keep the license with
   your local copy), or configure a license-approved `download_url` in the manifest.
3. Run `uv run python scripts/data/stage_sdd_dataset_issue_2657.py validate` to validate and
   record an aggregate checksum.

## Checksums And Manifest

- Manifest path: `configs/data/sdd_staging_manifest.yaml`
- Checksum algorithm: SHA-256 (aggregate tree checksum over relative path, size, and per-file
  sha256 for every matched annotation file).
- `checksums.tree_sha256` / `expected_tree_sha256` are **placeholders** filled on a real,
  user-confirmed staging run; pin `expected_tree_sha256` after first trusted staging to detect
  drift (a mismatch fails closed and refuses to mark SDD as `staged`).
- On a successful `validate`, the tool writes `sdd_staging_status.json` into the (git-ignored)
  staging dir recording the checksum, file count, and `local_availability: staged`.
- Verification command:
  `uv run python scripts/data/stage_sdd_dataset_issue_2657.py --check`

## Proxy-vs-dataset-backed gate

`scenario_prior.v1` distinguishes two modes:

- `proxy_schema_smoke` — no validated SDD; schema/proxy evidence only.
- `dataset_backed_prior` — SDD staged **and** validated (expected files present and, if pinned,
  checksum match).

The gate is exposed by `resolve_sdd_scenario_prior_mode()` in the canonical subsystem
`scripts/tools/manage_external_data.py` (still reachable as `resolve_scenario_prior_mode()` via the
thin `scripts/data/stage_sdd_dataset_issue_2657.py` wrapper) and consumed by
scenario-prior generation (`scripts/analysis/calibrate_scenario_priors_from_traces_issue_2726.py`
surfaces `scenario_prior_mode` / `dataset_backed` / `sdd_staging_gate` in its registry YAML and
`report.json`). A missing or unvalidated SDD copy forces `proxy_schema_smoke`; only a
staged-and-validated copy unlocks `dataset_backed_prior`. The trace-cluster calibration script is
proxy by construction (it consumes simulation traces, not SDD) and surfaces the SDD state for
provenance only — it never reports dataset-backed.

## Robot SF Use Decision

- Use status: reference-only until SDD is locally staged and validated.
- Benchmark eligibility: blocked until hydrated; `proxy_schema_smoke` is smoke-only.
- Redistribution status: no redistribution of raw SDD.
- Required follow-up: real-world staging/calibration tracked under `Issue #3161`.

## Reproducibility commands

> Canonical owner note: the commands below drive the original `scripts/data/stage_sdd_dataset_issue_2657.py`
> staging script. The provenance-safe BYO opt-in gate added for #1497 lives in the canonical
> external-data owner `scripts/tools/manage_external_data.py` (`sdd-preflight`); see the
> "Issue #1497" section below.

```bash
# Plan only -- NEVER downloads (default):
uv run python scripts/data/stage_sdd_dataset_issue_2657.py
# Availability status / checksums (no download):
uv run python scripts/data/stage_sdd_dataset_issue_2657.py --check
# Validate a locally-staged copy (no download):
uv run python scripts/data/stage_sdd_dataset_issue_2657.py validate
# Print the scenario-prior mode gate:
uv run python scripts/data/stage_sdd_dataset_issue_2657.py mode
```

## Issue #1497 — Bring-your-own (BYO) opt-in preflight (2026-06-27)

Issue #1497 owns the provenance-safe staging *gate* for licensed SDD annotations (scenario curation
itself stays in #1126). Under the BYO-dataset reframe (#3065, issue-audit 2026-06-22) the project
never licenses, hosts, or redistributes SDD; a contributor stages a copy they already have rights
to. This slice makes that opt-in explicit and machine-checkable without downloading, ingesting, or
transforming any data:

- The canonical manifest `configs/data/sdd_staging_manifest.yaml` now carries two fields:
  - `retrieval_recipe`: an ordered list of concrete acquisition steps (no auto-download).
  - `license_acknowledgment`: `{required, acknowledged, statement}`. It ships
    `acknowledged: false` so the committed manifest never implies redistribution rights; a
    contributor sets it `true` in their local checkout after reading the license.
- A new preflight reports the two staging prerequisites and the blocked-external-input state. It is
  `ready` only when the license acknowledgment is **satisfied** AND the annotation files are present
  locally; otherwise it fails closed (CLI exit 2). Parsing fails closed on a non-boolean
  acknowledgment, a malformed recipe, or a non-string statement so the gate cannot be bypassed by a
  typo. The acknowledgment is **mandatory**: `license_acknowledgment.required: false` is rejected, so
  a locally edited manifest cannot disable the gate and report `ready` without an explicit license
  affirmation.
- Canonical owner: extends `scripts/tools/manage_external_data.py`
  (`SddLicenseAcknowledgment`, `build_sdd_preflight`, CLI `sdd-preflight`); no parallel per-issue
  script. Tests: `tests/tools/test_sdd_preflight_issue_1497.py`.

Provenance state (placeholders until a contributor stages a real copy): source
<https://cvgl.stanford.edu/projects/uav_data/>; version `sdd-v1.0-uav-data`; staging dir
`output/external_data/sdd` (git-ignored); checksum pinned by `sdd-validate` after first trusted
staging; license CC BY-NC-SA 3.0 (manual, license-gated, no approved download URL).

```bash
# Report BYO prerequisites + blocked-external state (no download); exit 0 ready / 2 blocked:
uv run python scripts/tools/manage_external_data.py --json sdd-preflight
```

## Issue #4079 — Kaggle-reduced SDD Provenance Decision (2026-07-08)

Issue #4079 investigates whether the Kaggle-reduced SDD annotation package (aryashah2k/stanford-drone-dataset)
is byte/content-equivalent to the official Stanford SDD archive. The conclusion is that the Kaggle
source remains a **local BYO staging option only**; canonical equivalence cannot be verified because
the official archive is unreachable.

### Official Source Availability

- Official archive URL: `http://vatic2.stanford.edu/stanford_campus_dataset.zip`
- Official project page: <https://cvgl.stanford.edu/projects/uav_data/>
- Availability status: **blocked_official_source_unavailable**
- Verification date: 2026-07-02 (verified from two independent networks)
- Network observation: TCP 80 and 443 refused/filtered; DNS resolves (171.64.68.58) but host does
  not serve HTTP/HTTPS traffic. The archive is unreachable and cannot be fetched for comparison.

### Kaggle-Reduced Source

- Kaggle source: <https://www.kaggle.com/datasets/aryashah2k/stanford-drone-dataset>
- Staged local checksum (annotations only, 60 files):
  ```
  66dec2c82b0a01b23bf9fa9acef352af86549e7ea749811ea4ef9c47003d4acf
  ```
- Source classification: `local_byo_staging_only`
- Byte-equivalence to official archive: **unknown and unverified** (official archive unavailable)

### Provenance Decision

Because the official Stanford SDD archive is unreachable, byte-equivalence between the Kaggle-reduced
package and the canonical source cannot be established. The project treats the Kaggle-staged files as
a local bring-your-own staging convenience only, not as a canonical-equivalent source.

This means:

1. The Kaggle checksum (`66dec2c82...`) is **NOT** promoted to `configs/data/sdd_staging_manifest.yaml`
   as an accepted canonical checksum.
2. Any benchmark-facing claim using SDD-derived priors **MUST** cite the provenance caveat that the
   annotation source is a non-canonical Kaggle-reduced copy, not the official Stanford archive.
3. If the official archive becomes available in the future, a fresh equivalence comparison should be
   run before promoting any checksum to canonical status.

### Benchmark Claim Caveat

When using SDD-derived scenario priors or any benchmark result that depends on SDD annotations,
the provenance caveat must be stated:

> "Annotation source: Kaggle-reduced SDD (aryashah2k/stanford-drone-dataset), local BYO staging
> only. Byte-equivalence to the official Stanford SDD archive (cvgl.stanford.edu) is unverified;
> the official archive was unreachable as of 2026-07-02."

No claim may state or imply that the staged annotations are from the official Stanford source without
explicit qualification.

### Validation Commands

```bash
# Check current SDD staging status (including provenance note):
uv run python scripts/tools/manage_external_data.py --json sdd-status

# Validate any locally-staged copy (official or Kaggle-derived):
uv run python scripts/tools/manage_external_data.py --json sdd-validate
```

## Validation Checklist

- [x] Source URL and version tag are recorded.
- [x] License, access restrictions, and citation requirements are recorded.
- [x] Raw files, derived files, and redistribution decisions are separated.
- [x] Checksum plan recorded (aggregate SHA-256; placeholders until real staging).
- [x] Robot SF use decision is reference-only / smoke-only until hydrated.
- [x] Missing artifacts fail closed with an actionable message and force `proxy_schema_smoke`.
- [x] No restricted raw data is committed; staging dir is git-ignored.
- [x] Official SDD archive unavailability is documented with verification date.
- [x] Kaggle-reduced source is classified as local BYO only (not canonical-equivalent).
