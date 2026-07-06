# Release Artifact Badging

This document defines the reproducibility and artifact badging levels used for `robot_sf_ll7` releases.
Adopting this terminology makes each release's reproducibility claims explicit, checkable, and comparable.

## Badge Levels

We define three incremental badging levels:

1. **`available`**
   - **Definition**: The release artifact bundle is archived, checksummed, and has a durable identifier (such as a DOI or a permanent release asset URL).
   - **Requirements**:
     - A published archive (e.g., `.tar.gz`, `.zip`).
     - A SHA-256 checksum manifest listing all files in the bundle.
     - A registered DOI (e.g., Zenodo) or a permanent release asset pointer.
   - **Non-claims**: `available` does **not** imply that the code builds, installs, or runs in any environment.

2. **`functional`**
   - **Definition**: The release bundle is self-sufficient. A clean, independent environment can build, install, and execute a documented smoke benchmark using only the files contained within the release bundle.
   - **Requirements**:
     - Must satisfy the `available` badge.
     - A documented setup and installation process.
     - A functional smoke test command provided with the bundle.
     - Successful execution of the smoke test in a clean environment, producing expected outputs.
   - **Non-claims**: `functional` does **not** imply that the headline results, tables, or graphs in accompanying papers are reproduced.

3. **`reproduced`**
   - **Definition**: An independent execution of the documented commands on the release bundle regenerates the headline benchmark results/tables within a specified numeric tolerance.
   - **Requirements**:
     - Must satisfy the `functional` badge.
     - Clear specification of target seeds, scenario matrices, and environment constraints.
     - Documented tolerance levels for numeric metrics (e.g., SNQI, success rate).
     - Execution of the full campaign or full validation pipeline regenerating the target tables/figures within tolerance.
   - **Non-claims**: `reproduced` does **not** guarantee exact bitwise identity of outputs, only statistical or metric equivalence within the stated tolerance.

## Validation Policy

- A release must explicitly declare its claimed badge level in its metadata (`source_manifest.json` or publication manifest).
- Claimed levels are validated using `scripts/validation/check_release_artifact_badging.py`.
- If a release claims a badge but fails to meet the checklist or validation requirements, it must fail closed (i.e. default to `none` or fail verification).
