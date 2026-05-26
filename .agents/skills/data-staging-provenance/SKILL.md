---
name: data-staging-provenance
description: Stage external datasets and assets with checksum, license, raw-file, derived-file, and benchmark-readiness
  provenance.
category: benchmark-evidence
kind: atomic
phase: planning
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: true
delegates_to:
- artifact-provenance
output_schema: data_staging_summary.v1
---

# Data Staging Provenance

## When to use

Use this skill for SDD, SocNavBench, S3DIS, CARLA fixtures, or other external datasets/assets that need staging, license provenance, checksums, and benchmark-readiness decisions.

## Workflow

1. Record source URL, access date, license, citation, and any access restrictions.
2. Verify checksums for raw inputs and generated derived files when practical.
3. Keep raw licensed files untracked unless the license and maintainer decision explicitly allow tracking.
4. Classify derived outputs as `benchmark_ready` or `exploratory_only`.
5. Use `artifact-provenance` for manifests and compact tracked evidence.

## Guardrails

- Apply the fail-closed benchmark policy when artifacts or staged data affect benchmark evidence.
- Do not treat synthetic stand-ins, hand-authored placeholders, or fixtures as official-data evidence.
- Do not commit restricted raw data.
- Do not remove license or citation metadata from derived manifests.

## Output

Use `data_staging_summary.v1`.
