# Issue 2460 Evidence Bundle v1

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2460>

## What Was Implemented

`scripts/tools/benchmark_publication_bundle.py` now has an `evidence-bundle` subcommand for compact
research evidence packets. It is backed by `export_evidence_bundle` in
`robot_sf/benchmark/artifact_publication.py`.

The bundle writes:

- `payload/` with exactly the selected compact files;
- `evidence_bundle_manifest.json` with `schema_version: evidence_bundle.v1`;
- `checksums.sha256` for all payload files.

The schema contract lives at `robot_sf/benchmark/schemas/evidence_bundle.v1.json`.

## Claim Boundary

An evidence bundle is a provenance and reproducibility aid. It does not by itself establish that a
research claim is benchmark-strength, paper-grade, or scientifically sufficient. The bundle must
carry the conservative `claim_boundary`, including diagnostic-only, fallback, degraded, blocked, or
not-benchmark-evidence status when applicable.

## Scope Decisions

- The first helper is explicit-file-only rather than run-directory auto-selection. This keeps the
  initial contract minimum and prevents accidental mirroring of large `output/` trees.
- Large raw logs, videos, checkpoints, raw episode streams, and model caches remain excluded unless
  a separate durable artifact path is chosen.
- Existing publication-bundle and release-helper contracts are unchanged.

## Proof Surface

Targeted CLI tests cover:

- manifest/checksum generation for a compact fixture;
- validation against `evidence_bundle.v1.json`;
- fail-closed behavior for missing requested files.

The user-facing command and required fields are documented in `docs/context/evidence/README.md`.
