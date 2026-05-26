---
name: artifact-provenance
description: Classify, promote, or document generated artifacts so durable evidence is separated from
  local output caches.
category: benchmark-evidence
kind: atomic
phase: verification
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: true
delegates_to: []
output_schema: artifact_provenance_summary.v1
---

# Artifact Provenance

## When to use

Use this skill whenever generated files, datasets, checkpoints, model caches, benchmark outputs, reports, or videos need classification before they are cited, committed, archived, or discarded.

## Classification

- `discard`: disposable local output with no handoff value.
- `ignored-cache`: local cache required for reruns but not durable evidence.
- `tracked-compact-evidence`: small reviewable evidence under `docs/context/evidence/` or another tracked manifest.
- `durable-required`: downstream work needs the artifact, but it has not been promoted.
- `durable-promoted`: stored in W&B, release storage, registry, or another durable source with retrieval metadata.
- `non-evidence-local-only`: useful only for local debugging and not cited as proof.

## Workflow

1. List generated paths, including ignored `output/` files when relevant.
2. Record commit SHA, command, config, seed set, and artifact root.
3. Compute checksums for compact evidence or durable manifests when practical.
4. Promote only small reviewable summaries to git; keep raw logs, videos, checkpoints, and caches out of git unless there is a deliberate fixture reason.
5. Fail closed when a future workflow depends on local `output/` without a durable source.

## Guardrails

- Apply the fail-closed benchmark policy when artifacts or staged data affect benchmark evidence.
- Do not mirror `output/` wholesale into git.
- Do not add Git LFS as the default answer for generated benchmark artifacts.
- Do not claim a local artifact is durable until a retrieval pointer and metadata exist.

## Output

Use `artifact_provenance_summary.v1`.
