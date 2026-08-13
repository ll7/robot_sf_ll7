# Issue #6792 Chapter 7 admission execution packet

## Goal

Add a fail-closed, digest-bound admission receipt for the blocked Chapter 7 evidence package.
The receipt must combine the immutable package manifest, the trusted source registry, and the
author's exact approval without changing the package payload.

## Authority and boundaries

- Task: `RSF-6792-CH7-ADMISSION-01`
- Issue: https://github.com/ll7/robot_sf_ll7/issues/6792
- Related package PR: https://github.com/ll7/robot_sf_ll7/pull/6995
- Base: current `origin/main` at packet creation; implementation uses an isolated worktree.
- Allowed paths: `robot_sf/benchmark/schemas/ch7-evidence-admission.v1.json`,
  `scripts/analysis/verify_ch7_evidence_admission.py`, focused tests, this packet, and the
  repository-controlled source registry only after explicit author approval.
- Forbidden: raw traces, release archives, package payload mutation, new simulation, causal or
  matched-comparison language, dissertation edits, and source-registry promotion before approval.

## Frozen inputs

- Approved source `SHA256SUMS`: `011c644bac469a1ce6255ddb8731c53c84bd310887759174f4c734b54d6bb543`.
- Approved release archive: `3cfefaaa39aab6cae541cece9573848a7e0afc5e1d9e4c9a7bbf48df2330b1a7`.
- #6814 compact packet: `44360d5da575233131ac8e93c25a0dd539d980a2c8a0146651017d686c45dadb`.
- #6814 compact `SHA256SUMS`: `59ef90567c2eba5ef1f8431bc19e0962a9ddceec15ac81afce0b360c6ecac3b7`.
- Current package `SHA256SUMS`: `6807fdc9275133365812c8f51f51e057da6054f8dcaf77cb5fa8a32b08c4a87f`.
- Package payload: 21 primary files; raw traces and release archive are excluded.

## Required checks

- Schema self-check and fail-closed negative fixtures for every digest and authority binding.
- Focused admission-contract tests.
- Ruff and format checks for touched Python.
- `git diff --check`.
- `scripts/dev/evidence_registry_ratchet.py --check`.
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` before publication.
- Independent exact-head review with the source/package hashes and this packet.

## Stop conditions

- Any frozen package byte or SHA256SUMS entry changes.
- The source registry is modified without an explicit author approval comment.
- An approval receipt cannot bind the package, source, release, compact packet, registry, and scope.
- Any test or review promotes a typed-unavailable trajectory into a matched, causal, or
  counterfactual claim.
- The requested change crosses into dissertation manuscript work; that belongs to diss #698 and a
  separate contract.
