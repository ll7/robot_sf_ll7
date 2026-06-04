# Issue #2214 Hot-Path Synthesis Evidence

Date: 2026-06-04

This directory contains a compact summary for the issue #2214 simulator hot-path
optimization synthesis. It intentionally stores only small, reviewable derived
evidence instead of raw local outputs.

## Contents

- `summary.json` - baseline/current commits, measured diagnostic surfaces,
  excluded measurements, prior worker-scaling comparison, and interpretation.

## Provenance

- Baseline commit: `5eead086fceac0bbdd00bb10ec133a612dfc5b25`
- Issue-specified optimization-family tip: `c125d5ae`
- Measured current commit: `5bd87d58d4869e8420943301e45de1a8dc6513a1`
- Date measured: 2026-06-04

The raw JSON and pytest artifacts remained worktree-local because they are
regenerable diagnostic outputs. The durable claim surface is the derived summary
plus `docs/context/issue_2214_hot_path_synthesis.md`.
