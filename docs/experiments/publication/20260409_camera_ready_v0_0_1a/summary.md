# Camera-Ready Publication Snapshot

Date: 2026-04-09

## Purpose

This tracked snapshot records the benchmark publication result for the public prerelease
`camera-ready-v0.0.1a` without committing the raw benchmark bundle under `output/`.

The immutable publication artifacts live on GitHub Releases and Zenodo. This directory keeps a
small, reviewable record in git for branch history, manuscript cross-reference, and publication
audit.

## Public Endpoints

- Release: `https://github.com/ll7/robot_sf_ll7/releases/tag/camera-ready-v0.0.1a`
- DOI: `https://doi.org/10.5281/zenodo.19482026`
- Bundle archive:
  `https://github.com/ll7/robot_sf_ll7/releases/download/camera-ready-v0.0.1a/paper_experiment_matrix_v1_release_rehearsal_latest_20260409_20260409_120154_publication_bundle.tar.gz`
- Manifest:
  `https://github.com/ll7/robot_sf_ll7/releases/download/camera-ready-v0.0.1a/publication_manifest.json`
- Checksums:
  `https://github.com/ll7/robot_sf_ll7/releases/download/camera-ready-v0.0.1a/checksums.sha256`

## Frozen Source

- Campaign id:
  `paper_experiment_matrix_v1_release_rehearsal_latest_20260409_20260409_120154`
- Repository commit:
  `a60119aa10ed677a55a8b6b69264d4aeabfdb811`
- Scenario matrix:
  `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed policy:
  `eval` seed set resolved to `111,112,113`
- Episodes:
  `987` total, `141` per planner
- Runtime:
  `728.73 s`
- Benchmark success:
  `true`
- SNQI contract status:
  `warn`

## Headline Reading

- `ppo` has the highest success in the released run at `0.2411`, but with materially higher
  collision exposure (`0.1135`) than `orca`.
- `orca` is the cleanest paper-facing comparator in this rerun with `0.0284` collisions and
  `0.1844` success.
- `goal` remains a control-only row with near-zero success (`0.0142`) and nontrivial collisions
  (`0.2411`).

Confidence intervals in the companion CSV/JSON are the benchmark-owned handoff values computed with
`bootstrap_mean_over_seed_means` at confidence `0.95`, `400` bootstrap samples, and seed `123`.

## Tracked Companion Files

- `release_metadata.json`: compact release, asset, campaign, and publication metadata
- `paper_results_handoff_snapshot.csv`: tracked copy of the interval-inclusive planner rows
