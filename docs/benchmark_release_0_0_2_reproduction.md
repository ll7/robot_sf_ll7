# Reproducing Release 0.0.2

This document provides step-by-step instructions to verify and reproduce the canonical paper
handoff results for benchmark release `0.0.2`.

## Tooling checkout requirement

The immutable `0.0.2` tag contains the released source and bundle pointer, but it does not contain
the checksum manifest, checksum verifier, cold-start report entry point, or scoped replay config.
Use a separate tooling checkout at `0.0.3` or a newer `main` commit. `0.0.3` is the minimum
documented tooling version because it contains all of those paths.

Do not run the verifier from a checkout of tag `0.0.2` and interpret a missing-file error as a
release failure. The release artifact remains `0.0.2`; only the reproduction tooling comes from
`0.0.3` or `main`.

## Overview and Key Differences

- **Release Tag (`0.0.2`)**: Points to commit `cbeaca610` (three commits earlier than the campaign completion). This commit has the core simulation logic and code representation.
- **Campaign Commit (`f7ebdcae2375d085e925213197a75a386e26a79c`)**: Contains the final generated publication manifest, checksums, and the scoped release manifest that exists outside the tag `0.0.2` tree.
- **Parity Test**: The regeneration parity test (`tests/benchmark/test_paper_results_handoff.py::test_canonical_handoff_matches_durable_release_campaign_table`) lives in the tooling checkout (minimum `0.0.3`, or `main`), not in the tag tree. Therefore, reproduction must use the `0.0.2` bundle with a `0.0.3`/`main` tooling checkout.

> [!NOTE]
> This reproduction procedure does not modify the immutable release artifact on GitHub or Zenodo.

## Scoped Manifest Path

The exact scoped release manifest is located at:
`configs/benchmarks/releases/paper_experiment_matrix_7planners_v1_release_v0_0_2_scoped.yaml`

## Release Archive & Pinned SHA-256

- **Archive Name**: `paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz`
- **Expected SHA-256**: `64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90`

## Fresh Clone Checksum Verification

Run the following commands from a fresh checkout of the `0.0.3` tooling tag (or `main`) to
download the bundle and verify its checksum:

```bash
# Clone the minimum supported tooling checkout and navigate to the root
git clone --branch 0.0.3 https://github.com/ll7/robot_sf_ll7.git robot_sf_ll7-repro
cd robot_sf_ll7-repro

# For current tooling, use the default branch instead:
# git clone https://github.com/ll7/robot_sf_ll7.git robot_sf_ll7-repro
# cd robot_sf_ll7-repro

# Setup the virtual environment and sync dependencies
uv sync --all-extras

# Download the release bundle
mkdir -p output/benchmark_release_0_0_2
gh release download 0.0.2 \
  --pattern 'paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz' \
  --dir output/benchmark_release_0_0_2

# Verify the SHA-256 checksum of the downloaded bundle
sha256sum output/benchmark_release_0_0_2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz

# Verify the bundle and its embedded checksums against the tracked manifest
uv run python scripts/repro/verify_release_checksums.py --tag 0.0.2

# Execute the full cold-start reproduction report (clone + checksums + build + numeric subset replay + numeric comparison)
uv run python scripts/repro/cold_start_reproduction_report.py --tag 0.0.2
```

The checksum command establishes that the published archive and its embedded artifacts match the
tracked release manifest. The full cold-start reproduction report (`cold_start_reproduction_report.py`)
additionally executes the frozen benchmark subset (`francis2023_blind_corner`, seed 111, all 7 planners)
in `run` mode and compares the resulting numeric outcomes against the published release contract using
the already-published reproducibility bounds: near misses use the documented absolute `0.31` bound,
and SNQI uses the documented maximum near-miss propagation weight `0.3082583`. Scenario, seed,
outcome status, execution mode, algorithm-metadata status, source/config hashes, success,
collisions, and normalized time-to-goal must match exactly. Preflight mode alone is insufficient
and cannot produce a `run_subset=pass` verdict.

The runner resolves its output root to an absolute path before launching the release-tag child and
binds comparison to the explicit `campaign_root` returned by a parseable `run` payload. That root
must be a newly created direct child of `<output-root>/subset_run`; the output root itself, nested
descendants, and paths that existed before launch are rejected. It never selects an arbitrary or
stale directory from the output tree. The report's `lockfile_sha256` identifies the `uv.lock` in
the executed release clone; `tooling_lockfile_sha256` separately identifies the checkout that
provided this report runner. Missing or duplicate rows, missing metrics, wrong identity/provenance,
fallback/degraded execution, malformed runner output, or tolerance breaches fail closed. The older
report under
`docs/context/evidence/issue_5366_cold_start_reproduction_2026-07-12/` remains immutable historical
preflight evidence; changing the current manifest does not rewrite its timestamp, steps, commit, or
recorded manifest hash.

## Optional parity check

After checksum verification, the regeneration parity test can be run from the same tooling
checkout with the downloaded bundle:

```bash

# Run the regeneration parity test
ROBOT_SF_PAPER_HANDOFF_BUNDLE=output/benchmark_release_0_0_2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz \
  uv run pytest tests/benchmark/test_paper_results_handoff.py::test_canonical_handoff_matches_durable_release_campaign_table -q
```
