# Reproducing Release 0.0.2

This document provides step-by-step instructions to reproduce the canonical paper handoff results for benchmark release `0.0.2`.

## Overview and Key Differences

- **Release Tag (`0.0.2`)**: Points to commit `cbeaca610` (three commits earlier than the campaign completion). This commit has the core simulation logic and code representation.
- **Campaign Commit (`f7ebdcae2375d085e925213197a75a386e26a79c`)**: Contains the final generated publication manifest, checksums, and the scoped release manifest that exists outside the tag `0.0.2` tree.
- **Parity Test**: The regeneration parity test (`tests/benchmark/test_paper_results_handoff.py::test_canonical_handoff_matches_durable_release_campaign_table`) lives on `main` (not in the tag tree). Therefore, reproduction must be run from a fresh clone of `main` using the downloaded release `0.0.2` bundle.

> [!NOTE]
> This reproduction procedure does not modify the immutable release artifact on GitHub or Zenodo.

## Scoped Manifest Path

The exact scoped release manifest is located at:
`configs/benchmarks/releases/paper_experiment_matrix_7planners_v1_release_v0_0_2_scoped.yaml`

## Release Archive & Pinned SHA-256

- **Archive Name**: `paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz`
- **Expected SHA-256**: `64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90`

## Fresh Clone Reproduction Commands

Run the following commands from a fresh checkout of the `main` branch to download the bundle, verify its checksum, and run the regeneration parity test:

```bash
# Clone the repository and navigate to the root
git clone https://github.com/ll7/robot_sf_ll7.git
cd robot_sf_ll7

# Setup the virtual environment and sync dependencies
uv sync --all-extras

# Download the release bundle
mkdir -p output/benchmark_release_0_0_2
gh release download 0.0.2 \
  --pattern 'paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz' \
  --dir output/benchmark_release_0_0_2

# Verify the SHA-256 checksum of the downloaded bundle
sha256sum output/benchmark_release_0_0_2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz

# Run the regeneration parity test
ROBOT_SF_PAPER_HANDOFF_BUNDLE=output/benchmark_release_0_0_2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz \
  uv run pytest tests/benchmark/test_paper_results_handoff.py::test_canonical_handoff_matches_durable_release_campaign_table -q
```
