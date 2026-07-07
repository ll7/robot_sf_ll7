# Benchmark Release 0.0.2 Publication Snapshot

Date: 2026-05-09

## Purpose

This tracked snapshot records the durable publication pointer for benchmark release `0.0.2`
without committing the raw publication bundle under `output/`.

Release `0.0.2` is the scoped seven-planner benchmark release. It excludes `socnav_bench` because
the licensed SocNavBench dataset assets were not staged for the release run. The scope decision is
documented in
[`docs/context/benchmark_release_0_0_2_scoped_rationale.md`](../../../context/benchmark_release_0_0_2_scoped_rationale.md).

> [!TIP]
> For instructions on reproducing and verifying release 0.0.2, see the dedicated [Release 0.0.2 Reproduction Note](../../../benchmark_release_0_0_2_reproduction.md).


## Public Endpoints

- Release: `https://github.com/ll7/robot_sf_ll7/releases/tag/0.0.2`
- DOI: `https://doi.org/10.5281/zenodo.19563812`
- Bundle archive:
  `https://github.com/ll7/robot_sf_ll7/releases/download/0.0.2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz`

Unlike the older `camera-ready-v0.0.1a` prerelease, release `0.0.2` publishes a single durable
bundle asset. The publication manifest, checksums, and SNQI diagnostics are inside that archive:

- `publication_manifest.json`
- `checksums.sha256`
- `payload/reports/snqi_diagnostics.json`
- `payload/reports/snqi_diagnostics.md`

## Frozen Source

- Campaign id:
  `paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316`
- Repository commit:
  `f7ebdcae2375d085e925213197a75a386e26a79c`
- Release manifest:
  `configs/benchmarks/releases/paper_experiment_matrix_7planners_v1_release_v0_0_2_scoped.yaml`
- Canonical campaign config:
  `configs/benchmarks/paper_experiment_matrix_7planners_v1.yaml`
- Scenario matrix:
  `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed policy:
  `eval` seed set resolved to `111,112,113`
- Episodes:
  `987` total, `141` per planner
- Runtime:
  `681.34 s`
- Benchmark success:
  `true`
- SNQI contract status:
  `pass`

## Archive Verification

Verified on 2026-05-09 with:

```bash
gh release download 0.0.2 \
  --pattern 'paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz' \
  --dir output/issue_1062_release_probe
sha256sum output/issue_1062_release_probe/*.tar.gz
tar -tzf output/issue_1062_release_probe/*.tar.gz \
  | rg 'publication_manifest.json|checksums.sha256|snqi_diagnostics\.(json|md)'
```

The downloaded archive SHA-256 is:

`64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90`

## Recovery Command

A fresh checkout can recover the publication evidence with:

```bash
mkdir -p output/benchmark_release_0_0_2
gh release download 0.0.2 \
  --pattern 'paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz' \
  --dir output/benchmark_release_0_0_2
tar -xzf output/benchmark_release_0_0_2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz \
  -C output/benchmark_release_0_0_2
```

The paper Results handoff parity test uses this same release source. It accepts either the
downloaded archive or the extracted bundle through `ROBOT_SF_PAPER_HANDOFF_BUNDLE`, and it also
auto-discovers the documented paths under `output/benchmark_release_0_0_2`:

```bash
ROBOT_SF_PAPER_HANDOFF_BUNDLE=output/benchmark_release_0_0_2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz \
  uv run --active pytest tests/benchmark/test_paper_results_handoff.py::test_canonical_handoff_matches_durable_release_campaign_table -q -rs
```

Before using the bundle, the test verifies the archive SHA-256 from `release_metadata.json` and the
tracked embedded artifact checksums for `publication_manifest.json`, `checksums.sha256`, and the
SNQI diagnostics.

## Tracked Companion Files

- `release_metadata.json`: compact release, asset, campaign, and embedded-artifact metadata.
