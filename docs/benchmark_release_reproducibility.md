# Benchmark Release Reproducibility

This guide explains how to reproduce a benchmark release artifact set from a
tagged code state.

## Canonical Inputs

Canonical release manifest:

- `configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml`

Canonical campaign config:

- `configs/benchmarks/paper_experiment_matrix_v1.yaml`

Reduced smoke manifest for validation:

- `configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml`

## Reproduce From a Tag

1. Check out the repository tag that corresponds to the release.
2. Install dependencies:

```bash
uv sync --all-extras
```

3. Run release preflight:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml \
  --mode preflight
```

4. Run the release:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml \
  --label repro
```

## What Is Frozen

Comparable benchmark releases must keep these surfaces stable:

- canonical campaign config
- scenario matrix
- seed policy
- planner set and planner groups
- kinematics contract
- SNQI assets
- required artifact bundle contents

If one of those changes materially, the release is no longer comparable and
requires a major benchmark release increment.

## What Counts As Comparable vs Non-Comparable

Comparable:

- provenance enrichment
- stricter validation
- docs and release workflow improvements
- publication-bundle handling fixes that do not change benchmark metrics

Non-comparable:

- scenario additions/removals
- seed-policy changes
- planner-set changes
- kinematics changes
- metric-contract or SNQI normalization changes

## Release Artifact

The benchmark release artifact is the publication bundle generated from the
release workflow, not the raw source checkout alone.

Primary artifact locations:

- `output/benchmarks/camera_ready/<campaign_id>/`
- `output/benchmarks/publication/<bundle_name>/`
- `output/benchmarks/publication/<bundle_name>.tar.gz`

## Citation Surface

Repository-level software citation is defined in:

- `CITATION.cff`

The release manifest also records:

- repository URL
- release tag
- DOI placeholder or DOI

## Smoke Validation

For CI and local release-tool validation, use the reduced smoke manifest:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml
```

This preserves the release contract shape while avoiding a heavyweight full
benchmark run.
