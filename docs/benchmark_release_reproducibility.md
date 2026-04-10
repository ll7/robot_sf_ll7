# Benchmark Release Reproducibility

This guide explains how to reproduce a benchmark release artifact set from a
tagged code state.

## Canonical Inputs

Canonical release manifest:

- `configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml`

Canonical campaign config:

- `configs/benchmarks/paper_experiment_matrix_v1.yaml`

That canonical release config runs with `workers: 1` so the frozen release path
does not depend on process-pool scheduling for its published metrics.

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

When comparing two frozen release reruns, use the camera-ready campaign
comparison helper:

```bash
uv run python scripts/tools/compare_camera_ready_campaigns.py \
  --base-campaign-root output/benchmarks/camera_ready/<base_campaign_id> \
  --candidate-campaign-root output/benchmarks/camera_ready/<candidate_campaign_id> \
  --output-json output/camera_ready_compare.json \
  --output-md output/camera_ready_compare.md
```

Pass `--require-identical` only when verifying tooling correctness, not as a
release acceptance gate — the benchmark is outcome-stable but not bit-exact
(see [Reproducibility Contract](#reproducibility-contract) below).

## Reproducibility Contract

Empirically verified by running the full frozen release twice under identical
conditions (same commit, same manifest, `workers: 1`) on 2026-04-10:

**Stable across reruns (primary paper metrics):**

| Planner | `success_mean` | `collisions_mean` |
|---|---|---|
| `goal` | exact | exact |
| `ppo` | exact | exact |
| `sacadrl` | exact | exact |
| `social_force` | exact | exact |
| `socnav_sampling` | exact | exact |

**Borderline (1-episode outcome flip observed):**

| Planner | `success_mean` delta | `collisions_mean` delta |
|---|---|---|
| `orca` | ±0.0071 (1/141 episodes) | ±0.0071 |
| `prediction_planner` | ±0.0071 (1/141 episodes) | ±0.0071 |

**Inherently non-deterministic:**

- `near_misses_mean` varies for all planners (±0.01–0.31 per run). This is
  proximity-threshold nondeterminism in the simulation physics and cannot be
  eliminated without fixing the underlying pedestrian dynamics RNG.

**Interpretation:** The benchmark's primary outcome claims (success, collisions)
are rerun-stable for 5/7 planners and within a 1-episode tolerance for the
remaining 2. `near_misses_mean` should not be cited as a precision metric in
publication tables — report it with an explicit tolerance or omit it from
primary claims.

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
