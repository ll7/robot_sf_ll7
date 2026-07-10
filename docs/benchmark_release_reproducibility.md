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
When both campaigns include scenario and scenario-family breakdown CSVs, the
comparison JSON also includes those row-level deltas. `unfinished_mean` is a
derived route-incomplete metric (`1 - success_mean`), not raw timeout
attribution.

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

**Inherently non-deterministic (source identified and bounded, issue #5140):**

- `near_misses_mean` varies for all planners (±0.01–0.31 per run in the
  2026-04-10 full-release measurement). The source is now identified and
  quantified rather than asserted:
  - **The metric path is provably deterministic.** The near-miss reduction
    (`_compute_robot_ped_distance_summary`) is pure NumPy (`np.linalg.norm`
    distance matrix → `min` over pedestrians → `count_nonzero` against the
    0.5 m surface-clearance band). It contains no Numba kernel, no parallel
    reduction, and no `fastmath`, so it is bit-deterministic for any fixed input
    trajectory set. This *disproves* the "parallel reduction order / JIT fastmath
    / thread scheduling in the Numba kernels" hypothesis **for the metric path**;
    see `robot_sf/benchmark/near_miss_determinism.py::metric_path_is_deterministic`.
  - **The residual nondeterminism is upstream, in the pedestrian dynamics.**
    `pysocialforce.forces` computes per-agent forces with `@njit(fastmath=True)`;
    the resulting trajectories can cross the 0.5 m clearance threshold at
    knife-edge timesteps, so a sub-ULP trajectory difference can flip a
    near-miss count. The residual is *machine-/compiler-conditional*, not a
    property of the metric definition.
  - **Tolerance quantification.** `measure_exact_repeat_nondeterminism` runs `N`
    exact-repeat episodes and reports the per-metric maximum deviation. On the
    supported test environment, the committed low-density smoke scenario has
    exact-repeat `near_misses` deviation **0.0** (5 repeats, `horizon=30`) —
    i.e. it is bit-identical for that scenario on one machine. The ±0.01–0.31
    figure from the full release reflects *cross-run* divergence surfaced at
    knife-edge crossings under the full campaign pipeline; it is not a
    cross-machine guarantee. A broader measurement must be recorded as a
    reproducible, durable campaign artifact before this contract is generalized.
  - **SNQI propagation bound.** The near-miss SNQI term is
    `-w_near * clamp((nm - med) / (p95 - med), 0, 1)` with
    `w_near = 0.3082583` (camera-ready v3). A raw near-miss tolerance `delta`
    propagates to at most `w_near * delta / (p95 - med)` in the linear region,
    capped at `w_near ≈ 0.31` by the `[0,1]` clamp. Compute it with
    `snqi_near_miss_propagation_bound`.

**Interpretation:** The benchmark's primary outcome claims (success, collisions)
are rerun-stable for 5/7 planners and within a 1-episode tolerance for the
remaining 2. `near_misses_mean` should not be cited as a precision metric in
publication tables — report it with an explicit tolerance (measured via
`measure_exact_repeat_nondeterminism`) or omit it from primary claims. SNQI
consumers should propagate the measured near-miss tolerance through
`snqi_near_miss_propagation_bound` before claiming a composite precision.

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

For the current scoped seven-planner paper release, use the tracked publication snapshot:

- `docs/experiments/publication/20260414_benchmark_release_0_0_2/summary.md`
- `docs/experiments/publication/20260414_benchmark_release_0_0_2/release_metadata.json`
- `docs/benchmark_release_0_0_2_reproduction.md` - Dedicated copy-paste procedure for reproducing release 0.0.2 results

> [!NOTE]
> **Release 0.0.2 Note**: The annotated tag `0.0.2` tree references the generic `paper_experiment_matrix_v1_release_v0_1.yaml` manifest. The actual scoped manifest (`configs/benchmarks/releases/paper_experiment_matrix_7planners_v1_release_v0_0_2_scoped.yaml`) and the parity test logic live on the `main` branch, not in the tag tree. To reproduce release 0.0.2, follow the dedicated [Release 0.0.2 Reproduction Note](benchmark_release_0_0_2_reproduction.md).


Durable endpoints:

- Release: `https://github.com/ll7/robot_sf_ll7/releases/tag/0.0.2`
- DOI: `https://doi.org/10.5281/zenodo.19563812`
- Archive:
  `https://github.com/ll7/robot_sf_ll7/releases/download/0.0.2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz`

Release `0.0.2` publishes the publication manifest, checksums, and SNQI diagnostics inside the
archive rather than as separate release assets. A fresh checkout can recover them with:

```bash
mkdir -p output/benchmark_release_0_0_2
gh release download 0.0.2 \
  --pattern 'paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz' \
  --dir output/benchmark_release_0_0_2
sha256sum output/benchmark_release_0_0_2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz
tar -tzf output/benchmark_release_0_0_2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz \
  | rg 'publication_manifest.json|checksums.sha256|snqi_diagnostics\.(json|md)'
```

The expected archive SHA-256 is:
`64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90`.

Primary artifact locations:

- `output/benchmarks/camera_ready/<campaign_id>/`
- `output/benchmarks/publication/<bundle_name>/`
- `output/benchmarks/publication/<bundle_name>.tar.gz`

These paths are local generation outputs. A paper-facing handoff must additionally record a durable
release asset, DOI, or artifact-store pointer for the archive, checksums, publication manifest, and
required diagnostic reports. Do not treat the local `output/` paths above as recoverable evidence in
a fresh checkout unless they are paired with such a durable pointer.

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

## Benchmark Docker Reproduction Path

For a fresh headless container smoke that verifies the benchmark CLI and artifact-writing surfaces
without requiring a local Python setup, use the pinned Docker path:

```bash
scripts/repro/run_benchmark_docker_smoke.sh
```

The Docker smoke is documented in `docs/benchmark_docker_repro.md`. It builds
`docker/benchmark-repro.Dockerfile`, runs the small
`configs/scenarios/planner_sanity_matrix_v1.yaml` slice, and writes inspectable artifacts under
`output/docker_repro/benchmark_bundle_smoke/`.

This Docker path is intentionally narrower than the reduced release manifest above: it is a
containerized environment and artifact smoke, not a replacement for full release reproduction or
paper-facing campaign validation.
