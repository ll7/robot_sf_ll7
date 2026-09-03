# Benchmark Release Reproducibility

This guide explains how to reproduce a benchmark release artifact set from a
tagged code state.

For the current approved benchmark-data release, use the 14-arm S30/H600
campaign and keep the software release lane separate. The old seven-planner/S3
paths below are retained only where they document historical release `0.0.2`;
they must not be used as the current release contract.

## Canonical Inputs

Current S30/H600 campaign config:

- `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml`

Bounded 14-arm runtime-smoke manifest:

- `configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml`

Publication-grade release manifest plus the frozen September 2026 erratum successor:

- `configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml`
- `configs/benchmarks/releases/benchmark_data_release_s30_h600_2026_09_erratum_1.json`
  (successor tag `paper-matrix-v2-h600-s30-2026-09-59577bad289dd692ba3580e1600c4a649ae27880-erratum.1`,
  version DOI `10.5281/zenodo.22265925`, concept DOI `10.5281/zenodo.22227034`,
  orchestration SHA `09f6b1beeff71b4ee30deae7cc2504c0c7310729`).
  Reproduce from a clean checkout at the orchestration SHA; a dirty tree voids the
  derivation receipt.

The smoke keeps `workers: 1`, `horizon: 600`, differential-drive kinematics,
one scenario, and seed `111`. It checks runtime compatibility only; it is not
full benchmark evidence and does not authorize planner ranking.

## Reproduce From a Tag

1. Check out the repository tag that corresponds to the release.
2. Install dependencies:

```bash
uv sync --all-extras
```

3. Run release preflight:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml \
  --mode preflight
```

4. Run the release:

```bash
uv run python scripts/benchmark/preflight_campaign_checkpoints.py \
  --config configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke.yaml \
  --stage \
  --report-path output/release/checkpoints/runtime_smoke_staging_receipt.json
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml \
  --label runtime-smoke \
  --checkpoint-receipt output/release/checkpoints/runtime_smoke_staging_receipt.json
```

For the full benchmark-data release, run the separately reviewed S30/H600
manifest paired with the campaign config. Do not substitute the historical v1
seven-planner/S3 manifest, and do not publish the smoke output as a full result.

## What Is Frozen

Comparable benchmark releases must keep these surfaces stable:

- canonical campaign config
- scenario matrix
- seed policy
- planner set and planner groups
- kinematics contract
- SNQI assets
- required artifact bundle contents

For the current release, the frozen identity additionally includes 14 arms,
48 scenarios, 30 seeds (`paper_eval_s30`), H600, and differential-drive
kinematics. SNQI remains advisory/no-ranking even when its assets are present;
report raw and component metrics separately if calibration does not support a
composite interpretation.

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

## Historical v1 Reproducibility Contract

The empirical rerun table in this section records the historical seven-planner
v1/S3 contract. It remains useful for reproducing release `0.0.2`, but it is
not evidence for the current S30/H600 benchmark-data release and does not
authorize SNQI ranking.

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
    knife-edge timesteps, so a sub-unit-in-the-last-place (ULP) trajectory
    difference can flip a
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

## Cross-Context Reproducibility (issue #5816)

Trace re-executions in this benchmark are **bit-reproducible within a single
execution context** (identical `(commit, config, node, thread env)` → sha-equal
step traces) but **not trajectory-stable across execution contexts**.

- **What is stable across contexts:** the *outcomes* are attractor-stable. In the
  2026-07-16 doorway-butterfly re-export (pinned commit `a307ef2`), 28/30 seeds
  agreed with the release; only 2 knife-edge seeds (128, 130) flipped, and the
  flipped seeds were consistent per context.
- **What diverges across contexts:** the *trajectories* diverge for the same seed.
  In that re-export, seed 114 produced 62 near-misses on the release box
  (`workers=32`) vs 78 on a Slurm node (`workers=1`) vs 37 on the login node
  (`workers=1`). There was no code delta — the commit was pinned on both sides.
- **Mechanism:** ULP-level float differences from CPU, Basic Linear Algebra
  Subprograms (BLAS), and threading context are chaotically amplified in
  contact-rich scenarios (the same upstream
  `pysocialforce.forces` `@njit(fastmath=True)` dynamics described above).

**Rules for comparing runs:**

- Never compare *trajectories* across executions without labeling the execution
  context. Always label any cross-context comparison with hostname, CPU model,
  and thread environment.
- Compare *outcomes* (success/collision counts), not per-step traces, when the
  execution context differs.
- For trace-level re-exports, re-run in the *same* execution context that produced
  the baseline, or expect knife-edge seeds to flip.

**What the pipeline now pins and records:**

- The camera-ready runner (`scripts/tools/run_camera_ready_benchmark.py`) pins
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to `1`,
  overriding inherited values for this reproducibility-focused entry point.
  This makes re-executions at least thread-deterministic.
- Each campaign `run_meta.json` now carries an `execution_context` block with
  `hostname`, `cpu_model` (from `/proc/cpuinfo`), and the resolved `thread_env`
  mapping, so divergent execution contexts are detectable after the fact.

**Per-Context Determinism Smoke Test (issue #6126):**

- Per-context step trace bit-reproducibility is guarded in CI by the fixed-episode smoke test
  (`scripts/validation/run_per_context_determinism_smoke.py` and
  `tests/benchmark/test_per_context_determinism_smoke.py`).
- The test executes a fixed scenario/planner/seed/horizon episode twice in-process with forced
  numerical thread pinning (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`)
  and compares canonical step traces using `robot_sf.benchmark.step_trace_comparator`.
- This smoke test verifies per-context determinism within a single execution context. It explicitly
  does **not** claim bit-identical step traces across different CPU microarchitectures, OS versions,
  or BLAS libraries (preserving the empirical 2/30 cross-context divergence result from #5817).

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

The following is the historical seven-planner publication snapshot for release
`0.0.2`, not the current S30/H600 benchmark-data release:

- `docs/experiments/publication/20260414_benchmark_release_0_0_2/summary.md`
- `docs/experiments/publication/20260414_benchmark_release_0_0_2/release_metadata.json`
- `docs/benchmark_release_0_0_2_reproduction.md` - Dedicated copy-paste procedure for reproducing release 0.0.2 results

> [!NOTE]
> **Release 0.0.2 tooling boundary**: The immutable tag `0.0.2` does not contain the checksum
> manifest, verifier, cold-start report entry point, scoped manifest, or parity test logic. Use a
> tooling checkout at tag `0.0.3` or a newer `main` commit, then follow the dedicated [Release
> 0.0.2 Reproduction Note](benchmark_release_0_0_2_reproduction.md). A checksum pass verifies the
> published archive and embedded artifacts; it is not an independent numeric subset replay.


Historical durable endpoints (release `0.0.2` only; superseded for S30/H600 by the
September 2026 erratum successor, version DOI `10.5281/zenodo.22265925`):

- Release: `https://github.com/ll7/robot_sf_ll7/releases/tag/0.0.2`
- DOI: `https://doi.org/10.5281/zenodo.19563812`
- Archive:
  `https://github.com/ll7/robot_sf_ll7/releases/download/0.0.2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz`

Release `0.0.2` publishes the publication manifest, checksums, and historical SNQI diagnostics inside the
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

For CI and local release-tool validation of the current contract, use the
14-arm reduced smoke manifest:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml
```

This preserves the current S30/H600 release contract shape while avoiding a
heavyweight full benchmark run. A passing smoke is runtime evidence only: it
does not establish a benchmark result, SNQI ranking, or a software release.

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
