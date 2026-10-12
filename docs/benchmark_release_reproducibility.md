# Benchmark Release Reproducibility

This guide identifies the published benchmark-data release and the inputs needed
to verify or reconstruct its artifact set. The current release is **0.0.8, main
track only**, with 14 arms × 48 scenarios × 30 evaluation seeds = **20,160
rows**, differential-drive kinematics and authored per-scenario budgets. The
August campaign and September erratum are historical releases.

## Canonical Inputs

The immutable scientific source is
`373dbfde4f39667cf9e8732dabe7df5118bdeab1`, tagged
`paper-matrix-v2-h600-s30-2026-10-373dbfde4f39667cf9e8732dabe7df5118bdeab1`.
Use that tagged source for the following inputs:

- Campaign: `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml`.
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml`.
- Horizon schedule: `configs/benchmarks/horizon_schedules/release_0_0_8_authored_v1.yaml`.
- Actual resolved release manifest: `payload/release/release_manifest.resolved.json`
  inside the published archive. This is the concrete publication identity;
  `configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml`
  is the reusable template, not a substitute for that resolved manifest.
- SNQI-v2 diagnostics and input path/digest declarations:
  `payload/reports/snqi_v2_diagnostics.json` inside the archive.

The `h600` strings in the immutable tag and filenames are historical names.
The declared schedule has the following budgets, at `dt = 0.1 s`:

| Horizon steps | Scenario count | Episode budget |
| --- | --- | --- |
| 400 | 24 | 40 s |
| 500 | 13 | 50 s |
| 600 | 9 | 60 s |
| 650 | 1 | 65 s |
| 700 | 1 | 70 s |

Thus 39 of 48 scenarios use a budget other than H600.
`classic_station_platform_medium` uses H650;
`classic_realworld_double_bottleneck_high` uses H700. Resolve each cell's limit
from the authored schedule; observed episode length is not its declared budget.

The [main-only scope amendment](release/0.0.8/main_only_scope_amendment_20261009.md)
excludes the separate three-width doorway track and defers it to 0.1.0. Its
incomplete diagnostic rows are outside the 20,160-row main denominator.

## Verify the Published Artifact

The [GitHub release](https://github.com/ll7/robot_sf_ll7/releases/tag/paper-matrix-v2-h600-s30-2026-10-373dbfde4f39667cf9e8732dabe7df5118bdeab1)
and [version DOI 10.5281/zenodo.23150472](https://doi.org/10.5281/zenodo.23150472)
identify the actual 0.0.8 publication. The concept DOI is
[10.5281/zenodo.23150471](https://doi.org/10.5281/zenodo.23150471).
The archive is
`issue9850_release_snqi_v2_0_0_8_373dbfde4f_r39_20261008_publication_bundle.tar.gz`;
its detached publication manifest and checksums are also release assets.

Download the immutable assets into a fresh verification directory:

```bash
GH_TOKEN="$(gh auth token)" gh release download \
  paper-matrix-v2-h600-s30-2026-10-373dbfde4f39667cf9e8732dabe7df5118bdeab1 \
  --repo ll7/robot_sf_ll7 \
  --dir output/release-008-cold-verification
cd output/release-008-cold-verification
tar -xzf issue9850_release_snqi_v2_0_0_8_373dbfde4f_r39_20261008_publication_bundle.tar.gz
cd issue9850_release_snqi_v2_0_0_8_373dbfde4f_r39_20261008_publication_bundle
sha256sum -c checksums.sha256
```

Checksum verification establishes retained-byte integrity. Numerical
reconstruction also requires the acquired calibration anchors identified by
path and digest in the diagnostics; the published 0.0.8 archive omits those
anchors. Preserve the original bundle and obtain matching input bytes under
separate custody. Future publication tooling enforces stricter completeness;
its verdict must not be used to rewrite a frozen release.

## Reproduction and Development Boundaries

Scientific reconstruction binds the tagged source, actual resolved manifest,
scenario schedule, planner/checkpoint identities and exact diagnostic inputs.
The software release lane remains separate. Evaluation seed identities are
recorded by the frozen release contract; any release replay requires its own
explicit authorization. Development validation uses only seeds 1001–1200 and
does not reproduce the frozen evaluation dataset.

Install the tagged source's dependencies with `uv sync --all-extras --frozen` in
an isolated checkout. Prepare and review development-only smoke inputs before
running a campaign. The historical `runtime_smoke_v0_5` manifest is not the
current 0.0.8 reproduction contract. A smoke proves runtime compatibility only;
full dataset reconstruction and scientific admission require the release gates.

## What Is Frozen

Comparable benchmark releases must keep these surfaces stable:

- canonical campaign config
- scenario matrix
- seed policy
- planner set and planner groups
- kinematics contract
- SNQI assets
- required artifact bundle contents

For 0.0.8, the identity additionally includes 14 arms,
48 scenarios, 30 release-evaluation seeds (`release_eval_0_0_8`; D-049), the authored
H400/H500/H600/H650/H700 schedule above, and differential-drive
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
not evidence for the current 0.0.8 authored-budget benchmark-data release and does not
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
`0.0.2`, not the current 0.0.8 authored-budget benchmark-data release:

- `docs/experiments/publication/20260414_benchmark_release_0_0_2/summary.md`
- `docs/experiments/publication/20260414_benchmark_release_0_0_2/release_metadata.json`
- `docs/benchmark_release_0_0_2_reproduction.md` - Dedicated copy-paste procedure for reproducing release 0.0.2 results

> [!NOTE]
> **Release 0.0.2 tooling boundary**: The immutable tag `0.0.2` does not contain the checksum
> manifest, verifier, cold-start report entry point, scoped manifest, or parity test logic. Use a
> tooling checkout at tag `0.0.3` or a newer `main` commit, then follow the dedicated [Release
> 0.0.2 Reproduction Note](benchmark_release_0_0_2_reproduction.md). A checksum pass verifies the
> published archive and embedded artifacts; it is not an independent numeric subset replay.


Historical durable endpoints (release `0.0.2` only; the current 0.0.8 endpoints
are listed above):

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

For local development, use a reviewed development-only configuration with seeds
1001–1200 and the authored budget for each selected scenario. The old
`runtime_smoke_v0_5` examples document an earlier release and are not a current
approved reproduction route. Smoke success is runtime evidence only and does
not establish the 0.0.8 benchmark result or admit SNQI ranking.

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
