# Benchmark Release Checklist

This checklist covers the approved S30/H600 benchmark-data release. It is a
different release lane from the Robot SF software/package release: the
benchmark-data tag identifies an immutable campaign contract, while the
software tag identifies installable source. Do not infer a package version from
the benchmark-data tag, or reuse a software-release DOI for benchmark data.

The current campaign contract is the 14-arm, differential-drive matrix in
`configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml`.
Its publication-grade manifest is
`configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml`, with fresh
concept DOI `10.5281/zenodo.22077447` and reserved version DOI
`10.5281/zenodo.22077448`.
The bounded one-scenario/one-seed preflight and runtime smoke is tracked by
`configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml`.
The separate fallback-prone hybrid stress gate is tracked by
`configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_hybrid_stress_smoke_v0_1.yaml`
and documented in [`benchmark_release_hybrid_stress_smoke.md`](./benchmark_release_hybrid_stress_smoke.md).
The smoke is execution evidence only: the Social Navigation Quality Index
(SNQI) remains advisory and has no planner-ranking authority.

## Before Running

- confirm the target branch/tag is the intended immutable code state
- confirm the approved S30/H600 full-release manifest is
  `configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml`; do not
  substitute the historical v1 seven-planner/S3 manifest
- confirm the bounded smoke manifest is correct:
  - `configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml`
- confirm the fallback-prone hybrid stress manifest is correct:
  - `configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_hybrid_stress_smoke_v0_1.yaml`
- confirm manifest hashes still match referenced config and assets
- confirm benchmark fallback policy is fail-closed for benchmark mode
- confirm a fresh Zenodo concept is reserved for the benchmark-data record; the
  current reservation is concept `10.5281/zenodo.22077447` and version
  `10.5281/zenodo.22077448`; verify both again after publication; the
  historical concepts `10.5281/zenodo.19482025` and
  `10.5281/zenodo.19563812` must not be reused
- confirm the dataset metadata is the tracked benchmark-specific file
  `configs/benchmarks/releases/benchmark_data_release_s30_h600_zenodo_metadata.json`;
  do not modify or reuse the root software-release `.zenodo.json`
- confirm SNQI is documented as advisory/no-ranking, including when calibration
  reports a warning
- classify smoke artifacts before handoff: raw episode files remain worktree-local
  ignored cache, while the manifest/config hashes and compact summary are the
  tracked provenance surface; record source commit, command, seed, and campaign
  root in the release evidence

## Preflight

Run:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml \
  --mode preflight
```

Verify:

- manifest validation status is `valid`
- preflight artifacts were written
- matrix summary reflects all 14 planner arms, one smoke scenario, one seed,
  H600, and differential-drive kinematics
- smoke output is labeled `runtime-smoke`; it is not full benchmark evidence

## Runtime-Smoke Run Mode

Produce the required `release/release_result.json` with the canonical run-mode
command. Stage the smoke checkpoints first, then run at the exact release source
commit:

```bash
uv run python scripts/benchmark/preflight_campaign_checkpoints.py \
  --config configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke.yaml \
  --stage \
  --report-path output/release/checkpoints/runtime_smoke_staging_receipt.json
```

Require `submit_safe=true`, then run the 14-arm smoke with a deterministic
campaign id:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml \
  --mode run \
  --campaign-id issue7742_runtime_smoke_v0_2 \
  --checkpoint-receipt output/release/checkpoints/runtime_smoke_staging_receipt.json
```

The expected output is
`output/benchmarks/camera_ready/issue7742_runtime_smoke_v0_2/release/release_result.json`
(also referenced as `<smoke_id>` below). Runtime-smoke output is
release-admission evidence only and is **not** full benchmark evidence.

## Release Execution

First stage every referenced checkpoint into durable shared storage and persist
the exact admission receipt:

```bash
uv run python scripts/benchmark/preflight_campaign_checkpoints.py \
  --config configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml \
  --stage \
  --report-path output/release/checkpoints/staging_receipt.json
```

Require `submit_safe=true`. Before this v0.2 full-release launch, run the
canonical 14-arm runtime smoke at the exact release source commit and retain its
successful `release/release_result.json` (for example,
`output/benchmarks/camera_ready/<smoke_id>/release/release_result.json`). Also run
the hybrid stress gate at the same exact source commit and require zero fallback,
degraded, unavailable, failed, legacy emergency-stop, or all-candidates-rejected
markers. Native `*_protective_stop` and `static_protective_reorient` decisions
are not alternate-planner fallback; retain their explicit counts as
planner-outcome evidence and do not treat them as successful navigation.
Guarded PPO's exact `fallback_safe` label is likewise its declared Risk-DWA
shield intervention; best-effort, uncertainty, and generic fallback markers
remain forbidden. A benign one-cell runtime smoke does not replace this stress
gate. Then run:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml \
  --label release \
  --campaign-id issue7742_benchmark_data_release_s30_h600_20260822 \
  --checkpoint-receipt output/release/checkpoints/staging_receipt.json \
  --runtime-smoke-receipt output/benchmarks/camera_ready/<smoke_id>/release/release_result.json
```

That command is a fresh launch. If infrastructure interrupts it, an operator may
resume the same campaign id only with a fresh `benchmark-release-resume-receipt.v1`
passed through `--resume-receipt`. The receipt must classify a scheduler requeue,
node failure, walltime kill, cluster-filesystem interruption, or network interruption
and bind the unchanged source commit, campaign-config checksum, staged-checkpoint
receipt checksum, and prior campaign manifest. A code, config, dependency, or
checkpoint defect requires a corrected release commit and a fresh campaign id. The
single-node Slurm wrapper uses the same rule and does not blindly request `--requeue`.

Verify:

- process exit code is zero
- `benchmark_success` is `true`
- `release/release_manifest.resolved.json` exists
- `release/release_result.json` exists
- required campaign artifacts exist
- publication bundle archive, manifest, and checksums exist

From the extracted bundle root, verify the payload checksums with:

```bash
sha256sum -c checksums.sha256
```

The publication preflight also requires `release_result.json` and the rebuilt
`campaign_summary.json` to agree on status, validity, and cardinality. If episode
runtime commits differ from the publication commit, record
`provenance.commit_reconciliation` with the runtime commit list, publication
commit, and a plain-language explanation. Goal-reached plus timeout rows require
`provenance.goal_timeout_boundary` with timing evidence or an explicit exclusion
note.

## Publication

Before campaign submission, run the fail-closed release doctor against the
exact release worktree. Supply the private-ops packet and dissertation checkout
when they are available:

```bash
uv run robot-sf release doctor \
  --repo "$PWD" \
  --manifest configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml \
  --expected-release-sha <exact-release-sha> \
  --expected-base-sha cd831d7582c117ac9529065e7d1c60386933c92d \
  --tag paper-matrix-v2-h600-s30-2026-08-cd831d7582c1 \
  --checkpoint-receipt output/release/checkpoints/staging_receipt.json \
  --private-launch-packet <private-ops-launch-packet> \
  --dissertation <dissertation-worktree> \
  --token-file /home/luttkule/.config/robot-sf/zenodo.token
```

For diagnostic local validation of an exact receipt whose checkpoint paths belong to another
host, repeat `--checkpoint-path-map RECEIPT_PATH=LOCAL_PATH`. The source is matched as the exact
`resolved_path` string recorded in the receipt, while the destination must be a regular file under
the selected `--repo`; the validator recomputes its digest and still requires the receipt and model
registry bindings. This option does not rewrite the receipt and does not authorize publication or
turn a diagnostic remap into benchmark evidence.

The report must be `pass`. It prints stable status and identity data, never the
credential. The release doctor verifies that each required workflow (`CI`, `CodeQL`)
possesses at least one complete successful run for the exact source SHA (`--expected-release-sha`).
If subsequent historical runs or manual dispatches for that SHA were cancelled due to
GitHub Actions moving-main concurrency, the completed green run provides valid exact-SHA evidence
and the doctor records supporting run IDs. If all runs for a required workflow are cancelled,
pending, or failed, the doctor fails closed and lists the blocking run IDs. To reconcile a
blocking workflow run without altering workflow history, trigger a clean run for that exact ref
(`gh workflow run <workflow>.yml --ref <ref>`) and allow it to finish.

For a future release, reserve a fresh benchmark-data
concept/version before freezing the DOI into its v0.2 manifest:

```bash
uv run robot-sf release zenodo reserve \
  --token-file /home/luttkule/.config/robot-sf/zenodo.token \
  --state <credential-free-zenodo-state.json> \
  --metadata configs/benchmarks/releases/benchmark_data_release_s30_h600_zenodo_metadata.json
```

Keep the token file outside Git with mode `0600`. The state file contains no
credential and binds subsequent `upload`, `verify`, and irreversible `publish`
operations to the same deposition. Do not run `publish` until the accepted
20,160-cell campaign bundle has passed independent cold verification.

Upload and verify the generated bundle using:

- `docs/benchmark_camera_ready_release.md`

## Version Alignment (single source of truth: the software git tag)

The package version is derived automatically from the git tag by `hatch-vcs`
(release tags are plain `X.Y.Z`, e.g. `0.0.2`; release-candidate tags are
`rcX.Y.Z`, e.g. `rc0.0.3`). Do not hardcode a version in `pyproject.toml`.

When cutting a **full release** `X.Y.Z`:

- bump `CITATION.cff` `version:` to the new `X.Y.Z` (it tracks the latest full
  release tag; the benchmark release-protocol context stays in the title/abstract)
- run the alignment guard locally before tagging:

  ```bash
  uv run python scripts/dev/check_version_alignment.py
  ```

- push the tag; the `release-functional-badge` workflow re-runs this guard
  (gating) so `pyproject`, the built package, and `CITATION.cff` cannot drift

The guard also runs advisory (non-gating) on every CI run via the `lint` phase
of `scripts/dev/ci_driver.sh`.

Benchmark-data release tags are intentionally separate. They carry the frozen
S30/H600 config and manifest identity, but they do not change `pyproject.toml`,
`CITATION.cff`, or the package version. Publish benchmark data only after the
fresh Zenodo concept, checksums, claim boundary, and full-run evidence have been
reviewed.

## Archive and Citation

- ensure `CITATION.cff` remains current (aligned to the latest full release tag)
- keep the release tag and release asset URL stable
- require the manifest's reserved version DOI to match the published record and
  independently verify the concept/version distinction
- For the benchmark-data lane, follow
  [`benchmark_camera_ready_release.md`](./benchmark_camera_ready_release.md) for
  the direct Zenodo reserve/upload/verify/publish sequence, webhook disablement,
  and GitHub/Zenodo cold-download comparison. This lane does not bump the
  package version or publish the software-release DOI.
