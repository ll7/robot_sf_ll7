# Benchmark Release Checklist

This checklist covers the approved S30/H600 benchmark-data release. It is a
different release lane from the Robot SF software/package release: the
benchmark-data tag identifies an immutable campaign contract, while the
software tag identifies installable source. Do not infer a package version from
the benchmark-data tag, or reuse a software-release DOI for benchmark data.

For the software-package lane, first build the immutable candidate with
[`software_release_candidate.md`](./software_release_candidate.md), then follow
[`software_release_promotion.md`](./software_release_promotion.md) for the protected
TestPyPI → PyPI promotion. That workflow is separate from this benchmark-data checklist and
requires a passed public-index cold-install gate before production publication.

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

For a future v0.2 benchmark-data release, freeze a tracked identity template
instead of writing its own final commit SHA into tracked bytes. After the exact
clean source commit and already-reserved concept/version DOI coordinates are
known, generate and verify the ignored resolved identity:

- Manifest template:
  `configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml`
- Campaign template:
  `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml`
- Zenodo metadata template:
  `configs/benchmarks/releases/benchmark_data_release_s30_h600_zenodo_metadata.template.json`

The resolver derives `latest_main_base_commit` from the exact first parent of
the selected source commit (and applies the same value to the optional
`planning_base_sha`). This records the immutable mainline base without tracking
the moving `origin/main` or creating a self-referential manifest.

```bash
SOURCE_COMMIT="$(git rev-parse --verify HEAD^{commit})"
RELEASE_TAG="${RELEASE_PREFIX:?set the reviewed release prefix}-${SOURCE_COMMIT}"
uv run python scripts/tools/resolve_benchmark_release_identity.py generate \
  --template "${TRACKED_RELEASE_TEMPLATE:?set the tracked template path}" \
  --output output/release/release_identity.resolved.json \
  --source-commit "$SOURCE_COMMIT" \
  --release-tag "$RELEASE_TAG" \
  --concept-doi "${RESERVED_BENCHMARK_CONCEPT_DOI:?set the reserved concept DOI}" \
  --version-doi "${RESERVED_BENCHMARK_VERSION_DOI:?set the reserved version DOI}"
uv run python scripts/tools/resolve_benchmark_release_identity.py verify \
  --identity output/release/release_identity.resolved.json
```

Run the same verify command at the same repository-relative output path in a
disposable cold checkout of `SOURCE_COMMIT`. The identity and sibling
`zenodo_metadata.resolved.json` must be byte-identical to the first generation.
Use the resolved identity as `--manifest` for future runner and doctor checks. See
[`benchmark_release_protocol.md`](./benchmark_release_protocol.md#future-tracked-template-identity-resolution)
for the template slots and fail-closed rules. These commands do not reserve a
DOI, create a tag, publish a release, or submit a campaign.

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

To admit checkpoints that were staged by the canonical staging workflow, pass
the fresh receipt during preflight as well:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml \
  --mode preflight \
  --checkpoint-receipt output/release/checkpoints/runtime_smoke_staging_receipt.json
```

Preflight validates the receipt against the manifest's canonical campaign
config. A valid receipt is reported under
`checkpoint_admission.staged_checkpoint_admission` with `status: admitted` and
`submit_safe: true`. The separate metadata-only check remains visible as
`metadata_resolvable` and `metadata_submit_safe`; a metadata-only
`submit_safe: false` result is diagnostic when the staged receipt is admitted,
not a contradictory release decision. A supplied missing, stale, mismatched,
or otherwise invalid receipt fails closed with
`status: checkpoint_receipt_rejected` before campaign setup and is not
benchmark evidence.

### Full-release Slurm launch manifest

For a full release, bind the exact resolved identity and the successful public
runner preflight into a separate, deterministic launch manifest before handing
the packet to private operations:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest output/release/release_identity.resolved.json \
  --mode preflight > output/release/runner_preflight.json
uv run python scripts/tools/generate_slurm_launch_manifest.py \
  --resolved-identity output/release/release_identity.resolved.json \
  --runner-preflight output/release/runner_preflight.json \
  --output output/benchmarks/camera_ready/<campaign_id>/slurm_launch_manifest.json
uv run python scripts/tools/slurm_campaign_preflight.py \
  --manifest output/benchmarks/camera_ready/<campaign_id>/slurm_launch_manifest.json \
  --public-repo . \
  --json
```

`campaign_manifest.json` remains the runner's configuration/provenance
artifact; it is not the Slurm launch packet. The generated launch manifest is
an ignored, no-submit intent artifact with 14 planner-arm cells, 48 scenarios,
30 resolved seeds, H600, differential-drive kinematics, and 1,440 declared
rows per arm (20,160 total). It contains no scheduler identifier, execution
result, benchmark-success claim, or publication authorization. Preserve only
its source/config/input hashes and compact validation receipt when promoting
durable provenance; keep the generated packet and raw preflight output in the
worktree-local ignored output tree unless private operations explicitly
hydrates them from a canonical source.

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

Before a full release attempt, the public runner also supports a no-campaign
rehearsal. It requires the same staged checkpoint receipt and exact-source
runtime-smoke receipt, then stops before campaign allocation:

```bash
REHEARSAL_SOURCE_COMMIT="$(git rev-parse --verify HEAD)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml \
  --mode rehearsal \
  --source-commit "$REHEARSAL_SOURCE_COMMIT" \
  --checkpoint-receipt output/release/checkpoints/staging_receipt.json \
  --runtime-smoke-receipt output/benchmarks/camera_ready/<smoke_id>/release/release_result.json
```

The command pins and records the exact clean checkout being rehearsed; if a
reviewed SHA was selected separately, set `REHEARSAL_SOURCE_COMMIT` to that
exact 40-character value instead. Never substitute a planning/base SHA.
The canonical benchmark-data manifest is retained as a historical compatibility
manifest without `source_sha`, so this explicit pin is required. If a future
manifest declares `source_sha`, that manifest value is authoritative and any
explicit argument must match it.

Successful rehearsal output is admission/preflight evidence only. It reports
`campaign_execution_status: not_started` and must not be treated as benchmark
evidence or publication authorization; no campaign output, episode, bundle, or
scheduler submission is created. Both checkpoint receipts are validated
independently; their wrapper hashes may differ because each is bound to its own
campaign config, but their checkpoint arm identities and model-byte SHA-256
values must match. Resume-only options, including
`--resume-receipt-max-age-hours`, are rejected.

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

### Preserved post-execution evidence

If a fixed campaign completed scientifically but the terminal job failed only
at the publication validator gate, use the separate post-execution doctor. It
validates the reviewed derived revalidation receipt, the checksummed bundle and
preflight, and the historical checkpoint, runtime-smoke, stress, queue, and
job records. It binds the evidence to the frozen source, release tag, base,
campaign, and consumed job identity. The failed queue row is intentionally not
required to be dispatchable; a new benchmark submission is not authorized by
this mode.

```bash
uv run robot-sf release doctor --post-execution \
  --repo "$PWD" \
  --manifest configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml \
  --expected-release-sha b1d5ab6de708385c0828c99501a9d1c29727ec11 \
  --expected-base-sha cd831d7582c117ac9529065e7d1c60386933c92d \
  --tag paper-matrix-v2-h600-s30-2026-08-cd831d7582c1 \
  --expected-campaign-id issue7742_release_full-s30-h600-b1d5ab6de708-v1_20260825 \
  --expected-job-id 14890 \
  --derived-revalidation-receipt <derived-revalidation-receipt.json> \
  --publication-bundle <publication-bundle-directory> \
  --publication-archive <publication-bundle.tar.gz> \
  --publication-preflight <publication-preflight.json> \
  --private-launch-packet <private-launch-packet.yaml> \
  --private-queue <private-ops-queue.yaml> \
  --private-jobs <private-ops-jobs.yaml> \
  --private-evaluation-receipt <private-derived-evaluation-receipt.json> \
  --expected-validator-sha bd4bc4b4018b24c887c8e91ad834bc6898d7aad2
```

This mode is read-only and emits no credentials. A passing report is still
only an acceptance gate: publication requires the independent GitHub/Zenodo
cold-download checks below, and SNQI remains advisory when calibration fails.

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

### Benchmark-data tag identity (source-SHA binding)

For future benchmark-data releases, any SHA-like component in the release tag
must be derived from the **final immutable source SHA** — never from a
preliminary planning/base SHA. The workflow:

1. Freeze the final source commit first; record it as `source_sha` in the
   manifest (the manifest also stores any `planning_base_sha` as a separate
   provenance field).
2. Derive SHA-bearing tag names from that final SHA (e.g.
   `paper-matrix-<...>-<40-hex source_sha>`), or use an explicit semantic
   identifier scheme that contains no SHA-like component.
3. `robot-sf release doctor` rejects a SHA-bearing tag whose component
   disagrees with the manifest `source_sha`; a planning/base SHA in a
   SHA-bearing tag never satisfies the check. Short hex abbreviations must be
   a prefix of the final source SHA or they fail closed.
4. The publication contract cross-checks the final `source_sha` across the
   campaign result, resolved manifest, publication provenance, and derived
   receipt. Missing or conflicting future identity is a publication blocker;
   an explicit semantic tag still requires the same source provenance.

The published August 2026 tag (`paper-matrix-v2-h600-s30-2026-08-cd831d7582c1`)
is immutable; its authoritative source identity is the manifest/bundle SHA
(`b1d5ab6d…`), not the tag suffix. Do not rename or retarget it.

## Credential-free public audit

After publication, a reviewer can start the cold audit with only the exact
public GitHub tag and Zenodo version DOI:

```bash
uv run robot-sf release audit-published \
  --tag paper-matrix-v2-h600-s30-2026-08-cd831d7582c1 \
  --doi 10.5281/zenodo.22077448 \
  --output /tmp/published-release-audit.json
```

This command performs unauthenticated HTTPS `GET` requests, resolves the
published GitHub release and tag commit, checks the Zenodo version/concept DOI
and source-tag relation, then streams the public assets into isolated temporary
directories before calling the offline audit core. Downloads are bounded and
the command never reserves, uploads, edits, publishes, or renames a release;
it accepts no token or deposition-state path. The JSON receipt reports
`pass`, `invalid`, or `unavailable`. `unavailable` means the public service or
transport could not be checked and must be retried; neither a network pass nor
an unavailable result replaces the full benchmark and publication evidence
checks below.

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
