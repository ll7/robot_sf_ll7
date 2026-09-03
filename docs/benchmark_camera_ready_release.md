# Camera-Ready Benchmark Release Workflow

## Purpose

This runbook describes publication of a validated benchmark-data campaign bundle
as a GitHub release asset with checksum and manifest verification. It is not a
software/package release procedure. The current benchmark-data target is the
14-arm S30/H600 matrix; the seven-planner/S3 instructions in historical
artifacts must not be reused.

For the full benchmark release protocol, start with:

- `docs/benchmark_release_protocol.md`
- `docs/benchmark_release_reproducibility.md`

The command in this document is the publication/upload step after a benchmark
release run has already produced a valid publication bundle.

Before publication, run the bounded smoke manifest
`configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml`
and retain its result as runtime evidence. A smoke pass is not full benchmark
evidence. Social Navigation Quality Index (SNQI) is advisory/no-ranking for
this release, including when calibration emits a warning.

## Prerequisites

- Completed camera-ready campaign output containing:
  - `reports/campaign_summary.json`
  - `publication_bundle.archive_path`
  - `publication_bundle.checksums_path`
  - `publication_bundle.manifest_path`
- A final release bundle whose `publication_manifest.json` declares
  `release_metadata.schema_version: benchmark-release-publication-metadata.v1`
  and includes the resolved release manifest/result, citation metadata, exact
  Zenodo metadata, generated rights/provenance statement, and pinned SNQI
  weights/baseline for cold verification.
- An immutable release identity recorded in the resolved manifest and release
  result: exact tag, source SHA, campaign-config SHA, and bundle SHA-256.
- `gh` CLI authenticated for repository upload.

## Recommended Tag Naming

Use an immutable benchmark-data tag that carries the H600/S30 identity:

- `paper-matrix-v2-h600-s30-<commit-sha12>`

Keep software/package tags (for example, `0.0.3`) in their separate release
lane. Do not derive a package version from the benchmark-data tag.

## Zenodo Boundary

The benchmark-data publication requires a fresh Zenodo concept and a new
version DOI after the final bundle is validated. Do not reuse historical
concepts `10.5281/zenodo.19482025` or `10.5281/zenodo.19563812`, and do not
assume GitHub-to-Zenodo automation is enabled. Until a real record exists, keep
the manifest DOI as a pending placeholder.

That rule applies to a new scientific campaign. A derived-metadata-only repair
of an already published campaign uses a new linked version inside that
campaign's existing concept, never replacement files behind the old DOI. See
[Immutable publication errata](RELEASE.md#immutable-publication-errata) for the
required predecessor custody, scientific-leaf equality, `zenodo new-version`,
and two-channel draft checks.

The direct Zenodo path is deliberately separate from the GitHub release path.
Disable or remove the repository's GitHub-to-Zenodo webhook immediately before
publishing the GitHub Release and leave no active hook. Do not use the webhook
to create or update this benchmark-data deposition: unrelated software or
model releases must not contaminate the benchmark concept.

The release doctor only records a read-only snapshot of the GitHub hook list;
it cannot reserve that state or prevent a hook from being reactivated between
the check and publication (the time-of-check/time-of-use boundary). Treat the
snapshot as a publication gate at that instant, disable/remove the specific
integration through the approved operator path, and recheck immediately before
publishing. The release receipt must retain the observed hook state and must
not claim that a read-only check proves future non-reactivation.

The credential-free public audit pins API discovery to the exact production
endpoints `https://api.github.com` and `https://zenodo.org/api` (with trailing
slashes normalized only). API/document requests never follow redirects.
Route-mocked tests may inject alternate HTTPS bases through the explicit
test-only API, which is not exposed by the production CLI. GitHub release
assets are allowed their documented `github.com` to
`release-assets.githubusercontent.com` hand-off; Zenodo assets, arbitrary
`githubusercontent.com` subdomains, userinfo, and non-default ports are not
trusted.

## Exact-SHA Continuous Integration (CI) reconciliation

The release doctor requires one completed successful run of each required
workflow (`CI` and `CodeQL`) for the exact source SHA. It records the selected
supporting run IDs in its credential-free receipt. A later concurrency or
infrastructure cancellation for that same workflow is recorded as ignored when
the earlier successful run exists; a pending run or genuine failure remains a
blocker. Re-run the doctor against the same immutable checkout to reconcile
workflow history—do not delete cancelled runs or treat a retry on moving
`main` as new evidence for the frozen SHA.

If a required workflow has no successful exact-SHA run, inspect the credential-
free `gh run list --commit <sha>` output and retry that workflow only through
the normal GitHub Actions path. After a retry completes, run the doctor again
and retain its supporting or blocking run IDs with the release receipt. A
successful retry is sufficient; historical cancellations do not need to be
removed.

## Final queue reconciliation

Final-mode admission accepts either a still-dispatchable queue row (`ready` or
`queued`) or an honestly closed row whose state is `complete` or `done` and
whose execution, artifact, evaluation, completion, and preservation fields
jointly report success (`passed`, `verified`, a terminal evaluation, `complete`,
and `preserved`). A terminal state or preservation claim by itself is not
publication evidence. Failed, incomplete, and partially preserved rows remain
blocked; reconcile the private queue and rerun the doctor after closeout.

## Frozen-checkout validation boundary

The public doctor and private launch packet are versioned contracts. A doctor
imported from a newer checkout must not reinterpret an older packet as if its
missing inputs or traceability fields were valid. `--repo` is the explicit
public-checkout root: the doctor passes it through manifest, campaign,
scenario, and checkpoint validation, and uses it for clean exact-HEAD checks.
Legacy packet fields are synthesized only after the strict private-ledger and
queue-export equality checks described below; otherwise the mismatch remains
fail-closed. Do not weaken packet or file-hash checks to make a historical
packet pass.

Run the reviewed doctor code from its own tooling checkout and point `--repo`
at the untouched frozen public checkout. The code checkout and the release
checkout intentionally have different Git identities: manifest/config/scenario
resolution, checkpoint containment, and `_git_check` all use `--repo`, while
the Python implementation and tests come from `TOOLING_ROOT`. Do not patch or
otherwise change the frozen execution checkout.
Use absolute paths for private packet, queue, and private-ops inputs because
they are not public files beneath `--repo`:

```bash
TOOLING_ROOT=<reviewed-public-tooling-checkout>
FROZEN_ROOT=<untouched-exact-public-release-checkout>
PRIVATE_OPS_ROOT=<trusted-private-ops-git-checkout>
PRIVATE_LAUNCH_PACKET=<absolute-private-launch-packet>
PRIVATE_QUEUE=<absolute-private-queue>
FROZEN_SHA=b1d5ab6de708385c0828c99501a9d1c29727ec11
(
  cd "$TOOLING_ROOT"
  uv run --project "$TOOLING_ROOT" pytest \
    "$TOOLING_ROOT/tests/benchmark/test_release_doctor.py" \
    "$TOOLING_ROOT/tests/benchmark/test_release_doctor_edge_cases.py"
  uv run --project "$TOOLING_ROOT" robot-sf release doctor \
  --repo "$FROZEN_ROOT" \
  --manifest "$FROZEN_ROOT/<frozen-manifest-path>" \
  --expected-release-sha "$FROZEN_SHA" \
  --expected-base-sha <frozen-manifest-base-sha> \
  --tag <frozen-release-tag> \
  --checkpoint-receipt "$FROZEN_ROOT/<frozen-checkpoint-receipt>" \
  --private-launch-packet "$PRIVATE_LAUNCH_PACKET" \
  --private-queue "$PRIVATE_QUEUE" \
  --private-ops-repository "$PRIVATE_OPS_ROOT" \
  --expected-campaign-id <frozen-campaign-id> \
  --publication-mode final
)
```

Use the packet's immutable manifest/base/tag and private paths as the explicit
values; never copy credentials into the command or receipt. The private-ops
reviewed commit is a trusted private-ledger assumption: the doctor verifies
that the commit object exists and reads `ops/jobs/jobs.yaml` and
`ops/jobs/queue.yaml` with object-addressed `git show`, never from the private
worktree. Git signatures are not required because no trusted signing key is
available; the packet-pinned commit, exact job `14884`, queue identity, source
SHA, result/preservation digests, terminal statuses, and a future/stale-safe
`submitted_at` window provide the fail-closed binding. This invocation keeps
the execution checkout unchanged and makes any schema incompatibility visible
as a blocked doctor result.

## Command Path

1. Dry-run validation + command plan:

```bash
uv run python scripts/tools/publish_camera_ready_release.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --repo ll7/robot_sf_ll7 \
  --tag <release_tag> \
  --output-json output/benchmarks/camera_ready/<campaign_id>/reports/release_publish_plan.json
```

2. Execute asset upload (first publication: create the draft release first):

```bash
uv run python scripts/tools/publish_camera_ready_release.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --repo ll7/robot_sf_ll7 \
  --tag <release_tag> \
  --create-draft \
  --expected-source-sha <exact-40-char-source-sha> \
  --execute-upload
```

The upload helper requires `--expected-source-sha` for every mutating invocation.
It uploads only into an unpublished draft whose release target and peeled tag
ref both resolve to that exact SHA. Use `--create-draft` to create the missing
tag-targeted draft before the first upload; it fails closed when the tag already
exists, the release is published, or either target is ambiguous. Existing
drafts are retry-safe: the helper validates the complete remote asset inventory
(`name`, `state=uploaded`, positive `size`, and `sha256:<digest>`) before any
upload, rejects extras, duplicates, stale bytes, and mismatches, uploads only
missing assets, and never passes `--clobber`. If all expected assets already
match, it skips the upload. Dry-run (without `--execute-upload`) prints the
planned `gh release create`/`gh release upload` commands without touching GitHub.
For an erratum, the detached `publication_custody.json` is also checked against
the local archive, manifest, checksums, source SHA, and canonical custody fields
before a draft can be created or reused.
The helper never reserves, uploads to, or publishes Zenodo. Use the direct
Zenodo CLI for the reserved deposition after the bundle has passed the
independent cold check:

```bash
# Keep this file outside Git with mode 0600; never print its contents.
ZENODO_TOKEN_FILE=/home/<user>/.config/robot-sf/zenodo.token
ZENODO_STATE=output/release/zenodo-deposition.json
ZENODO_MANIFEST=configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml
ZENODO_METADATA=configs/benchmarks/releases/benchmark_data_release_s30_h600_zenodo_metadata.json

uv run robot-sf release zenodo reserve \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --metadata "$ZENODO_METADATA"

# If this unpublished draft still exists but its local state was lost, recover
# that exact deposition instead of reserving a second DOI. This performs one
# authenticated read and fails closed on any manifest, metadata, DOI, or state drift.
uv run robot-sf release zenodo recover \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --manifest "$ZENODO_MANIFEST" \
  --metadata "$ZENODO_METADATA" \
  --deposition-id <existing-unpublished-deposition-id>

uv run robot-sf release zenodo upload \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  output/benchmarks/publication/<campaign_id>_publication_bundle.tar.gz

uv run robot-sf release zenodo verify \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --metadata "$ZENODO_METADATA"

# Run only after acceptance and independent cold verification pass.
uv run robot-sf release zenodo publish \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --metadata "$ZENODO_METADATA"

# Mandatory post-publication check; require a passing published-record receipt.
uv run robot-sf release zenodo verify \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --metadata "$ZENODO_METADATA"
```

`reserve` must return a fresh concept and version DOI; freeze those values in
the release manifest before the immutable execution point. Use `recover` only
when that exact unpublished draft still exists and the credential-free local
state was lost; it never reserves or mutates a deposition and refuses published
or mismatched drafts. `upload` must send
the byte-identical bundle used for GitHub. `verify` is read-only and must check
the title, dataset type, GPL-3.0-only license, creator union, exact source tag,
and concept/version DOI distinction. `publish` is irreversible; never run it
for a draft with missing files, unaccepted rows, or an unresolved DOI.
The publisher performs a mandatory fresh draft verification immediately before
the irreversible request. The documented legacy [Zenodo deposition actions
API](https://developers.zenodo.org/#depositions-actions-publish) exposes no
conditional compare-and-publish precondition, so a final time-of-check/time-of-use
window remains. Run the authenticated `zenodo verify` command again immediately
after publication and require its published-record receipt to pass before
treating the DOI as accepted.

Disable or remove the specific GitHub-to-Zenodo webhook through repository
settings or the approved GitHub API operation immediately before GitHub
publication. Confirm that no hook is active with the release doctor and retain
only webhook id/state in the operator receipt. Treat the hook configuration URL
as credential-bearing: never print or retain it, a token, or an authorization
header in a receipt or log.

## Validation Checklist

- `release_publish_plan.json` contains expected paths and URLs.
- `checksums.sha256` is non-empty and references bundle files.
- Release page contains archive + checksums + manifest assets.
- Campaign summary contains URL placeholders:
  - `release_url`
  - `release_asset_url`
  - `doi_url`

## Cold verification and exact identity checks

Perform these checks from a clean temporary directory that contains neither
the build output nor the source worktree:

For the public, credential-free discovery and download step, run:

```bash
uv run robot-sf release audit-published \
  --tag <release_tag> \
  --doi <version-doi> \
  --output /tmp/published-release-audit.json
```

The command uses only unauthenticated HTTPS `GET` requests, bounds streamed
downloads and archive extraction by member count, per-file expanded bytes, and
cumulative expanded bytes, and writes no release or Zenodo state. It checks the
exact public tag/release and Zenodo version record before passing both channel
directories to the offline audit core. Invalid receipts use bundle-relative
diagnostics rather than audit-host temporary paths. A receipt with
`status: unavailable` is a transport or service condition, not a failed
release; retry it. A `pass` is a repeatable identity/download check, not full
benchmark evidence and does not authorize publication. The command cannot
reserve, upload, edit, publish, or rename a release.

For a canonical `-erratum.1` release, the same command requires reviewed pins
for the exact predecessor DOI, predecessor tag, predecessor archive digest and
size, scientific source SHA, accepted builder/validator SHA, concept DOI, and
orchestration SHA. The caller must provide these independently; the observed
Zenodo `isNewVersionOf` relation and embedded successor receipt are evidence to
check, not sources for expected values. For the September erratum, use:

```bash
uv run robot-sf release audit-published \
  --tag paper-matrix-v2-h600-s30-2026-09-59577bad289dd692ba3580e1600c4a649ae27880-erratum.1 \
  --doi <successor-version-doi> \
  --expected-source-sha 59577bad289dd692ba3580e1600c4a649ae27880 \
  --expected-concept-doi 10.5281/zenodo.22227034 \
  --expected-predecessor-doi 10.5281/zenodo.22227035 \
  --expected-predecessor-tag paper-matrix-v2-h600-s30-2026-09-59577bad289dd692ba3580e1600c4a649ae27880 \
  --expected-predecessor-archive-sha256 e8f301c6f4eae16fdaf83f59b31bef060d84bf5a0e23dfdbf375f834b25d7b4b \
  --expected-predecessor-size-bytes 54219004 \
  --expected-builder-sha a4aaf1f06860cf632d0173c5a13e11ad855b6df2 \
  --expected-validator-sha a4aaf1f06860cf632d0173c5a13e11ad855b6df2 \
  --expected-orchestration-sha <reviewed-orchestration-sha>
```

The orchestration value must be the exact reviewed 40-character lowercase SHA;
it is intentionally not inferred from public release data. The audit compares
the pins with public records, release bodies, downloaded predecessor bytes,
embedded receipt, custody receipt, and scientific snapshots before accepting
the successor.

1. Download the GitHub Release archive and its checksum/manifest assets. For a
   derived-metadata erratum, also require the detached
   `publication_custody.json`; the archive, `publication_manifest.json`,
   `checksums.sha256`, and custody receipt must have byte-identical counterparts
   on Zenodo. For a canonical `-erratum.N` tag, the GitHub publication helper
   automatically requires and uploads that detached receipt beside the archive;
   a missing or symlinked receipt blocks before draft mutation. Extract the
   archive and run `sha256sum -c checksums.sha256` from the bundle root.
2. Confirm that `payload/release/release_manifest.resolved.json` and
   `payload/release/release_result.json` agree on release id, tag, source SHA,
   acceptance status, and 20,160 episode identities. No fallback, degraded,
   failed, or unavailable row may be treated as evidence.
3. Confirm that the metadata roles in `publication_manifest.json` point to
   files inside the archive and that every declared digest matches. The citation,
   Zenodo metadata, rights/provenance statement, and both pinned SNQI assets
   must be present. Raw episode and component-metric files remain in the
   payload; `output/` remains working storage, not a citation target.
4. Confirm the tag points to the exact source SHA recorded by the release result
   and that the frozen manifest/config hashes match the bundle. A tag or DOI
   mismatch is a release blocker, not a documentation warning.
5. Download the Zenodo file independently after upload (and again after
   publication), extract it into a second clean directory, and repeat steps
   1–4. Compare the GitHub and Zenodo archive SHA-256 values byte-for-byte.
6. Verify that Zenodo reports the reserved version DOI and parent concept, the
   exact title/type/license/creators, and the source-tag relation. Keep the
   readback receipts with the private durable-artifact record.

These checks are independent of the local build directory. A successful local
`export` or draft upload alone is not benchmark evidence or a publication.

## Paper Ingestion Links

After upload, reference:

- release URL from `release_url`
- archive URL from `release_asset_url`
- DOI URL from `doi_url`
