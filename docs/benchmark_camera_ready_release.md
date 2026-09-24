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
version Digital Object Identifier (DOI) after the final bundle is validated.
Do not reuse historical concepts `10.5281/zenodo.19482025` or
`10.5281/zenodo.19563812`, and do not assume GitHub-to-Zenodo automation is
enabled. Until a real record exists, keep the manifest DOI as a pending
placeholder.

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

After campaign evidence is frozen and before step 2 creates the GitHub draft,
run the passing pre-tag release doctor described in
[`RELEASE.md`](RELEASE.md#publication). It checks that both the planned tag and
release are unused, so save its JSON report and checksum before creating the
draft. Also retain the exact local asset names, byte sizes, and
SHA-256 digests while the bundle is available; those values are the readback
contract after the GitHub release is published. Preserve these small receipts
with durable release evidence before removing the worktree; `output/` alone is
not durable.

```bash
set -euo pipefail
mkdir -p output/release
PUBLISH_PLAN=output/benchmarks/camera_ready/<campaign_id>/reports/release_publish_plan.json
ARCHIVE_PATH="$(jq -er '.archive_path' "$PUBLISH_PLAN")"
CHECKSUMS_PATH="$(jq -er '.checksums_path' "$PUBLISH_PLAN")"
MANIFEST_PATH="$(jq -er '.manifest_path' "$PUBLISH_PLAN")"
test -f "$ARCHIVE_PATH" && test -f "$CHECKSUMS_PATH" && test -f "$MANIFEST_PATH"
{
  wc -c -- "$ARCHIVE_PATH" "$CHECKSUMS_PATH" "$MANIFEST_PATH"
  sha256sum -- "$ARCHIVE_PATH" "$CHECKSUMS_PATH" "$MANIFEST_PATH"
} > output/release/github_assets_pre_tag.txt
sha256sum output/release/github_assets_pre_tag.txt \
  > output/release/github_assets_pre_tag.txt.sha256
```

Use the paths from the dry-run `release_publish_plan.json`, not the raw path
strings in `campaign_summary.json`: the helper has already resolved each path
against the validated campaign and repository roots and records the resulting
absolute paths in its plan output.

2. Execute asset upload into the draft (the draft does not yet materialize the
Git tag):

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
It uploads only into an unpublished draft whose exact `target_commitish` matches
that SHA. GitHub normally creates the tag ref when the draft is published, so an
explicit tag-ref 404 is valid while the release remains a draft; if the ref
already exists, the helper peels it and requires the same exact SHA. Use
`--create-draft` to create the missing tag-targeted draft before the first
upload. It retries only a briefly absent post-create listing for a bounded
interval and fails closed on API errors, a published release, or any ambiguous
or conflicting identity. Existing drafts are retry-safe: the helper validates
the complete remote asset inventory
(`name`, `state=uploaded`, positive `size`, and `sha256:<digest>`) before any
upload, rejects extras, duplicates, stale bytes, and mismatches, uploads only
missing assets, and never passes `--clobber`. If all expected assets already
match, it skips the upload. Dry-run (without `--execute-upload`) prints the
planned `gh release create`/`gh release upload` commands without touching GitHub.
For an erratum, the detached `publication_custody.json` is also checked against
the local archive, manifest, checksums, source SHA, and canonical custody fields
before a draft can be created or reused.

After reviewing and publishing the GitHub draft through the approved release
workflow, verify the tag and public release readback before any Zenodo file
upload. Publishing the draft materializes the tag. Use the full source SHA and
tag from the resolved release identity, not the current branch tip:

```bash
set -euo pipefail
SOURCE_SHA=<full source SHA from the resolved release identity>
RELEASE_TAG=<exact tag from the resolved release identity>

git fetch --no-tags origin \
  "refs/tags/${RELEASE_TAG}:refs/tags/${RELEASE_TAG}"
test "$(git rev-parse "${RELEASE_TAG}^{commit}")" = "$SOURCE_SHA"
gh release view "$RELEASE_TAG" --repo ll7/robot_sf_ll7 \
  --json tagName,isDraft,assets \
  --jq '{tagName,isDraft,assets:[.assets[] | {name,size,state,digest}]}'
```

Require the exact tag and source SHA, `isDraft: false`, and exactly the
archive, `checksums.sha256`, and `publication_manifest.json` assets. Each asset
must have `state: uploaded`; compare its size and `sha256:` digest with the
local asset inventory recorded before tag creation. Stop before Zenodo upload
on any mismatch. Do not rerun `release doctor` here: its unused-tag failure is
expected after publication and cannot replace this exact-tag/asset readback.

The helper never reserves, uploads to, or publishes Zenodo. Use the direct
Zenodo CLI for the reserved deposition after the bundle has passed the
independent cold check:

After the exact GitHub tag, source commit, release visibility, and three assets
have passed readback above, reconcile the Zenodo draft before uploading files.
If no deposition has been reserved for this release, run `reserve` exactly
once. If it already exists, skip `reserve` and use only its reviewed deposition
ID and DOI-bound manifest; never create a replacement DOI to recover local
state. Preview and, when required, repair the empty draft's version/date and
only explicitly reviewed source-provenance drift, then `recover` the local
state. Upload the archive and both companions, run `verify`, publish once,
then run `verify` and the anonymous `audit-published` check. Run the commands
below as separate operator steps, pausing after metadata preview to inspect the
complete before/after diff before deciding whether to apply a repair.

```bash
set -euo pipefail
# Keep this file outside Git with mode 0600; never print its contents.
ZENODO_TOKEN_FILE=/home/<user>/.config/robot-sf/zenodo.token
FROZEN_SOURCE_ROOT=<absolute-untouched-source-checkout>
TOOLING_ROOT=<reviewed-checkout-containing-the-Zenodo-CLI>
RELEASE_CONFIG_DIR="$FROZEN_SOURCE_ROOT/configs/benchmarks/releases"
ZENODO_STATE=output/release/zenodo-deposition.json
ZENODO_METADATA="$RELEASE_CONFIG_DIR/benchmark_data_release_s30_h600_zenodo_metadata.json"
ZENODO_ARCHIVE=<exact-publication-bundle-archive-path>
ZENODO_CHECKSUMS=<exact-publication-bundle-checksums-path>
ZENODO_PUBLICATION_MANIFEST=<exact-publication-bundle-manifest-path>
VERSION=<reviewed-release-version>
PUBLICATION_DATE=<reviewed-YYYY-MM-DD>

cd "$TOOLING_ROOT"

# New release only: skip this command when using an already-reserved draft.
uv run robot-sf release zenodo reserve \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --metadata "$ZENODO_METADATA"

# Freeze the returned concept/version DOI in the reviewed release identity
# before any bound post-reservation operation.
ZENODO_MANIFEST=<reviewed-manifest-path-bound-to-returned-doi>
DEPOSITION_ID=<returned-or-reviewed-existing-deposition-id>

# Preview and validate this exact empty draft before recovering local state.
# Stop if it is published, has files, names another concept/DOI, or has any
# unreviewed metadata drift.
uv run robot-sf release zenodo repair-draft-metadata \
  --token-file "$ZENODO_TOKEN_FILE" \
  --repository-root "$FROZEN_SOURCE_ROOT" \
  --manifest "$ZENODO_MANIFEST" \
  --metadata "$ZENODO_METADATA" \
  --deposition-id "$DEPOSITION_ID" \
  --version "$VERSION" \
  --publication-date "$PUBLICATION_DATE"

# Apply only after reviewing the preview's exact before/after metadata diff,
# old source tag/SHA/base, and metadata digest. Any unrelated drift or
# non-empty draft is a stop. If source SHA or base SHA differs from the
# corresponding `*_after` value, pass its reviewed `*_before` value as shown.
# Add `--expected-remote-source-sha <reviewed-preview-old-source-sha>` and/or
# `--expected-remote-base-sha <reviewed-preview-old-base-sha>` to this command
# only when the corresponding preview field is present and differs from `*_after`.
uv run robot-sf release zenodo repair-draft-metadata \
  --token-file "$ZENODO_TOKEN_FILE" \
  --repository-root "$FROZEN_SOURCE_ROOT" \
  --manifest "$ZENODO_MANIFEST" \
  --metadata "$ZENODO_METADATA" \
  --deposition-id "$DEPOSITION_ID" \
  --version "$VERSION" \
  --publication-date "$PUBLICATION_DATE" \
  --expected-remote-metadata-sha256 <reviewed-preview-digest> \
  --expected-remote-source-tag <reviewed-preview-old-source-tag> \
  --apply

# Re-read the now-matching draft and write fresh local state before upload.
uv run robot-sf release zenodo recover \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --repository-root "$FROZEN_SOURCE_ROOT" \
  --manifest "$ZENODO_MANIFEST" \
  --metadata "$ZENODO_METADATA" \
  --deposition-id "$DEPOSITION_ID"

uv run robot-sf release zenodo upload \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --repository-root "$FROZEN_SOURCE_ROOT" \
  --manifest "$ZENODO_MANIFEST" \
  "$ZENODO_ARCHIVE" \
  "$ZENODO_CHECKSUMS" \
  "$ZENODO_PUBLICATION_MANIFEST"

uv run robot-sf release zenodo verify \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --repository-root "$FROZEN_SOURCE_ROOT" \
  --manifest "$ZENODO_MANIFEST" \
  --metadata "$ZENODO_METADATA" \
  --expected-version "$VERSION" \
  --expected-publication-date "$PUBLICATION_DATE"

# Run only after acceptance and independent cold verification pass.
uv run robot-sf release zenodo publish \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --repository-root "$FROZEN_SOURCE_ROOT" \
  --manifest "$ZENODO_MANIFEST" \
  --metadata "$ZENODO_METADATA" \
  --expected-version "$VERSION" \
  --expected-publication-date "$PUBLICATION_DATE"

# Mandatory post-publication check; require a passing published-record receipt.
uv run robot-sf release zenodo verify \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --repository-root "$FROZEN_SOURCE_ROOT" \
  --manifest "$ZENODO_MANIFEST" \
  --metadata "$ZENODO_METADATA" \
  --expected-version "$VERSION" \
  --expected-publication-date "$PUBLICATION_DATE"
```

The repair is limited to the source commit SHA, mainline base commit SHA, and
release tag inside the exact structured provenance sentence, their matching
source-tag and source-commit identifiers, plus the Zenodo-only `version` and
`publication_date` overlay. It rejects changes to other identifiers, release
identity, or scientific prose before any PUT. It does not edit the resolved
metadata file or its copy inside the immutable archive. If the draft already
matches, skip `--apply` and continue with `recover`. Record the exact preview
and resulting metadata diff. Before
upload, require the archive SHA-256 to match the frozen release identity and
the GitHub asset digest, then verify the extracted payload with
`sha256sum -c checksums.sha256`; stop on any mismatch. Use the reviewed tooling
checkout for the CLI and pass `--repository-root "$FROZEN_SOURCE_ROOT"` on
each manifest-bound command so all operations validate the untouched
exact-source identity. If the deposition already exists, do not reserve a
second DOI: set `DEPOSITION_ID` and the DOI-bound manifest from the reviewed
release identity, preview/repair that exact empty draft, then run `recover` to
rebuild local state.

### DOI resolution after publication

A reserved version DOI is already bound to the draft but normally does not
resolve through `doi.org` until Zenodo publishes the record. Do not make DOI
resolver success a pre-publication gate. After the Zenodo `publish` request,
run the post-publication `verify` above and the anonymous
[`audit-published`](#cold-verification-and-exact-identity-checks) audit below;
then check DOI resolution separately from the public record URL:

```bash
set -euo pipefail
curl --location --max-time 30 --output /dev/null \
  --write-out 'DOI resolver HTTP %{http_code}\n' \
  "https://doi.org/<version-doi>"
curl --max-time 30 --output /dev/null \
  --write-out 'Zenodo record HTTP %{http_code}\n' \
  "https://zenodo.org/records/<record-id>"
```

If `doi.org` returns 404 while the direct Zenodo record URL and public file
audit succeed, report **“record public, DOI resolution pending”** and share
`https://zenodo.org/records/<record-id>`. Do not submit `publish` again, create
a `new-version`, edit the published record, or reserve a replacement DOI to
work around resolver delay. If registration remains unresolved when you
recheck after a delay, hand it to Zenodo support and leave the published
record untouched.

For the September 2026 derived-metadata successor, keep the generic v0.2
manifest above out of the post-reservation commands and bind the reserved DOI
through the checked-in erratum contract instead:

```bash
ZENODO_MANIFEST=configs/benchmarks/releases/benchmark_data_release_s30_h600_2026_09_erratum_1.json
ZENODO_METADATA=configs/benchmarks/releases/benchmark_data_release_s30_h600_2026_09_erratum_1_zenodo_metadata.json
```

That contract preserves the scientific source identity while binding the new
erratum tag and successor version DOI. The CLI validates it before constructing
an authenticated session. The successor bundle keeps the predecessor campaign's
exact `release_id`; only `release_tag`, DOI, URL, and metadata coordinates move
to the new publication identity.

`reserve` is the explicit pre-reservation step: it may omit `--manifest` because
Zenodo assigns the version DOI in its response. Freeze the returned concept and
version identity in the reviewed release manifest or erratum contract before
continuing. The CLI requires that binding for `recover`, `upload`, `verify`,
and `publish`, and rejects an omission before constructing an authenticated
HTTP session.
The same pre-reservation exception applies to `new-version`: create the linked
successor draft without `--manifest` when its version DOI is not yet known,
then freeze that DOI and use the manifest-bound post-reservation commands above.
Use `recover` only when that exact unpublished draft still exists and the
credential-free local state was lost; it never reserves or mutates a deposition
and refuses published or mismatched drafts. When the bound metadata declares a
successor, recovery also reconstructs and validates its predecessor, concept,
and source-tag lineage so inherited-file cleanup remains available. `upload` must send
the byte-identical bundle used for GitHub. Zenodo new-version drafts inherit the
predecessor files. When the sealed state proves exact new-version provenance,
`upload` validates the complete local and remote inventories, uploads the
intended files, and re-reads the exact successor deposition response after the
upload, immediately before each deletion, and after cleanup. Immediately
before each `DELETE`, it also fetches the deposition-scoped successor file and
requires the returned filename and file ID to match the admitted target. It does not use a
separate `/files` collection response for lifecycle or deletion admission. Local
symlinks, control/query/fragment/encoded-collision filenames, duplicate aliases,
and local changes detected immediately before a PUT are rejected. The
server-supplied upload bucket must use the canonical `/api/files/<opaque-bucket-id>`
shape, with no duplicate path separators or query/fragment delimiters; deposition,
record, and collection paths are not accepted as PUT targets. Credential-shaped
remote filenames, including parameterized `Authorization: <scheme> <value>` forms,
are rejected before they can enter state or a receipt. It deletes only stable
inherited filenames from the unpublished successor and requires an exact final
filename inventory. It never addresses the published predecessor. Before the first
PUT, `upload` persists a pending attempt with the exact local inventory digest and
the exact initial successor file-ID inventory. A retry must use the same inventory
and API base; a changed or incomplete file list, an unknown remote extra, or missing
prior-attempt proof blocks reconciliation without issuing a delete. The CLI writes
this pending state before mutation, atomically replaces the state file, and records
each stable deletion so an interrupted process can safely be rerun with the
original file list without losing earlier cleanup proof. Verification also binds
each download URL to the expected record and filename, rather than trusting
same-origin bytes alone.
DELETE 204, 404, 403, 5xx, network, and other unexpected results are classified;
non-204 results are treated as conditionally idempotent only when a bounded pair
of consecutive exact readbacks proves the target absent with unchanged
identity, unpublished lifecycle, inventory, and the server revision when one
is available. When Zenodo exposes no revision field, the receipt records an
exact final-response snapshot digest including the credential-free metadata
contract; optimistic revision bindings carry the same metadata digest and the
readback proof compares it with the pre-delete response. Malformed metadata
fails closed before a receipt can be accepted. That fallback is not a
server-side concurrency token. Any failed or unstable
readback remains blocking, so rerun the same upload after a partial network interruption. A
successful upload stores a credential-free `robot-sf-zenodo-reconciliation.v1`
receipt binding the intended-inventory SHA-256, deleted filename list, and
final remote revision or exact snapshot digest/state. `verify` is read-only
and must check
the title, dataset type, GPL-3.0-only license, creator union, exact source tag,
and concept/version DOI distinction. `publish` is irreversible; never run it
for a draft with missing files, unaccepted rows, or an unbound or mismatched
concept/version DOI. The reserved DOI itself need not resolve before
publication; check resolver status separately afterward.
The publisher performs a mandatory fresh draft verification immediately before
the irreversible request. The documented legacy [Zenodo deposition actions
API](https://developers.zenodo.org/#depositions-actions-publish) exposes no
conditional compare-and-publish precondition, so a final time-of-check/time-of-use
window remains. Run the authenticated `zenodo verify` command again immediately
after publication and require its published-record receipt to pass before
treating the Zenodo record as accepted. DOI resolver status is a separate
post-publication check.

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
