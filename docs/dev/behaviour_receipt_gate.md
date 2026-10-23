# Behaviour receipt gate integration

## Frozen train reports (v3, shadow only)

Train scheduling, external preflight and promotion reports are **report-only**.
They never authorize merging. The existing v2 receipt gate remains authoritative:
a v3-only body still waits for its v2 receipt, and a valid v2 receipt is not vetoed
by a missing, malformed or rejected v3 report. There is no adoption switch in the
PR body, manifest, environment or CLI. Separate independent review and adoption
must precede any future use of v3 as merge evidence.

| Reported state | Meaning |
| --- | --- |
| `admitted` | Component scheduling phase; no merge authority. |
| `WAITING_FOR_TRAIN` | Existing receipt missing; admission remains nonzero. |
| `candidate` | Candidate/train freeze is available from its independent owner. |
| `executing` | Acquisition phase reported by the caller. |
| `blocked` | Evidence or identity check failed. |
| `ready` | Full shadow validation passed; no merge authority. |

Scheduling output carries `mode: shadow` and `merge_authorized: false`.
`merge_blocked` and the process exit status still reflect the existing receipt
gate. Caller-supplied phases cannot override that result. External preflight is a
separate `preflight` report, so its verdict does not alter existing admission.

### Independent owner records

`OWNER_REPOSITORY`, `FREEZE_OWNER_IDS`, `ACQUISITION_OWNER_IDS` and
`EVIDENCE_OWNER_IDS` in the base-owned `scripts/ci/frozen_candidate.py` define
trusted authority. They ship **unconfigured**. An independent adoption owner must
select a repository controlled separately from candidate/evidence authors and
account IDs for the freeze, acquisition and independent evidence-review roles.
The worker and its envelope cannot select or grant those owners. Source review
remains a separate requirement in the existing receipt validator.

Each owner record is a `record.json` asset of a published **immutable** GitHub
release in that owner repository. The resolver checks the immutable flag, exact
tag, authorized release author and asset uploader, bounded asset size and API
SHA-256 digest against downloaded bytes. Only assets carry authority; release
notes remain editable. Immutable tags cannot be reused after deletion, so deleting
a record causes refusal rather than allowing replacement at the same identity.
See [GitHub's immutable release contract](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases).
These commands read existing records; they never publish releases, tags, reviews,
artifacts or statuses. Publication and custody belong to the independent owners.

The freeze owner publishes one record **before acquisition** at deterministic tag
`artifact/behaviour-freeze-<train_id>-<candidate_sha>`:

```json
{"schema_version":"behaviour-freeze-record.v1","manifest":{"...":"complete frozen manifest"}}
```

The `behaviour-candidate.v1` manifest contains:

- `train_id`: explicit train identity, restricted to letters, digits, underscore
  and hyphen (at most 64 characters). A replacement needs a new candidate/train
  identity, new acquisition pins and new evidence acceptance.
- `candidate_sha`, `base_sha`, `candidate_tree`: exact source identities, with the
  base preserved as candidate ancestry.
- `components`: original `{pr, head_sha, reviews}` records, preserving component
  ancestry and `{head_sha, verdict: "accepted", uri}` source reviews.
- `suite_sha256`, `config_sha256`, `baseline_sha256`,
  `evaluation_policy_sha256`: canonical trusted inventory, effective resolved
  config, baseline and validator/schema/policy identities.

Canonical JSON uses sorted keys and `separators=(",", ":")`, encoded as UTF-8.
`policy_digest()` hashes the trusted validator, receipt schema and frozen policy.
The caller's local file is only a copy. `freeze` resolves the original owner
record and refuses different content for that identity, even at another output
path. Exclusive creation protects local files from accidental overwrite; it does
not establish authority or immutability.

```bash
uv run python scripts/ci/behaviour_receipt.py freeze \
  --input "$MANIFEST_INPUT" --output "$FROZEN_MANIFEST" \
  --candidate "$CANDIDATE_SHA" --expected-base "$BASE_SHA"
```

Every episode execution entry carries its existing row/source/execution identity
plus `train_id`, `manifest_sha256` and `freeze_record_id`. For each distinct
`execution_sha256`, the acquisition owner publishes the actual observed identity
at `artifact/behaviour-execution-<execution_sha256>`:

```json
{
  "schema_version":"behaviour-acquisition.v1",
  "candidate_sha":"<candidate>",
  "train_id":"<train>",
  "manifest_sha256":"<original manifest digest>",
  "freeze_record_id":"<numeric owner release ID>",
  "execution_sha256":"<actual execution identity>",
  "started_at":"<observed acquisition start in timezone-aware ISO format>"
}
```

The actual record ID is an integer. The observed start must follow publication of
the original freeze and precede publication of the acquisition record. Assembly
and promotion resolve these owner records again and compare every execution pin.
A caller-supplied hash alone cannot satisfy the acquisition-owner check. Production
acquisition must measure effective config, assets, loaded model capability and
runtime dependency identities; these read-only reports do not supply that proof.

### Evidence acceptance and assembly

Evidence acceptance binds four values: `candidate_sha`, `manifest_sha256`,
`content_sha256`, and `episode_executions_sha256`. The content digest is the full
receipt with `refute_review` removed, including all results and classifications.
The source review still passes the unchanged exact-head validator separately.
Excluding the verdict avoids a self-referential reviewed-content hash.

The independent evidence owner publishes the exact binding tuple plus
`schema_version: behaviour-evidence-acceptance.v1` and `verdict: accepted` as
`record.json` at `artifact/behaviour-verdict-<canonical binding tuple digest>`.
The resolver fetches this acceptance from the trusted owner instead of trusting
an envelope's verdict. Changing results, classifications, manifest or execution
inventory requires a new independent acceptance, even at the same source SHA.

```bash
uv run python scripts/ci/behaviour_receipt.py assemble \
  --repo ll7/robot_sf_ll7 --manifest "$FROZEN_MANIFEST" \
  --receipt "$FULL_RECEIPT" --config "$RESOLVED_CONFIG" \
  --executions "$EPISODE_IDENTITIES" --output "$EVIDENCE_JSON"
```

Assembly validates the complete existing receipt, original freeze, acquisition
records and evidence acceptance. Its external envelope is
`behaviour-candidate-evidence.v1`, with `manifest`, `receipt`, `config`,
`episode_executions`, `receipt_sha256` and `episode_executions_sha256`.
Local writes use exclusive creation; interrupted or partial files must be rejected
and replaced at a new local output path without changing the owner's record.

Publish the envelope as exactly one `evidence.json` in a same-repository immutable
Actions artifact at the candidate SHA. The shadow reference is:

```text
<!-- behaviour-change-receipt:v3
{"artifact_id": 701, "sha256": "<archive digest>", "manifest_sha256": "<original manifest digest>"}
-->
```

Shadow preflight verifies candidate-run identity, retention, metadata/archive
checksums, bounded decoded size, original freeze and independent acceptance.
The existing v2 reference remains required alongside it for merge admission.
Live producer/publisher integration and a real startup canary remain with their
existing owners; synthetic tests do not establish that those integrations exist.

### Read-only promotion check

```bash
uv run python scripts/ci/behaviour_receipt.py promotion \
  --repo ll7/robot_sf_ll7 --reference "$REFERENCE_JSON" \
  --candidate "$CANDIDATE_SHA" --expected-base "$BASE_SHA" \
  --landed "$PROPOSED_LANDED_SHA" --receipt-sha256 "$RECEIPT_SHA256" \
  --executions-sha256 "$EXECUTIONS_SHA256"
```

Promotion retains **full** receipt coverage, classifications, baseline and
native/adapter controller checks, original-freeze and acquisition checks, and
independent evidence acceptance. It reads live GitHub main and rejects a changed
base, source, tree, receipt or execution inventory. A permitted landing is the
candidate itself, a same-tree squash whose sole parent is expected main, or a
same-tree merge whose ordered parents are expected main and the candidate.
Unrelated same-tree history is rejected. The landed mapping records source
execution separately; it never transfers a verdict to a new integration candidate.

The report always has `mode: shadow` and `merge_authorized: false`. Its nonzero
result reports a failed plan without replacing the existing merge gate. The
independent promotion owner still needs serialized landing and an atomic
compare-and-swap against expected main: this read-only check cannot prevent main
from moving after its final read. Existing CI, domain, review and release gates
remain in force.

## Legacy committed transport (v2)

Behaviour PRs carry a compact `<!-- behaviour-change-receipt:v2 ... -->` JSON
header in their GitHub body. Rows and detailed classifications live in a committed
`receipts/behaviour/*.json` file, read from the event's exact PR source SHA.
The payload has `schema_version: behaviour-change-rows.v1`, `rows` and
`classifications`. It does **not** embed its enclosing commit SHA: committing a
file containing that SHA would create a circular identity requirement.

The header uses `schema_version: behaviour-change-receipt-header.v2`. It carries
`head_sha`, `scheduler` (Slurm job and source SHA), `scope_sha256`, external
`artifact` URI/SHA-256, `baseline`, `vehicle`, `exceptions`, `totals`,
`interaction_audit` and `refute_review` URI/verdict/head. `rows_artifact` gives
the committed `path` and SHA-256 of its exact bytes. `classifications` is a
compact `{count, sha256}` summary: the SHA-256 hashes the full classification
array serialized with sorted keys and `separators=(",", ":")`. Moving detailed
classifications to the payload keeps the header small even if many episodes
regress. The schema definitions for both transport objects live alongside the
full receipt schema in `scripts/ci/behaviour_receipt.schema.json`.

The gate rejects inline all-row v1 bodies, oversized bodies (GitHub's 65,536
character limit), payloads over 32 MiB, uncommitted files, symlinks, paths outside
`receipts/behaviour/`, byte-digest mismatches and inconsistent classification
summaries. It reconstructs the full receipt and applies the existing coverage,
source, baseline, execution, totals, exception and classification checks.
Changes to dirty worktree payload bytes do not replace the source Git blob.
Run the sweep and real-row audit on their recorded source commit(s), then commit
the rows/classifications. Set the PR header's `head_sha` and refute review's
`head_sha` to that final PR head. Keep `scheduler.source_sha` and
`interaction_audit.source_sha` at the commits where they actually executed.
Each execution source must be an ancestor of head. Its two-tree Git diff to head
must contain only paths under `receipts/behaviour/`, with rename detection disabled
so both endpoints are checked. Any other changed path invalidates that identity,
including docs, tests and attribute-policy edits.
The refute review must cover the final head. Committing the header is unnecessary.

The repository already uses Git LFS, so `.gitattributes` routes
`receipts/behaviour/*.json` through LFS. Commit that attribute policy before the
run; changing it afterward is a non-receipt change. The loader reads the source
blob with `git cat-file blob`, resolves its LFS pointer through Git LFS, then
verifies the payload's byte digest and declared size. It never reads dirty
checkout payloads. CI requires Git LFS and access to the referenced objects;
missing objects fail closed. The 32 MiB limit applies to decoded payloads,
including LFS objects; use only receipt data rather than raw traces in these
files, and keep traces at the external artifact URI.

## Producer flow

Use `scripts/ci/make_behaviour_receipt.py` to build the committed payload and
the compact PR-body header. The default baseline is the latest published
software release, including release records whose semver appears in the release
name rather than the tag. Selection preserves publication order. Conflicting
versions, an unversioned newest publication, duplicate highest-version records,
or tied latest publications fail closed. Older unversioned dataset publications
do not displace a uniquely identified newest release. For real evidence, run
from the SLURM host and use
`--mode sbatch --submit-only` to submit the head and baseline empty-world sweeps;
after both jobs finish, rerun with `--mode existing` and the two completed sweep
directories to write `receipts/behaviour/<id>.json` and print the header.

One-command collection flow after both sweeps complete (no `PYTHONPATH` needed):

```bash
uv run python scripts/ci/make_behaviour_receipt.py \
  --head-sha "$FINAL_OR_EXECUTION_SHA" \
  --baseline latest \
  --receipt-id "$PR_NUMBER" \
  --mode existing \
  --head-sweep-dir "$HEAD_SWEEP_DIR" \
  --baseline-sweep-dir "$BASELINE_SWEEP_DIR" \
  --job-id "$HEAD_JOB_ID" \
  --baseline-job-id "$BASELINE_JOB_ID" \
  --head-artifact-uri https://example.org/replace-with-head-sweep \
  --baseline-artifact-uri https://example.org/replace-with-baseline-sweep \
  --audit-uri https://example.org/replace-with-real-row-audit \
  --classification-evidence-base-uri https://example.org/replace-with-classifications
```

Production receipt generation uses the full gate seed set (`1001..1030`) because
that is what the validator admits today; the producer rejects seeds outside the
wider development range (`1001..1200`) and also rejects currently unsupported
development seeds above `1030` instead of emitting a receipt the gate cannot
accept. Existing sweeps must have matching `execution_<suite>.json` source,
suite, seed, completion and row-count metadata for every consumed summary.
The audit reads the summary's linked raw campaign episode, including complete
step traces and selected controller actions, rather than inferring execution
from a written summary. Missing controller evidence fails the audit; fallback
or degraded evidence produces a failed or degraded audit. Both head and baseline
are audited, and unsuccessful audits prevent receipt emission.

Receipt assembly decodes each linked raw campaign file once per invocation and
indexes its scenario/seed identities. Duplicate matches, wrong arms, summary/raw
outcome disagreements and escaping source paths retain their existing rejection
checks. The index is discarded after each load; another invocation rereads current
bytes. Keep input artifacts immutable during assembly, as with the artifact digest.

For a bounded tooling cost comparison, run
`scripts/validation/profile_receipt_loading.py --reference <pre-change-sha>
--output-dir <new-directory>`. This clones one archived trace across dev seeds
1001–1030, measures reference/current loading, checks identical serialized outputs
and preserves compressed inputs/results. It executes no simulator episodes and
emits no receipt. The [recorded cost diagnostics](../../tests/analysis/fixtures/valgate_cost_20261012/README.md)
distinguish loader timing from the incomplete MPPI acquisition profile. The full
gate's roster, seeds, horizons, controller traces, audits and independent refute
requirements are unchanged.

The artifact digest is SHA-256 of a canonical JSON mapping from every relative
sweep file path to its SHA-256 (sorted keys, `separators=(",", ":")`). Supplied
`--head-artifact-sha256` / `--baseline-artifact-sha256` values must match those
actual bytes. Preserve and publish the same files at the artifact URI.

The producer always emits `refute_review.verdict: pending_independent_review`,
including when `--refute-review-uri` supplies a candidate link. This deliberate
placeholder fails the unchanged real gate. The independent reviewer must fill
in the final-head review URI and set `verdict: accepted` after completing review;
the producer cannot grant acceptance.

Because the payload commit changes the PR head, the exact-head sequence is:

```bash
RUN_SOURCE=$(git rev-parse HEAD)
uv run python scripts/ci/make_behaviour_receipt.py --head-sha "$RUN_SOURCE" ... \
  > /tmp/behaviour-header.before-commit.txt
git add receipts/behaviour/<id>.json
git commit -m "ci: add behaviour receipt for PR <id>"
FINAL_HEAD=$(git rev-parse HEAD)
uv run python scripts/ci/make_behaviour_receipt.py --head-sha "$FINAL_HEAD" \
  --mode existing \
  --head-sweep-dir <completed-head-sweep> \
  --baseline-sweep-dir <completed-baseline-sweep> \
  --receipt-id <id> \
  --head-source-sha "$RUN_SOURCE" \
  --audit-source-sha "$RUN_SOURCE" \
  --refute-review-uri <candidate-review-url>
```

Paste the final `<!-- behaviour-change-receipt:v2 ... -->` block into the PR
body. Keep `scheduler.source_sha` and `interaction_audit.source_sha` at the
actual execution commits; only `head_sha` and `refute_review.head_sha` should
move to the final receipt commit.
`--head-source-sha` explicitly identifies the recorded sweep source when
`--head-sha` names its receipt-only descendant; the existing gate's ancestor
check verifies that relationship. Baseline metadata must match the published
tag source exactly and cannot use this descendant allowance.

## Scope and dependency rule

Production triggers cover planner/adapters, simulator/dynamics, maps/scenarios,
benchmark configuration and writers/metrics, algorithm/baseline/planner/robot
configs, model weights/registry, sensors, pedestrian logic, shared runtime code,
training, prediction, feature extractors (including `robot_sf/feature_extractor.py`),
`robot_sf/ped_ego/`, all `fast-pysf/` code and benchmark
runners under `scripts/`. Markdown files are exempt under every prefix, as are
docs-only changes and the owner inventory itself. Mixed planner/inventory PRs
remain in scope. The top-level extractor is imported by
`feature_extractors/config.py`. `ped_ego/unicycle_drive.py` is used by
`gym_env/unified_config.py` and `sim/simulator.py`, and pedestrian environment
state comes from `ped_ego/pedestrian_state.py`. The current checkout has no
benchmark/planner runtime imports of the additive `robot_sf/core/` facade, so it
remains exempt; revisit this decision when those consumers adopt it.

A separate base-owned `DEPENDENCY_RECEIPTS_ENABLED` rule examines `uv.lock`
and `pyproject.toml` (root and `fast-pysf/`) through the existing Dependabot policy parsers. A change to
`torch`, `stable-baselines3`, `numpy` or `gymnasium` requires a receipt; other
package bumps, comments and metadata edits stay exempt. Added/removed packages
and changed dependency-group requirements count. Comparison uses the source's
merge base, so runtime dependency updates made only on main do not taint a stale
branch. Ambiguous/malformed dependency inputs fail closed. An author ruling can
change this one constant without switching off production path triggers.

## Trusted policy and source checkout

CI keeps GitHub's default synthetic merge checkout with complete history/tags.
Existing seed-holdout and added-file guards therefore compare the combined tree
against the current base. `BEHAVIOUR_PR_HEAD_SHA` separately binds the receipt
to `github.event.pull_request.head.sha`; source and merge identities are distinct.
The published baseline resolves to the latest published software release tag,
excluding model and other artifact releases.

The workflow creates a detached worktree at the immutable event base SHA and
executes **that base's** validator, schema, owner inventory, algorithm registry
and dependency parser. It independently collects the source diff from Git rather
than trusting a head-owned changed-file list. Head policy edits cannot weaken
admission in the same PR. The head contract checker does not import receipt
policy; the base validator is a separate blocking workflow step. Before the gate
is first installed on base, CI reports it inactive and never substitutes head
policy. Gate changes take effect on subsequent PRs after landing on base.

## Owner inventory and evidence limits

The release integration inventory at
`configs/benchmarks/releases/behaviour_gate_0_1_0.json` binds 14 named arms and
51 scenario/map slots from the empty-world sweep's `main` and `width` inputs.
Scenario names identify slots, including distinct configurations sharing SVGs.
It includes arm-to-registry mappings, repository-relative map paths and
campaign/matrix/map digests. Receipts bind its canonical JSON SHA-256.

The vehicle is `differential_drive_r1m`, the current differential-drive body with
1.0 m radius. The existing 2.0 m narrow doorway remains an infeasible-by-design
exception, linked to its declaration at the integration base. Release integration
owns inventory maintenance; missing inventory blocks behaviour admission.

Each row binds its algorithm and command execution mode. Native-capable
algorithms require native commands; adapter-only algorithms require supported
adapter commands. `controller_executed` must be true and `fallback`/`degraded`
false. Every new failure/collision must be classified with evidence. No success
rate is required, and this gate does not test pedestrian interaction.

Digests establish byte identity, not scientific truth. A fabricated receipt can
be internally consistent: independent review must verify its linked job,
execution artifacts, body/config/source provenance and classifications. Review
plus the digest/binding checks remain the safeguards against receipt forgery.
Synthetic test rows prove validator behaviour only.

## Reconstructing a baseline with repaired acquisition tooling

The historical width template requests twelve MPPI steps from an eight-output
predictor. Receipt acquisition derives the two predictive width bindings from
the same source checkout's main release template, which uses the published
eight-step, 0.1-second profiles. Historical templates and released algorithm
configs remain unchanged. Both suites check actual checkpoint output before
any episode collection; unsupported requests are rejected rather than capped.

When the published baseline predates this acquisition repair, pass
`--sweep-harness-source-sha <full-commit-sha>` to the producer in `local` or
`sbatch` mode. It installs that commit's sweep script in **both** isolated
source checkouts, using the sweep's existing tooling-only source exception.
Runtime HEAD stays at the requested source, including the immutable release
source for the baseline. Existing unrelated script edits and conflicting
provenance refuse instead of being overwritten. Use a fresh work directory
for each source/harness pair.

Each sweep artifact retains `harness.json` with runtime source, harness source,
and original/replacement script digests. The launcher verifies the selected
committed digest, then compiles and executes that same captured byte buffer
with the source path retained as `__file__`. The direct CLI dispatches through
the same captured-buffer contract. An imported `main()` refuses collection and
reanalysis unless `_SWEEP_SOURCE_BYTES` was supplied by that dispatcher; a
separate pathname read cannot establish imported execution identity.
Execution metadata uses its loaded digest for both suites; a later source-file
replacement refuses completion
instead of certifying the replacement. Collection rejects disagreement and
includes the reconstruction in the receipt's declared baseline differences.
`--summarize-only` requires authoritative original execution metadata with
matching source, suite, development-seed selection and loaded harness identity;
it preserves that provenance and refuses missing or conflicting identity before
rewriting outputs. A campaign summary alone cannot reconstruct execution identity. Preserve these manifests with the raw
rows and derived campaign configs. The repaired width binding differs from the
historical width controller beyond its horizon: it uses the published profile
as a whole. These are new development reconstructions, not unchanged historical
outcomes. Both sides must be rerun with the repaired harness; old width rows
cannot be substituted. Use all seeds 1001–1030, all arms and slots, the required
Slurm environment, raw-row audit and independent final-head refute review.
A check-only preflight is not a behaviour receipt, and this option changes no
trusted admission rule or review requirement.

## Incremental acquisition contracts and recovery

The sweep compiles a `receipt-contract.v1` before any trajectory. It expands the promotion
roster independently (14 arms × 51 slots × development seeds 1001–1030 = 21,420 IDs), checks
selected combinations, enters the production batch preflight, and constructs every selected
arm/slot through the episode resolver, planner builder and simulator factory. Readback includes
live simulator/robot settings, owned planner configurations and the loaded predictor's declared
and measured forecast capability. Unsupported horizons fail; they are never clamped. Runtime
input checksums, model-registry checkpoint digests, row/metric contracts, output publication and
available disk are checked before collection. `--minimum-free-bytes` declares a storage floor;
it is a capacity check, not a prediction of the final artifact size.

Submit preflight through the existing producer, with the full development roster and immutable
candidate/baseline sources:

```bash
python scripts/ci/make_behaviour_receipt.py \
  --head-sha "$CANDIDATE_SHA" --baseline "$BASELINE_TAG" --receipt-id "$RUN_ID" \
  --mode sbatch --submit-only --check-only --work-dir "$ARTIFACT_ROOT"
```

The submitted script pins its immutable checkout with `sbatch --chdir`, checks its helper path,
and limits library threads. Set `UV_BIN` to an absolute executable path when submitting, or the
producer detects `uv` on its own PATH and embeds the resolved absolute path. The batch shell
checks that executable, then runs `uv sync --frozen --all-extras` before
`uv run --frozen --all-extras` so preflight and production planner imports have their dependencies.
Check-only/canary jobs have a fourteen-minute limit; canaries accept
at most two workers. Supply site-specific scheduling and installed-environment settings outside
this public repository. Inspect terminal scheduler status and artifacts; submission success is
not preflight success. Both full-roster contracts must exist, match their requested sources, and
agree on row schema and metric meanings before the producer submits a long paired acquisition.

Use `--canary` for diagnostic construction/reset/short-step probes and terminal-state fixtures.
Use `--arms` and `--scenarios` for a handful of probes in a tiny scheduler allocation; the report
records the selected coverage. All canary scenario identities begin with `canary/` and their
outputs live below `canary/`, carry `receipt_eligible: false`, and cannot fill scientific slots.
A selected canary does not establish full arm/slot coverage. The unfiltered canary exercises the
whole roster; scheduler sizing must still obey the task's compute limit.

Execution, scoring and comparison identities use domain-separated canonical SHA-256 records.
The conservative execution identity includes the exact source SHA, harness digest, installed
runtime, resolved inputs, assets and execution-time metric definitions. Continuation requires
that identical identity and output root. This does not authorize cross-source caching.

For the first complete seed, submit the same full contract with `--first-seed-only`. Continue
with the same source, roster, seeds, environment and output root, omitting that flag. Dispatch
selection is separate from execution identity. The first seed's full-length observations remain
in the journal and count toward the final matrix; the deliberately partial stage cannot pass
receipt coverage. A timeout or collision is a completed observation and is never retried.

Validated per-episode completion JSON is written to a temporary file, checked, synced and
published without overwriting a different record. Parallel futures publish immediately rather
than waiting for an arm to finish. JSONL is a rebuildable view. On continuation, valid completion
records rebuild it before the normal missing-job filter; invalid records are preserved under
`invalid/` and only those missing/invalid observations execute again. Banked rows also reconstruct
arm reporting, including interruptions before `summary.json`. Existing JSONL without completion
provenance is refused; it is never assigned a new producing source. Keep the artifact directory
outside the candidate checkout and preserve it before retiring any worktree.

Comparison can be repeated without invoking a simulator:

```bash
python scripts/validation/compare_behaviour_rows.py \
  --head-sweep-dir "$HEAD_ROWS" --baseline-sweep-dir "$BASELINE_ROWS" \
  --output "$COMPARISON_JSON"
```

This reads immutable rows and their producing contracts, requires full coverage, checks schema
and definition compatibility, preserves each input's actual execution-time scoring source,
and publishes an immutable comparison artifact. It does not rescore arbitrary old traces or
assign the current scorer to historical metrics. Row and contract digests are checked again
before publication. Changed comparison tooling changes comparison identity; the original scoring
identities remain bound to their producers. Events requiring classification remain unresolved
and the output explicitly says `admission: not_evaluated`.

All existing receipt/audit/classification/exact-head review gates remain in force. This tooling
provides no partial-coverage admission, review exemption, or automatic promotion. For a historical baseline, explicitly pin `--sweep-harness-source-sha` to this reviewed
acquisition tooling. The installer copies its receipt modules and applies a bounded acquisition
patch to the historical batch/writer/resume functions. It preserves the historical policy and
simulator implementation. `harness.json` binds original and installed digests for every file;
the sweep independently reconstructs the overlay from committed objects before exempting those
exact bytes from runtime-source verification. An incompatible patch or unrelated local edit is
refused. This is an explicitly recorded development reconstruction, never unchanged archived data.
Non-finite observation sentinels (for example infinite empty-world clearance) use versioned,
lossless numeric tags in completion JSON and are restored when rebuilding the production JSONL.

For historical runtimes without a declared definitions digest, the contract records that absence
and hashes the actual metric implementation files. It never copies current metric code over the
historical implementation. Paired preflight rejects incompatible semantic source bundles even
when a version label is unchanged. Run standalone comparisons from the reviewed acquisition
checkout; historical runtime overlays contain only the modules needed for acquisition.
