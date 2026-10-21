# Behaviour receipt gate integration

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
`--mode sbatch --submit-only --sbatch-node <node>` to submit the head and baseline
empty-world sweeps on the same single node;
after both jobs finish, rerun with `--mode existing` and the two completed sweep
directories to write `receipts/behaviour/<id>.json` and print the header.

The node argument must be one literal hostname from the checked-in
[`behaviour_receipt_nodes.json`](../../scripts/ci/behaviour_receipt_nodes.json)
inventory; node lists and range expressions are refused before submission.
The inventory is owned by the trusted base policy and maps scheduler node names
to CPU model, architecture and node class (partition). Unknown nodes are refused.
Updating this mapping requires review; job-written CPU claims cannot update it.

Each job creates `sweeps/<side>/<SLURM_JOB_ID>` with an exclusive `mkdir` before
writing any evidence. An existing directory causes failure. Resubmission receives
a new job ID and a new directory; it never overwrites previous execution evidence.
Submission manifests are separately keyed by job ID. After the sweep command
succeeds, its summary and raw rows receive `scheduler_job_id` and `scheduler_node`
from the Slurm allocation environment. Existing or inconsistent row bindings fail.
Local execution cannot produce an admissible scheduler-backed receipt.

Collection queries `sacct` through the authenticated, read-only scheduler endpoint
on `imech192`, using allocation records rather than job-written state. Every
selected shard must have exactly one `COMPLETED` record with exit code `0:0` and
a single known node. It refuses mixed nodes within either side. Both sides must
match node, CPU model, architecture and node class. Every summary and raw row's
job ID must match its job-keyed directory; node claims and source execution
contexts must agree with the authoritative allocation and inventory.

The four job-written identity files are claims only. Collection checks them
against scheduler records and the trusted inventory, and embeds an additional
`execution_scheduler.json` verification snapshot with inventory artifact path and
digest. SHA-256 digests preserve bytes; they do not grant job-written evidence
authority. Admission independently queries the same scheduler endpoint and
compares the snapshot to fresh authoritative records, then checks both job/node
bindings on every joined row. Rehashed invented CPU claims remain invalid.

Missing, unavailable, failed or inconsistent scheduler evidence explicitly
invalidates collection and admission with `hardware comparison invalid`, even
when outcomes are unchanged. Collection exits without writing a receipt, header,
audit or outcome report. There is no unsigned offline fallback. Hosted admission
requires authenticated scheduler read access; absence blocks admission rather
than silently trusting the stored snapshot. Read-access integration is tracked
under the gate owner issue #10000 and must be available before promoting this
policy for behaviour-changing PRs.

Legacy receipts without scheduler verification or row allocation bindings cannot
be admitted. Acquire fresh job-keyed evidence; do not backfill observations or
bindings into frozen artifacts. CPU-dependent inference can still differ across
hosts. Same-node selection does not promise numerical equality or admit a
tolerance, and it waives no classification or real-row audit requirement.

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

For a sharded side, pass its parent directory and a comma-separated allocation
list, for example `--head-sweep-dir "$HEAD_SIDE_DIR" --job-id "$HEAD_JOB_IDS"`.
Only the selected job-keyed directories enter the receipt artifact digest and
row comparison. Their combined coverage must still exactly satisfy the gate;
duplicate rows across shards are refused.

Allocation stamping writes a separate, exclusively created
`execution_binding.json`; it never edits raw episodes, summaries or their
producer sidecars. The binding hashes every comparison input and identifies
its job and node. Both normal raw sidecars (`.provenance.json` and
`.manifest.json`) are required. Collection invokes the repository publication
digest verifier against their sealed raw hashes and the allocation binding;
any mismatch invalidates the comparison before a receipt or outcomes are emitted.

Collection also preserves exact source bytes in job-keyed `.tar.gz` bundles
under `receipts/behaviour/<receipt-id>-sources/`, referenced and hashed by the
receipt payload's `source_artifacts`. Commit these bundles with the payload
(use Git LFS for large evidence). Admission reads their exact committed bytes,
verifies the artifact/LFS digest, rejects unsafe archive entries, and reruns
the same producer-seal and binding verifier. It compares the joined outcomes
and allocation identities against those verified raw rows. Dirty checkout
files and a collector's verification flag cannot substitute for source evidence.
The compact payload retains its 32 MiB limit; each source bundle is limited to
512 MiB compressed and 2 GiB expanded. Existing identical bundles are preserved;
conflicting bytes at an existing evidence path are refused.

Allocation binding v2 also records `source_sha` and `arm_role` (`head` or
`baseline`). The producer's run and per-row source commits, raw `git_hash`,
explicit arm-role claims and recorded execution context must agree with that
binding and with the receipt's independently selected source/node. The producer
hostname and CPU model must match the scheduler and trusted inventory; its
canonical context digest must verify. Head and baseline must use different
producer commits. Consistent hashes alone cannot relabel baseline bytes as head
evidence. Older bindings cannot be backfilled into frozen artifacts.

Collection and admission share `verified_raw_rows`, `derive_valid_comparison`
and `require_valid_execution`. They verify metadata and summary/raw coverage,
reconstruct allocation context in memory, and apply the same registry algorithm,
command mode, controller, trace and fallback/degradation checks to both sides.
Admission re-derives the full joined-row projection and totals from verified
raw evidence, then compares every projected field with the receipt. A healthy
payload cannot override invalid execution in a validly re-sealed source.

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

### Required source outcomes

Collection and admission use `source_outcomes` in
`scripts/ci/behaviour_receipt.schema.json` for every raw episode and sweep summary.
Success must be an explicit boolean, collisions a nonnegative integer, and steps a
positive integer. The schema records the accepted producer field paths. If several
paths are present, every value must be valid and they must agree. Raw and summary
observations must agree on all three fields. Missing fields, nulls, coercible
strings, invalid counters and conflicting aliases invalidate the comparison before
outcomes are reported; explicit `false` success and zero collisions remain valid.
Partial historical evidence must be reacquired, never backfilled into sealed files.
