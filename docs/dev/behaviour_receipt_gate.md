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
