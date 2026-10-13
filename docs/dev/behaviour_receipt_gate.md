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

## Required outcome evidence

The base-owned `scripts/ci/receipt_outcomes.py` is the single outcome contract
used by collection, classification, raw comparison and committed admission.
Every written raw episode requires `metrics.success` (a JSON boolean),
`metrics.collisions` and `metrics.total_collision_count` (equal nonnegative
integers), and top-level `steps` (a positive integer). Every summary requires top-level `success`, `collisions` and `steps`
with the same types and values as its raw episode. No missing value becomes
false or zero; missing fields refuse even when both arms omit them and their
producer seals and compact hashes are coherently regenerated.

Explicit success/collision aliases must be correctly typed and agree with the
canonical value. They never replace a missing canonical field. The benchmark
producer emits both mandatory raw collision counts from the same aggregate;
a missing or contradictory count refuses.
Outcome reads do not alter original sealed rows. All behavioral fields,
including outcomes, unknown fields and presence changes, remain in the exact
canonical byte comparison. The existing raw legacy NaN/Infinity token policy
for other metrics, strict duplicate/overflow checks, full receipt requirements
for releases, and hardware/source/execution/coverage guards remain in force.

The equivalence classifier separately owns a twelve-entry literal map of
reviewed preparation and diagnostic blobs in `REVIEWED_BASE_ONLY_BLOBS`.
Each exception requires the current-main committed file to be a regular Git
blob with exactly the reviewed SHA-256. The candidate's merge-base diff must
not include that path, even if its final bytes equal main. Edits, deletions,
renames and type changes refuse. Protected paths are checked before exceptions,
and a separate reviewed neutral transition is still required. These entries do
not exempt directories, releases, runtime fields or downstream admission checks.

The finite `REVIEWED_CLOSED_TREES` inventory unconditionally requires each
declared current-main tree, even when both compared trees omit it. Its one
production entry pins the current-main
`docs/context/robsweep` snapshot: exactly 731 recursive `100644 blob` members,
its Git tree object ID and the SHA-256 of the raw tree object. Only members of
that verified committed snapshot receive the exception. Any candidate history
change at the root or below it refuses before `.md`, test or receipt skips,
including a change whose final bytes equal main. Git diff and tree paths are
read as binary NUL-delimited records, preserving newline and tab filenames.
Different current-main tree bytes, added members, root/file types or modes
refuse. The catalog has one current-main byte pin; an older catalog is not an
alternative. These checks never grant a directory or documentation-prefix waiver.

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

## Candidate default-path equivalence route

`default-path-equivalence:v1` is a PR-only alternative for a narrowly classified
change whose default benchmark behavior is declared unchanged. This proposal
requires independent review and explicit author promotion of the base-owned gate;
its existence on a feature branch does not grant authority to queued PRs. It does
not replace the full 21,420-row latest-published-release receipt for releases or
for changes outside its scope.

The candidate evaluates **all 14 arms and all 51 scenario slots**, using dev seeds
**1001, 1002, 1003**: 2,142 episodes per source. The baseline is the exact current
main commit, acquired once and shared across candidate PRs. A moved main invalidates
the baseline. At admission the gate queries GitHub's live `git/ref/heads/main`
endpoint: the immutable event base and receipt baseline must both equal that
current remote main commit. The event base or a local `origin/main` ref alone is
never freshness proof. The private producer performs the same live-main check
before consuming an acquisition; an API/authentication failure refuses it. Each head runs its exact source; an acquisition script may come from
a separately pinned tooling commit, but this never patches or normalizes runtime
source. Base and head must have the same observed node, node class, CPU model and
architecture, using #10365's authenticated scheduler identity and pair validator.
Job-written hardware files are claims: successful literal Slurm allocations and
the trusted base node inventory supply the node, CPU and partition authority.
Each source's files must live in regular directories named by the actual job IDs.
Raw and summary row allocation fields and any known nested execution context are
validated before comparing episode bytes; compact paired rows retain each side's
`scheduler_job_id` and `scheduler_node`, independently checked again at admission.

Three seeds reduce a sweep to one tenth of the full development-seed acquisition.
The prior planning estimate was 2.088 node-hours per source, hence 2.088 marginal
node-hours per candidate head with a shared baseline. This is an estimate, not a
measured bound: the acquisition must report actual wall time and node-hours. A
reduced slice proves equality only for that slice; it supports a neutral-default
claim backed by source classification, not a scientific or release conclusion.

### Scope refuses uncertainty

The validator independently computes the **two-tree** diff from current main to
the executed head. The PR merge-base diff alone cannot hide default fixes missing
from an older head. A declaration never exempts unknown code. The only policy-drift exception is the
finite base-owned set `scripts/ci/behaviour_receipt.py`,
`behaviour_receipt.schema.json`, `make_behaviour_receipt.py`,
`default_path_equivalence.py`, `behaviour_receipt_nodes.json`, and `receipt_provenance.py` (all under
`scripts/ci/`), plus exact `scripts/dev/merge_queue_gate.py` and
`.github/workflows/merge-queue-gate.yml`, standalone synthetic saved-row profiler
`scripts/validation/profile_receipt_loading.py`, `.github/workflows/scripts-catalog.yml`,
`docs/dev/ci_job_value_audit_2026_10_12/observations.csv`,
`docs/dev/ci_job_value_audit_2026_10_12/summary.json`, and `.gitattributes`.
The two saved CI-audit data files have no simulation consumers; this is not a
general documentation JSON/CSV waiver. Only the two exact
receipt-owned LFS rules may drift in base `.gitattributes`; model or other rules
remain uncertain. Missing/older copies in an
old queued runtime are irrelevant only when the PR branch has not edited/deleted
them relative to its merge base. These tools and the trusted node inventory have no default rollout consumers.
Actual PR policy edits remain unknown; no acquisition launcher drift is exempt.

Protected directory prefixes are `robot_sf/planner/`, `baselines/`, `sim/`,
`robot/`, `nav/`, `sensor/`, `ped_npc/`, `ped_ego/`, `common/`, `prediction/`,
`models/`, `feature_extractors/` (all relative to `robot_sf/`), plus `model/`,
`fast-pysf/`, `maps/`, `configs/algos/`, `configs/baselines/`, `configs/planners/`,
`configs/robots/`, `configs/scenarios/`, `configs/benchmarks/`, and
`robot_sf/benchmark/map_runner/`, `map_runner_policies/`, `safety/` (the latter
three relative to `robot_sf/benchmark/`). Protected individual files are
`robot_sf/feature_extractor.py` and these files under `robot_sf/benchmark/`:
`metrics.py`, `metric_definitions.py`, `path_utils.py`, `exact_repeat_campaign.py`,
`event_ledger.py`, `termination_reason.py`, `algorithm_metadata.py`,
`paired_effect_metric_contract.py`, `social_compliance.py`,
`public_requirement_events.py`, and `interaction_exposure.py`.

Every other executable/configuration change is refused unless it matches the
explicit inventory in `scripts/ci/default_path_equivalence.py`. This means, for
example, `robot_sf/gym_env/robot_env.py` and unlisted benchmark modules are refused
as uncertain, even though they are outside the named protected prefixes. Markdown,
tests and receipt-only files retain their existing non-production status. The
current proposed inventory covers only these precise transitions:

| Surface | Reviewed default-neutral boundary |
| --- | --- |
| `benchmark/camera_ready/_run_state.py:_episode_identity` | Exact integer validation of saved episode seeds (#10342); all other module AST is unchanged |
| `gym_env/reward.py:_reward_with_terms` and `gym_env/snqi_proxy.py:compute_snqi_step_proxies` | The exact reward-only running-jerk to step-jerk transition (#10351) |
| `gym_env/reward.py`, `gym_env/snqi_proxy.py` | Exact whole-module TTC reward-signal transition (#10347) |
| `gym_env/reward.py` | Exact whole-module opt-in curriculum weight merge/logging transition (#10355) |
| `training/released_warm_start.py`, `prepare_released_ppo_warm_start.py` (added under the training-script directory) | Exact added standalone training helpers (#10357), absent from base, without default import/registry edits |
| `training/scenario_family_weights.py`, `train_ppo_with_family_weights.py` (added under the training-script directory) | Exact added standalone training helpers (#10358), absent from base, without default import/registry edits |
| `training/predictive_supervision.py` and four named predictive collection/pipeline scripts | Exact whole-module provenance/training-only transitions (#10346); default runtime never imports these modules |
| Eight named predictive training YAML files | Exact byte transitions in the inventory (#10346); these explicit training inputs are never executed by this route |
| `benchmark/artifact_publication.py` | Exact saved-row SNQI-v2 publication validation transition (#10360); episode/metric producers are unchanged |
| `analysis_workbench/release_row_anomalies.py` | Exact reversed offline analyzer identity-validation drift in old queued heads; only saved-row audit consumers |

Paths without `scripts/` above are relative to `robot_sf/`. The four predictive
scripts are `build_predictive_mixed_dataset.py`, `collect_predictive_hardcase_data.py`,
`collect_predictive_planner_data.py`, `run_predictive_training_pipeline.py` under
`scripts/training/`. The eight YAML names are listed in `NEUTRAL_DATA_TRANSITIONS`;
there is no training-config prefix exemption. An inventory entry authorizes only
its exact before/after AST or byte hashes, not future edits to the filename.
AST hashes retain signatures, imports, decorators, docstrings and every body
expression. Canonical encoding omits only empty optional AST fields that Python
versions add. Any inventory extension changes gate authority and requires review
and author promotion. Source eligibility still requires measured byte equality;
a reward-only source change can and should fail if it changes emitted reward.

### Byte equality and evidence custody

Both rule-based and learned arms require identical canonical episode bytes.
Claims, metadata, summaries, bindings, headers and compact receipts require finite
standard JSON numbers. Only `campaigns/**/episodes.jsonl` may retain the existing
producer's explicit `NaN`, `Infinity` and `-Infinity` tokens as typed behavioral
values. Episode serialization preserves those tokens distinctly from null,
strings and finite numbers; no metric is replaced or removed. Any non-finite
value within an excluded transport leaf refuses before removal. Numeric literals
that overflow the float decoder to infinity refuse in every input, including raw
rows. Finite numbers retain their existing types; strings containing the token
names remain ordinary data. Independent raw parses compare canonical token bytes
rather than floating-point self-equality. The same guards apply to private
comparison and committed PR/queue admission. Every input rejects duplicate JSON
object keys at every nesting. Generic receipt and cryptographic serialization
remain finite-only; the existing full route is unchanged.
A coherently resealed ambiguous field cannot disappear under last-key-wins parsing.
The existing full-route JSON parser is unchanged. The
new loader hashes the untouched raw episode, including the presence and value of
any raw `execution_status` field. The authoritative summary's `written` transport
status is supplied only to a separate controller audit view; it never replaces or
adds a field in the row being hashed. The existing full-route loader retains its
previous transport projection. There is no tolerance, rounding, trace truncation, reward removal or learned-arm exception.
The comparison retains requested/applied actions, complete controller step traces,
reset and step poses, pedestrian states, collisions/events, termination flags,
rewards, metrics, configuration identities and every unknown field. Missing traces,
fallback/degraded execution, incomplete coverage, node/CPU differences, or one
changed behavioral byte refuse the route.

Guarded PPO retains its source-defined `mixed` command mode; the receipt never
relabels it as native. This case requires the guarded algorithm and planner
identity, both registry and emitted command capabilities, the named guarded
adapter, a complete trace, finite selected and applied unicycle command axes,
and the source-defined action conversion at every step. Other mixed algorithms
remain refused. The two shield-decision calibration leaves and the shield
contract's calibration/fallback declarations are recognized only with their
exact source schemas, prediction source and values (`not_calibrated` and
`RiskDWAPlannerAdapter`). Arbitrary statuses or neighboring fault fields still
refuse. These declarations remain in the canonical comparison bytes.
The guarded runtime's planner type, policy semantics, event-ledger planner,
scenario algorithm and paired-effect planner identity must all be present and
agree with the native guarded controller. Its fallback diagnostic identifies the
configured `RiskDWAPlannerAdapter`. Both last-decision copies must agree on a
source-defined standard label and its policy: `stop_safe`/`stop_best_effort` use
`stop`; the five standard goal/PPO/fallback labels use the configured RiskDWA
adapter. The reviewed matrix has no optional prior; prior availability, prior
labels, uncertainty branches and unknown labels refuse rather than acquiring a
generic mixed-mode allowance.

Existing authenticated guarded-native decision semantics remain unchanged.
Best-effort, no-admissible-command and hard-constraint outcomes remain present
and byte-compared; a byte-equality receipt is not a safety certification or a
claim that each episode succeeded. Actual controller substitution, load failure
and runtime degradation still refuse.

Only these exact transport/provenance leaves are excluded:

- `git_hash`
- `provenance.git_hash`, `provenance.planner_commit`, `provenance.artifact_uri`,
  `provenance.artifact_sha256`
- `result_provenance.repo_commit`
- `event_ledger.software_commit`
- `algorithm_metadata.paired_effect_native_trace.source_commit`
- `algorithm_metadata.obstacle_force_law.sites.fast_pysf.source_commit`
- `algorithm_metadata.obstacle_force_law.sites.socnav_social_force.source_commit`
- `algorithm_metadata.paired_effect_metric_producer.source.source_commit`
- `timestamps.start`, `timestamps.end`
- `wall_time_sec`, `timing.steps_per_second`
- top-level `scheduler_job_id`, `scheduler_node`, after allocation validation and
  retention as independently checked per-side compact-row bindings

The eight exact runtime Git alias leaves are checked against their side execution
source before exclusion. Optional SocNav site metadata may be absent; every
other site/configuration/hash/metric field remains compared. These aliases are
producer-defined in `map_runner_episode.py` and `paired_effect_metric_contract.py`;
there is no wildcard source-commit rule. Their source/allocation identities and raw bytes remain bound by verified scheduler
records, the trusted node inventory, hardware records, exact
execution source, and committed regular/LFS sealed source archives. Admission verifies
normal producer provenance/checksum seals and write-once v2 allocation bindings
against the exact side source and head/baseline role, then rebuilds every behavioral digest and execution assertion from the raw rows.
Each summary must resolve uniquely to a canonical bound raw `episodes.jsonl` file
with matching identity/outcome. Original runtime repository commit fields must
match their side source; historical planner/model commits remain provenance, not
runtime source assertions. Sealed execution manifests independently establish
source, completeness, seed slice, pinned tooling and actual archived wrapper bytes. No entire provenance/timing
subtree is removed: unknown descendants remain compared. JSON object order and
whitespace are canonical transport representation; numbers are never rounded and
all projected values retain their JSON types.

Fresh MapRunner acquisition must install `receipt_producer_capture.install_capture`
before collection, using the shared strict outcome reader. MapRunner natively emits
the provenance seal; the capture hook validates that successful, complete original
seal and every canonical outcome before asking the runtime manifest writer to emit
the second checksum seal. Original raw and provenance bytes remain unchanged.
Existing secondary files, partial rows, writer errors and conflicting digests refuse.
The hook never repairs or re-seals a historical failed allocation.

Execution validation follows the published per-scenario dispatch only when its
exact source/profile digest and selected arm are proved. The finite inventory in
`receipt_arm_execution.py` covers the two adaptive hybrid profiles that explicitly
select ORCA for `francis2023_leave_group`. This is not a general algorithm alias or
fallback exception. Both normal producer seals and raw execution traces must agree
with the proved dispatch; unexpected scenarios, arms, profiles and algorithms refuse.
For this finite proof, raw `algo` and nine native identity leaves are mandatory:
`algorithm_metadata.algorithm`, `algorithm_metadata.canonical_algorithm`,
`algorithm_metadata.planner_contract.planner_id`,
`algorithm_metadata.planner_kinematics.adapter_name`,
`algorithm_metadata.planner_runtime.planner_type`, `algorithm_metadata.policy_semantics`,
`algorithm_metadata.paired_effect_metric_producer.pairing.key.planner`,
`event_ledger.planner`, and `scenario_params.algo`. Each must identify the actual
ORCA execution (including its adapter and policy semantics). An optional top-level
`algorithm` must agree if present. Missing or contradictory identity leaves and
ambiguous duplicate JSON keys refuse proof, including coherently re-sealed inputs.
Archive admission reconstructs this proof from the sealed sources before restoring
its in-memory attributes; JSON headers and projections cannot manufacture it.
All fourteen arm slots and every canonical behavioral byte remain compared.

The producer verifies complete sweep execution metadata, raw-row/summary identity,
full arm/slot/seed coverage, and the acquisition tooling's file SHA against the
pinned Git blob. Every suite must record `tooling_sha`, `tooling_path`,
`tooling_file_sha256`, and `private_harness_sha256`; every base/head suite must use
the same acquisition wrapper digest. Each raw archive must retain the original
submitted wrapper as a regular `acquisition_harness.py` file in each job-ID
directory; its actual byte SHA256
must match every suite's metadata. The archive also retains its original
`execution_request.json`; its schema, source, arm role, literal shard job ID and
requested node must match the verified allocation. All request fields must equal
the SHA-bound hardware request after replacing only its aggregate job ID with
the shard's literal job ID. The archive retains its manifest. Compact comparison rows bind each behavioral digest to a raw archive
path (including the job-ID prefix for a multi-shard archive) and arm/scenario/seed
selector. A numeric job prefix must match that row’s verified job; multi-job
archives require the prefix. Direct `campaigns/...` paths are accepted only for
a single-job archive whose root is its job directory. The diagnostic contains every compared episode,
including all differing episode digests, before refusal; it cannot satisfy the
receipt schema.

For separately acquired sweeps, use `scripts/ci/make_behaviour_receipt.py` with
`--receipt-route default-path-equivalence --mode existing --base-ref origin/main
--tooling-sha <exact-tooling-commit> --seeds 1001 1002 1003`, plus exact head/source,
paired sweep directories, real job IDs, scope and durable raw artifact URIs. Use a
durable `--work-dir` for the comparison diagnostic. Local acquisition is forbidden. The parent sbatch engine supports
`--submit-only --seeds 1001 1002 1003` with single-node pinning; its native output
alone lacks the new route's separately pinned tooling/wrapper manifest and is not
admissible as equivalence evidence. This candidate consumes previously submitted
single-node acquisitions only when their actual archived wrapper and immutable
manifests satisfy all of the requirements above. It never adds unverified tooling
claims while consuming an acquisition. Submission automation that emits this
manifest may be added separately; the tested route here is the evidence consumer.

For private measurements before authority promotion, add `--comparison-only`.
This performs the same exact-source, scope, hardware, tooling, complete-controller
and byte checks, writes the complete diagnostic and returns 0 for equality or 1
for changed episodes. It emits no receipt or PR-body header and needs no invented
public artifact URI. Preserve the raw sweeps, wrapper and diagnostic in durable
private custody; durable HTTPS evidence is still mandatory when a receipt is
later proposed for gate admission.

The producer writes a `default-path-equivalence-header.v1` inside the
`<!-- default-path-equivalence:v1 ... -->` marker. Its initial review verdict is
`pending_independent_review`, so admission refuses it. The committed regular/LFS
payload under `receipts/behaviour/` must match the header's byte SHA256. It contains
execution sources, hardware, exact scope/transport inventory identities, raw archive
identities and all 2,142 comparison rows. The header carries the final PR head and
independent review, avoiding a self-referential final-commit hash in the committed
payload. Any commits after execution may change only receipt files. The accepted
review must bind that final head, a durable review URL and the complete comparison
digest. Mixed/duplicate receipt routes are refused. Any explicit reduced-route
marker invokes all of its guards before the legacy out-of-scope shortcut, even
for an executable path outside the legacy trigger inventory. Ordinary neutral
changes without that marker retain the existing out-of-scope behavior.

The header and committed payload explicitly bind `admission_context=pull_request`.
The actual validator CLI rejects release and other non-PR event payloads without
requiring an environment switch. Hosted `pr-contract-check` invokes it only for
PR events. Actual software promotion and benchmark publication use their separate
strict candidate/rights/full-release-acceptance contracts; neither consumes this
PR receipt type. Release tooling changes such as #10360 are not publication
requests, and this route does not infer release intent from PR titles or labels.

`--receipt-context release` refuses the reduced route in the producer.
`BEHAVIOUR_RECEIPT_CONTEXT=release` refuses it in the gate and requires the full
route even for a documentation-only release diff. No candidate receipts should be
posted before independent review and author promotion.


### Queue-time main freshness

The trusted `Merge Queue Gate` rechecks this route when it admits a native
`merge_group`. An earlier green exact-head PR check cannot authorize an equivalence
baseline after main moves. The queue gate reads the actual regular/LFS receipt
bytes from the exact source head, verifies their SHA256 and committed regular-file
mode and size, and extracts the baseline from that payload. It independently
reads live GitHub main and requires the payload baseline to equal both that main
and `merge_group.base_sha`. A body-only baseline assertion cannot substitute for
this proof. The queue then repeats complete admission of the currently selected
header and payload using the same base-owned helper as the PR gate: trusted
schema/scope, execution source and ancestry, authenticated scheduler/hardware,
per-row controller/equality/path bindings, and current independent review. This
prevents an unchanged head containing two receipts from reusing A's old green
check after the body selects invalid B. The audit binds complete admission to the
current header digest as well as payload path/digest. Both PR and queue paths
re-read main after validation and refuse a main move during admission.
Missing, malformed, stale, unavailable, semantically rejected or mixed-route evidence refuses
admission and appears in the durable JSON audit and job summary. Pure evaluation
also refuses an equivalence route without a complete, bound admission proof.

The queue reader uses trusted base-owned stdlib code and never imports or executes
PR code. When CI lacks the immutable head object it performs an authenticated,
read-only Git fetch by exact SHA using process-local `GH_TOKEN` credentials,
then reads that commit's regular blob or exact LFS pointer. It never checks out the
PR. Missing live-main or tooling Git objects are fetched by validated exact SHA;
trusted shallow history is deepened without a checkout so source-ancestry proofs
remain complete. Full selected-receipt validation queries the authenticated
scheduler again; unavailable scheduler access refuses admission, with no hardware
waiver. It does not rerun simulations. The workflow installs pinned PyYAML6.0.3,
jsonschema4.26.0, loguru0.7.3, NumPy2.4.6, NumBa0.67.0, Shapely2.1.2, Matplotlib3.11.1 and python-motion-planning2.0.1.
The last two satisfy the existing publication-checksum validator import graph;
complete isolated synthetic admission validates this dependency closure. These
imports never execute a simulation. Full receipts and ordinary PRs retain their existing
queue behavior. Explicit malformed/no-newline equivalence comment markers are
refused at both PR and queue dispatch before any neutral-path shortcut.

The guard applies to live Merge Queue Gate audits for source PRs and native
merge groups. Source-PR audits remain advisory; native-group evaluation fails
closed when invoked. This candidate does not assert that the repository currently
requires `Merge Queue Gate / merge-queue-gate`: the present enforcement readback
did not establish that required-check activation. Author-controlled promotion
must establish activation before claiming backend enforcement; other direct merge
dispatchers are outside this candidate's assessment. The closed
[#6274](https://github.com/ll7/robot_sf_ll7/issues/6274) supplies historical contract
context, rather than an open follow-up. This change does not activate branch
protection, change repository settings, or grant authorization to merge.
