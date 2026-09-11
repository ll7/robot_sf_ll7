# Post-access restoration and local-analysis runbook

This runbook helps you continue Robot SF locally after a university host, scheduler, private
mount, or pre-existing worktree is no longer available. It restores only inputs that have a
public-safe pointer and a checkable identity, and it labels missing capabilities explicitly. It
does not recreate inaccessible infrastructure or turn a local smoke into a benchmark result.

The runbook is a smoke/diagnostic workflow. Run commands from the repository root. Replace every
`<PLACEHOLDER>` before executing a command; an unresolved placeholder is not a valid input. Keep
generated files under `output/`, which is worktree-local scratch and is not durable evidence.

## The bounded path

Use this order, stopping at the first missing or contradictory contract:

1. Clone a clean checkout and record its commit.
2. Select one dependency profile and check the host.
3. Run the no-private-input synthetic path, or identify a public-safe real-artifact pointer.
4. Restore real inputs into a new task-owned `output/` root and verify checksums, manifests, and
   lineage before analysis.
5. Check model and normalizer identity separately when a learned policy is part of the input.
6. Regenerate local summaries or figures from validated inputs.
7. Report exact source, inputs, commands, statuses, and unavailable capabilities.
8. Clean only the task-owned restore root after preservation decisions are recorded.

The [development setup guide](dev_guide.md#setup), [runtime requirements](dev_runtime_requirements.md),
and [doctor troubleshooting contract](troubleshooting/doctor.md#statuses-and-exit-code) remain the
source of truth for their subsystems. This page joins those owners into a post-access sequence; it
does not replace them.

## 1. Start from a clean checkout

Do not copy an old `.venv`, `output/` tree, private mount, or unresolved symlink from the lost
machine. Clone the repository, select the release tag or commit named by the durable pointer, and
record the exact source identity:

```bash
git clone https://github.com/ll7/robot_sf_ll7.git robot_sf_ll7-local
cd robot_sf_ll7-local
git fetch --tags origin
git checkout --detach "<release-tag-or-commit>"
git rev-parse HEAD
git status --short --branch
```

If the pointer does not name a public tag or commit, record `not_available` and stop. Do not use
the current branch as a substitute for a historical source identity. The repository quickstart is
also linked from [README.md#quickstart](../README.md#quickstart).

## 2. Choose the smallest local profile

The profiles separate package installation from optional machine access. The names below are
runbook labels, not new `uv` extras; the exact extra names come from `pyproject.toml` and
[`docs/dev_runtime_requirements.md`](dev_runtime_requirements.md#optional-python-extras-and-external-artifacts).

| Profile | Install or sync | Use | If unavailable |
| --- | --- | --- | --- |
| `minimal` | `uv sync` | Core imports, repository checks, and the CPU-only demo. | Required setup failure is `failed`; stop before interpreting later output. |
| `analysis` | `uv sync --extra benchmark --extra analytics` | Benchmark-shaped JSONL summaries and local analysis tools. | Record `missing_optional` or `not_available`; do not omit missing rows or substitute a different metric. |
| `visualization` | `uv sync --extra viz` | Matplotlib/Pygame figure and playback paths. | Record `missing_optional` or `not_available`; do not write an empty figure as success. |
| `model-loading` | `uv sync --extra training` | Optional checkpoint and normalizer loading. Add `--extra gpu` only for a declared GPU path. | Record `model_unavailable` or `not_available`; do not substitute another policy or claim planner quality. |

For a broad local setup, `uv sync --all-extras` is valid. It does not install the CARLA group:

```bash
uv sync --all-extras
```

Only opt into the external CARLA client when the pointer and task require it:

```bash
uv sync --all-extras --group carla
scripts/dev/check_carla_runtime.sh --smoke
```

The CARLA check is optional infrastructure proof. Without a compatible server, Docker runtime,
and client, its dependent replay is `unavailable`; a CPU demo is not a CARLA fallback.

After syncing, run the repository-owned checks. The second doctor command avoids environment and
quickstart execution when those capabilities are intentionally absent:

```bash
scripts/dev/check_runtime_requirements.sh
uv run robot-sf doctor --format json --skip-env-smoke --skip-quickstart-smoke
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; print('Import successful')"
```

The doctor contract uses `ok`, `skipped`, `missing_optional`, and `failed`. A skipped check is not
an `ok` result. Required failures stop the path; optional warnings identify the capability that
cannot be used. Use the exact check `name` from the JSON report rather than scraping prose.

For a source-bound status example, run the existing optional-capability owner:

```bash
uv run python examples/advanced/36_optional_capability_handling.py --format json
```

Its stable diagnostic reason codes include `core_available`, `extra_missing`, `model_unavailable`,
`model_unknown`, `dataset_unavailable`, and `runtime_unsupported`. The example reports
`available: false` for unavailable probes and fails closed on unknown codes. These codes describe
local capability probes; they do not certify a benchmark row.

## 3. Synthetic default path: no private artifact

This path proves only that the fresh checkout can produce and inspect a local trace. It needs no
cluster, GPU, CARLA server, private dataset, checkpoint, or external artifact. The output remains
disposable:

```bash
uv run robot-sf demo --output-root output/offline_runbook/demo --seed 270

DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
uv run python scripts/generate_figures.py \
  --episodes output/offline_runbook/demo/episode.jsonl \
  --out-dir output/offline_runbook/demo/figures \
  --no-pareto
```

The demo writes a recorded episode, summary, metrics, viewer, and thumbnail. The figure command
regenerates local plots from that trace when the visualization profile is installed. Set
`ROBOT_SF_VALIDATE_VISUALS=1` only when the visualization-manifest validation contract and its
inputs are present; otherwise report that validation as `not_available`. The headless variables
are defined by [the runtime rendering section](dev_runtime_requirements.md#headless-rendering),
and the visualization owner is [benchmark visual artifacts](benchmark_visuals.md#validation).

This path is a local smoke only. Do not report its collision count, distance, plot, runtime, or
viewer as a planner comparison, benchmark result, or paper-facing evidence. If the demo cannot
run because a required dependency is missing, fix the `doctor` result or report `failed`; do not
quietly switch to a different simulator backend.

For a benchmark-shaped local input, use the exact tiny matrix and `robot_sf_bench` sequence in
[`docs/ENVIRONMENT.md#tiny-batch-smoke`](ENVIRONMENT.md#tiny-batch-smoke). The current repository
paths are [`configs/baselines/example_matrix.yaml`](../configs/baselines/example_matrix.yaml) and
[`robot_sf/benchmark/schemas/episode.schema.v1.json`](../robot_sf/benchmark/schemas/episode.schema.v1.json);
keep those source identities visible in any local diagnostic. That sequence is still only a smoke
unless a separate benchmark contract, source commit, seed set, artifact, and eligibility decision
are recorded.

For a retained failed-campaign dossier, the current diagnostic owner is also safe to run locally:

```bash
uv run python examples/advanced/39_failed_campaign_diagnosis.py --json
```

It diagnoses synthetic retained files and does not submit a job, retry a campaign, or create
scientific evidence. Use its owner-specific findings; do not copy its failure taxonomy into a new
global status vocabulary.

## 4. Find and restore a real artifact safely

Use a durable pointer, not a guessed mount path. A usable pointer names, at minimum:

- a public release URL, DOI, tracked evidence path, or external artifact URI;
- artifact ID/version, source commit, configuration identity, and seed or episode scope;
- license/access terms, expected SHA-256, and the documented hydration command;
- the failure domain and the condition that would unblock a missing or restricted input.

Committed pointers can be inventoried without searching private files:

```bash
rg -n 'artifact_uri|github_release|sha256|doi' docs/context/evidence model/registry.yaml
```

If no public-safe pointer, checksum, license note, or hydration command is available, record
`not_available` with the missing field and stop. Do not paste credentials, signed URLs, private
hostnames, mount paths, or access tokens into this repository or an issue report. Do not infer an
artifact from a filename or copy a surviving `output/` directory and call it durable.

Create a fresh restore root owned by this task. Keep the manifest outside the result tree:

```bash
ARTIFACT_ID="<artifact-id>"
RESTORE_ROOT="output/offline_runbook/restore/${ARTIFACT_ID}"
mkdir -p "$RESTORE_ROOT"
printf '%s\n' 'task-owned offline restore root' > "$RESTORE_ROOT/.offline-runbook-owner"
```

Run the exact hydration command declared by the pointer with its destination set to
`$RESTORE_ROOT`. If the pointer does not define a safe destination or the source cannot be
hydrated, leave the root untouched or preserve it as an explicitly incomplete copy and report
`not_available`; do not invent a download or retry policy.

For a release bundle that already has the repository release manifest, use the release checksum
owner without downloading during verification:

```bash
sha256sum -c "<CHECKSUM_FILE>"
uv run python scripts/repro/verify_release_checksums.py \
  --tag "<release-tag>" \
  --bundle-path "<downloaded-bundle.tar.gz>" \
  --output-dir output/offline_runbook/checksum-verification \
  --no-download
```

For a restored result tree, build or locate its chunk manifest and verify it before opening rows:

```bash
uv run python scripts/tools/chunk_manifest.py manifest \
  --root "$RESTORE_ROOT" \
  --output output/offline_runbook/restore-manifest.json \
  --artifact-id "$ARTIFACT_ID" \
  --artifact-version "<artifact-version>" \
  --retention-role long-lived \
  --json

uv run python scripts/tools/chunk_manifest.py verify \
  --root "$RESTORE_ROOT" \
  --manifest output/offline_runbook/restore-manifest.json \
  --json
```

The manifest verifier rejects mutation, truncation, missing or unexpected members, path escape,
symlink/hardlink/special-file inputs, collisions, and partial manifests. A nonzero result is
`failed`; it is not permission to analyze the surviving subset.

Capture the environment packet only from the exact checkout whose identity you intend to report:

```bash
uv run python scripts/repro/capture_release_environment.py \
  --release-tag "<release-tag>" \
  --output output/offline_runbook/environment.json \
  --require-clean
```

`--require-clean` intentionally fails on a dirty checkout. The packet records dependency and
verification-environment identity; without a supplied historical runtime record,
`campaign_runtime_records.status` is `missing`. That missing record is not evidence that a
campaign failed or succeeded. Keep campaign-runtime parity as a separate, explicitly available
input.

If the lost host and the local host each have a sanitized `cross_host_environment.v1` manifest,
compare them only under the declared workload requirements:

```bash
uv run python scripts/validation/compare_execution_environments.py \
  --host-a "<SANITIZED_HOST_A_MANIFEST>" \
  --host-b "<SANITIZED_HOST_B_MANIFEST>" \
  --requirements "<WORKLOAD_REQUIREMENTS_JSON>" \
  --format json
```

The comparator classifies missing or redacted fields as unavailable or non-comparable and blocks
an equivalence claim on undeclared material differences. Environment equivalence does not imply
output equivalence, campaign success, or planner quality. If either manifest or the requirements
contract is unavailable, record `not_available` and do not reconstruct it from `uv.lock`, current
package versions, or local output.

If the restored data has sanitized lineage, validate and query it before analysis:

```bash
uv run python scripts/tools/lineage_index.py \
  --input "<SANITIZED_LINEAGE_JSON>" \
  --check \
  --format json

uv run python scripts/tools/lineage_index.py query \
  --input "<SANITIZED_LINEAGE_JSON>" \
  --campaign "<CAMPAIGN_ID>" \
  --format json
```

The lineage input must not expose private locators. A missing campaign match or a schema/check
failure is `not_available` or `failed`, respectively; do not construct lineage from filenames.

## 5. Verify checkpoints and normalizers independently

A model registry entry is a pointer and checksum contract, not a guarantee that the artifact is
present on this machine. Inspect the registry and verify pinned model artifacts:

```bash
uv run robot-sf models list
uv run robot-sf models verify --format json
uv run python scripts/validation/check_local_model_artifacts.py configs/baselines --json --fail-on-blocked
```

When the selected checkpoint is a model-registry entry supported by the legacy PPO validator, an
opt-in smoke can load it, predict one action, and take one current environment step without
downloading a replacement:

```bash
uv run python scripts/validation/check_legacy_ppo_snapshot_parity.py \
  --registry-path model/registry.yaml \
  --smoke-model-id "<MODEL_ID>" \
  --json
```

Run this only after the pointer has hydrated the checkpoint locally. A missing local file or
unsupported model is `not_available`/`failed`; do not add `--allow-download` unless the public
pointer explicitly authorizes that hydration. This smoke checks load and environment-step
plumbing only, not policy quality, historical parity, or benchmark eligibility.

For a learned policy, check all of these against its manifest before loading it:

- checkpoint `artifact_role`, durable URI, SHA-256, source/training commit, and config identity;
- observation and action schemas, including deployment-visible fields and bounds;
- `normalizer_uri`, normalizer checksum, fit split, and the explicit `not_required` case;
- license/access note, split contract, benchmark eligibility, and `fail_closed_behavior`.

The [model registry](../model/registry.md#using-models-from-the-registry) owns registry metadata,
and the [artifact evidence vocabulary](context/artifact_evidence_vocabulary.md) owns the learned-
policy manifest fields. A successful checkpoint or normalizer load proves only that this local
plumbing accepted the supplied files. It does not establish planner quality, reproducibility of
the historical run, benchmark eligibility, or a paper-facing result. If either artifact is
missing, mismatched, or cannot be hydrated, record `model_unavailable`, `model_unknown`,
`not_available`, or `failed` as appropriate and stop the dependent learned-policy path. Do not
fall back to another checkpoint, normalizer, planner, or CPU mode while keeping the original label.

## 6. Run local analysis and regenerate figures

Analyze only after source, artifact identity, schema, checksum, and lineage checks pass. Keep
generated analysis under a task-owned output root:

```bash
uv run robot_sf_bench summary \
  --in "<EPISODES_JSONL>" \
  --out-dir output/offline_runbook/summary

DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
uv run python scripts/generate_figures.py \
  --episodes "<EPISODES_JSONL>" \
  --out-dir output/offline_runbook/figures \
  --no-pareto
```

If the input is a report or figure source with a semantic catalog, validate that catalog too:

```bash
uv run python scripts/validation/validate_artifact_catalog.py "<CATALOG_YAML>"
```

The figure metadata should retain the source commit, input path or durable pointer, command,
scope, and claim boundary. A generated file under `output/` is local scratch until a separate
promotion workflow records a durable location and checksum. Missing plotting dependencies,
missing source rows, schema mismatch, or an absent catalog are `not_available`/`failed`—never an
empty or imputed figure.

If the restored input is a tracker-backed imitation-learning run, use the existing report owner
only when the exact tracker run and dataset are available:

```bash
uv run python scripts/research/generate_report.py \
  --tracker-run "<TRACKER_RUN_ID>" \
  --experiment-name "<EXPERIMENT_NAME>" \
  --output output/offline_runbook/research-report
```

The [research reporting guide](research_reporting.md) defines that owner. A missing tracker run,
dataset, or report input is `not_available`; do not create an empty report and call it analysis.

## 7. Produce an issue-ready diagnostic handoff

Use the following fields in an issue or local handoff. Keep placeholders and private locators out
of the final report:

```text
Status: pass | diagnostic | blocked | not_available | failed
Source commit: <40-character commit>
Profile and doctor status: <profile>; <doctor JSON status/check names>
Artifact pointer/version: <public-safe URI or tracked pointer>; checksum: <sha256>
Restore manifest/tree identity: <manifest path and digest>; rows/scope: <count and identity>
Environment packet: <output path>; campaign runtime record: complete | missing | not_available
Checkpoint/normalizer: verified | not_available | failed | not_required, with separate URIs
Analysis command: <exact command>
Figure/report outputs: <local output paths>; durable pointer: <URI or not promoted>
Unavailable capabilities and unblock condition: <capability/status/reason>
Claim boundary: local restore/analysis plumbing only; no benchmark or paper-facing claim
Cleanup: retained | task-owned scratch removed | blocked, with reason
```

When a structured diagnostic card is useful, the repository owner is
`scripts/reporting/generate_result_card.py`. It requires an actual summary, at least one supplied
metric, exact command provenance, and caveats; use `diagnostic`, not `promote`, for this runbook:

```bash
uv run python scripts/reporting/generate_result_card.py \
  "<SUMMARY_JSON>" \
  --output-dir output/offline_runbook/result-card \
  --evidence-tier smoke \
  --decision diagnostic \
  --comparator "none: local restore/analysis smoke only" \
  --claim-boundary "Local restore and analysis plumbing only; no planner-quality, benchmark, or paper-facing claim." \
  --metric "<METRIC_NAME>=<FINITE_VALUE>" \
  --command "<EXACT_ANALYSIS_COMMAND>" \
  --artifact "<DURABLE_POINTER>" \
  --caveat "Missing cluster, private, GPU, or CARLA inputs remain unavailable." \
  --non-transfer-note "This diagnostic card is not benchmark or research evidence." \
  --allow-local-output-with-durable-pointer
```

The vocabulary distinguishes these categories:

| Category | Meaning here | Can it support a benchmark or paper claim? |
| --- | --- | --- |
| Local output | Disposable files under `output/` from a smoke or analysis run. | No, not by itself. |
| Durable artifact | Verified copy with pointer, checksum, hydration path, scope, and retention decision. | It supports only the declared artifact contract. |
| Compact evidence | Small tracked pointer, manifest, or evidence copy with provenance and caveats. | Only within its recorded scope. |
| Release asset | Immutable release/DOI/W&B bundle with its published checksum and version. | Only the declared release contract. |
| Dissertation-facing evidence | A frozen benchmark or evidence package with exact claims, inputs, provenance, and checksums. | Only after the paper-facing contract is independently satisfied. |

See [artifact evidence vocabulary](context/artifact_evidence_vocabulary.md) and the
[artifact retention guide](context/artifact_retention_and_cleanup.md#4-checked-workflows) before
using the words “preserved”, “benchmark”, “release”, or “dissertation evidence”.

## 8. Unavailable-state matrix

| Missing capability or input | Required status | Safe next action |
| --- | --- | --- |
| University scheduler or SLURM allocation | `not_available` or `blocked` | Preserve retrieved outputs and report the missing allocation; do not replace a historical campaign with a CPU smoke. |
| GPU runtime or training extra | `missing_optional`, `not_available`, or `failed` | Install the declared extra only if allowed; otherwise stop model-dependent work. |
| CARLA server/client/Docker runtime | `unavailable` or `failed` | Follow `scripts/dev/check_carla_runtime.sh` and its owner docs; no live replay or parity claim without the exact server boundary. |
| Private dataset or mount | `not_available` | Record the public-safe pointer and unblock condition; never include the private locator. |
| Checkpoint or normalizer | `model_unavailable`, `model_unknown`, `not_available`, or `failed` | Stop the dependent learned-policy path; do not choose an alternate artifact. |
| Missing rows, checksum, schema, or lineage | `failed` | Keep the source incomplete and report the exact validator output; do not analyze a surviving subset as the original run. |

`unavailable`, `not_available`, `blocked`, `missing_optional`, and `failed` are non-success states.
They are useful evidence about what could not be verified, not permissions to weaken the contract.
If a tool emits an identifier not covered by its current owner documentation, record a contract
gap and stop rather than inventing a new interpretation.

## 9. Clean only task-owned scratch

Before cleanup, apply the [retention and cleanup owner](context/artifact_retention_and_cleanup.md):
verify the restore, check consumers and leases, record any durable promotion, and preserve the
pointer/manifest. Never delete the original durable source as part of local cleanup.

For the root created in section 4, inspect the path and require its ownership marker before
removing it. The guard is intentionally narrow:

```bash
printf 'Review before removal: %s\n' "$RESTORE_ROOT"
case "$RESTORE_ROOT" in
  output/offline_runbook/*) ;;
  *) printf 'Refusing non-task-owned restore path: %s\n' "$RESTORE_ROOT" >&2; exit 2 ;;
esac
test -f "$RESTORE_ROOT/.offline-runbook-owner"
rm -rf -- "$RESTORE_ROOT"
```

If the marker is absent, the path escapes `output/offline_runbook/`, a durable consumer still
needs it, or a preservation gate is unresolved, retain it and report `blocked`. The worktree
reaper is a separate read-only lifecycle check; follow its documented command rather than
deleting a worktree from this runbook.

## Canonical owners and companion example lanes

This runbook routes to existing owners:

- Setup and profiles: [`docs/dev_guide.md#setup`](dev_guide.md#setup),
  [`docs/dev_runtime_requirements.md`](dev_runtime_requirements.md), and `pyproject.toml`.
- Host/status checks: [`scripts/dev/check_runtime_requirements.sh`](../scripts/dev/check_runtime_requirements.sh),
  `robot-sf doctor`, and [`docs/troubleshooting/doctor.md`](troubleshooting/doctor.md).
- Preservation and manifests: [`scripts/tools/chunk_manifest.py`](../scripts/tools/chunk_manifest.py),
  [`scripts/tools/lineage_index.py`](../scripts/tools/lineage_index.py), and the retention guide.
- Models and evidence vocabulary: [`model/registry.md`](../model/registry.md) and
  [`docs/context/artifact_evidence_vocabulary.md`](context/artifact_evidence_vocabulary.md).
- Analysis and figures: [`scripts/generate_figures.py`](../scripts/generate_figures.py),
  [`scripts/validation/validate_artifact_catalog.py`](../scripts/validation/validate_artifact_catalog.py),
  and the research-report owner.

Issue [#8900](https://github.com/ll7/robot_sf_ll7/issues/8900) owns the companion synthetic
campaign-capsule restore/summary example. Issue
[#8909](https://github.com/ll7/robot_sf_ll7/issues/8909) owns the companion task-owned output-tree
preservation manifest example. Those owners may be unavailable in an older checkout; this
runbook does not duplicate their implementations or depend on their future paths. If a companion
owner is not present at the selected source commit, use the current owner commands above and
record the companion lane as `not_available`.

This page does not run a real campaign, restore private data, load a historical checkpoint, or
make a scientific comparison. It documents how to stop safely and leave an auditable local
diagnostic when those inputs cannot be verified.
