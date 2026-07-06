# Issue #2918 Closure Audit

Date: 2026-07-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/2918>
Context note: [`issue_2918_pedestrian_prior_extraction_preflight.md`](../issue_2918_pedestrian_prior_extraction_preflight.md)
External-data blockers: <https://github.com/ll7/robot_sf_ll7/issues/3065>,
<https://github.com/ll7/robot_sf_ll7/issues/2657>, <https://github.com/ll7/robot_sf_ll7/issues/1498>

## Purpose

Closure audit for #2918 (`data: pilot external pedestrian-prior extraction after dataset staging`).
This note maps each acceptance criterion to merged-PR evidence and to a locally reproduced
validation run, then records the closure decision. It asserts only extraction-logic and
staging-contract readiness; it ingests no external data, stores no raw trajectories, and makes no
calibrated- or representative-prior claim.

## Authoritative scope

The issue body's `agent-exec-spec:v1` block (appended 2026-06-20) is the authoritative scoping and
overrides the older issue-body wording. It defines the **agent-executable slice** as the
prior-extraction pipeline + manifest, fixture-tested, gated on `manage_external_data.py`
availability and fail-closed otherwise, with no representativeness claim. It marks the
dataset-backed run as `Blocked-until` license-compatible staged data via #2657 / #1498 / #2414. The
issue-audit reframe (2026-06-22) narrows the data path to the opt-in BYO-dataset copy under #3065,
and the 2026-07-05 maintainer state comment frames the identical residual: "dataset-backed prior
smoke still blocked on staged license-compatible external data (#3065/#2657); no
calibrated/representative claim admitted."

Per the repository COMPLETE-FIRST rule, an issue whose only remaining work needs external inputs
that do not exist (here, a license-gated / manually-acquired external dataset) counts as complete
for the agent-executable contract.

## Acceptance criteria → evidence

| Acceptance criterion (agent-exec-spec:v1) | Status | Evidence |
| --- | --- | --- |
| Extraction pipeline + manifest implemented + fixture-tested; no raw trajectories in git | Met | PR #4566 (`98b7e86d9`) — `robot_sf/benchmark/pedestrian_prior_extraction.py`, CLI `scripts/tools/extract_pedestrian_prior.py`, fixture `tests/benchmark/fixtures/issue_2918_pedestrian_prior_fixture.yaml`, tests `tests/benchmark/test_issue_2918_pedestrian_prior_extraction.py`. Manifest contract from PR #3754 (`de64c273c`) — `robot_sf/benchmark/pedestrian_prior_extraction_manifest.py`, schema `.../schemas/pedestrian_prior_extraction_manifest.v1.json`, checker `scripts/tools/check_pedestrian_prior_extraction_manifest.py`. Report emits summaries only; provenance stamps `raw_trajectory_storage: not_stored_in_git`. |
| Runs only on staged license-compatible data; missing data → fail-closed | Met | Manifest checker returns `contract_status: blocked`, `dataset_backed_prior_claim_allowed: false` on the placeholder example; the extractor's `dataset-backed` `value_status` is only a stamp and cannot itself admit a claim (`test_dataset_backed_status_is_only_a_stamp_not_manifest_admission`). All external assets (`sdd`, `eth-ucy`, `socnavbench-*`) report `status: missing`/`incomplete` via `manage_external_data.py list`; reproduced below. |
| No representativeness claim beyond the staged source | Met | `evidence_boundary: prior_extraction_plan_only_no_calibrated_prior_claim` stamped on every report and manifest report; enforced by manifest boundary/separation checks (`test_proxy_only_with_dataset_source_is_rejected_as_conflation`, `test_byo_only_source_cannot_back_a_dataset_backed_claim`). |

| Original-body Acceptance / Stop rule item | Status | Evidence |
| --- | --- | --- |
| Deterministic importer has checksum and provenance manifest | Met | PR #3754 manifest schema requires per-source provenance (`checksum` / `staging_manifest`) before a `dataset-backed` claim; fixture-validated. |
| One prior smoke scenario generated from dataset-backed parameters | Partial — external-data-gated | The **fixture** smoke runs (proxy-placeholder, reproduced below). The **dataset-backed** smoke requires a license-gated, manually-acquired external dataset that the project does not hold; not agent-executable. Tracked by #3065 / #2657 / #1498. |
| Report states limitations and non-claims | Met | Preflight context note + `evidence_boundary` stamp on every emitted report/manifest; this closure audit records the non-claims explicitly. |
| If evidence cannot be produced, close/update as blocked rather than upgrading the claim | Met | Manifest stays `blocked-external-input`; no calibrated/representative claim admitted anywhere. |

### Contributing merged PRs

| PR | Commit | Merged | Contribution |
| --- | --- | --- | --- |
| #3754 | `de64c273c` | 2026-06-27 | External pedestrian-prior extraction staging/preflight contract + schema + fail-closed checker + preflight context note |
| #4566 | `98b7e86d9` | 2026-07-05 | Local fixture prior-extraction pipeline (extractor module + CLI + fixture + tests), stamped plan-only / no-calibrated-prior claim |

## Reproduced validation (2026-07-06, `origin/main` @ `405eb5b5a`)

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python -m pytest \
  tests/benchmark/test_issue_2918_pedestrian_prior_extraction.py \
  tests/benchmark/test_issue_2918_pedestrian_prior_extraction_manifest.py -q
# 23 passed

# Fixture smoke: extracts all 5 required prior parameters, stamped proxy-placeholder,
# no raw trajectories, evidence_boundary = prior_extraction_plan_only_no_calibrated_prior_claim
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/tools/extract_pedestrian_prior.py \
  --input tests/benchmark/fixtures/issue_2918_pedestrian_prior_fixture.yaml
# exit 0

# Fail-closed contract on the placeholder manifest
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/tools/check_pedestrian_prior_extraction_manifest.py \
  --manifest configs/research/pedestrian_prior_extraction_manifest_issue_2918_example.yaml
# contract_status = blocked; dataset_backed_prior_claim_allowed = false;
# extraction_status = blocked-external-input

# External-data availability (dataset-backed blocker)
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/tools/manage_external_data.py list
# sdd: status missing; eth-ucy: status missing; socnavbench-*: status incomplete
```

The fixture smoke produces bounded summaries (min/max/mean/count) for all five required prior
parameters (walking speed, crossing angle, density, interaction distance, stop/yield timing) with
`value_status: proxy-placeholder`, and the manifest checker refuses any dataset-backed claim while
external data is missing.

## Closure decision

**Close #2918.** Every agent-executable acceptance criterion from the authoritative
`agent-exec-spec:v1` block is met and reproducibly validated on `origin/main`: the extraction
pipeline + manifest are implemented and fixture-tested with no raw trajectories in git, the manifest
fails closed when data is missing, and no representativeness/calibration claim is admitted. The
single residual — a **dataset-backed** prior smoke from real staged trajectories — is not
agent-executable: it requires a license-gated, manually-acquired external dataset (ETH/UCY, SDD, or
SocNavBench-derived) that the project does not hold and cannot redistribute. That residual is owned
by the external-data staging blockers (#3065 opt-in BYO-dataset, #2657 SDD staging, #1498 SocNavBench
ETH assets), so closing #2918 does not lose the tracking trail.

### Residual (tracked by #3065 / #2657 / #1498)

- Stage a license-compatible external pedestrian-trajectory dataset through the opt-in BYO-dataset
  path, then run the extractor with `--value-status dataset-backed` behind a `ready` manifest to
  produce the dataset-backed prior smoke and feed the authored-vs-dataset comparison (#2919 / #3161).
  No further agent-executable step is available until that data is staged and license-reviewed.
