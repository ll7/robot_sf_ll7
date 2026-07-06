# Issue #1456 Closure Audit

Date: 2026-07-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/1456>
Related: <https://github.com/ll7/robot_sf_ll7/issues/562> ·
<https://github.com/ll7/robot_sf_ll7/issues/1584> ·
<https://github.com/ll7/robot_sf_ll7/issues/1454> ·
<https://github.com/ll7/robot_sf_ll7/issues/1353> ·
<https://github.com/ll7/robot_sf_ll7/issues/1873>
Prior status note: [`issue_2397_socnavbench_control_status_2026-06-06.md`](../issue_2397_socnavbench_control_status_2026-06-06.md)
Downstream row policy: [`issue_1584_socnav_unavailable_row_policy.md`](../issue_1584_socnav_unavailable_row_policy.md)

## Purpose

Closure audit for #1456 (`data-infra: restore SocNavBench control-pipeline assets`), triggered by
the 2026-07-05 ready-queue reconcile flag ("only terminal rows but issue open"). This note maps each
acceptance criterion — from both the original issue body and the appended `agent-exec-spec:v1`
block — to merged-PR evidence and to a locally reproduced fail-closed validation, then records the
closure decision. It asserts only local asset-readiness/tooling status; it does not stage licensed
external data, run any benchmark, or promote any benchmark/paper claim.

## Authoritative scope

The issue is an **asset-restoration tracker only** (narrowed by the issue-audit decisions of
2026-05-22 / 2026-05-25 and the 2026-05-31 tooling split into #1873). Two distinct layers of
acceptance exist:

1. **Local tooling / inventory layer** (agent-executable now): explain/check/stage/provenance
   behavior and fail-closed availability reporting that never counts placeholders as restored.
2. **Asset-staging layer** (external gate): the actual licensed SocNavBench control-pipeline data
   assets present on the checkout, followed by a clean re-entry probe.

Per the repository `COMPLETE-FIRST` rule, a compute/SLURM-only residual would count as complete.
This issue's residual is **not** compute-gated — it requires maintainer-staged **licensed external
data** ("downloadable but not republishable", per the 2026-06-22 / later triage notes), so the
issue correctly stays open as `state:blocked-external-input`. The local tooling layer, however, is
fully met.

## Acceptance criteria → evidence

### Original issue-body acceptance criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Required SocNavBench control-pipeline assets are locally staged/hydrated | **Blocked** (external gate) | Reproduced below: `wayptnav_data`, `sd3dis/stanford_building_parser_dataset`, `.../traversibles` all missing. Requires maintainer-staged licensed data. |
| `prepare_socnav_assets.py` completes without `MISSING_REQUIRED_ASSETS` | **Blocked** (external gate) | Reproduced below: exit `2`, `status=MISSING_REQUIRED_ASSETS`. Depends on the staging gate above. |
| Focused re-entry probe completes without `fallback`/`degraded`/`not-available` | **Blocked** (external gate) | Probe path documented in [`issue_562_socnav_bench_reentry.md`](../issue_562_socnav_bench_reentry.md); cannot pass until assets are staged. |
| A durable context note records asset source, validation command, result, limitations | **Met** | [`issue_2397_socnavbench_control_status_2026-06-06.md`](../issue_2397_socnavbench_control_status_2026-06-06.md) (PR #2400, `1514bc79b`) + this closure-audit note. |
| Downstream benchmark issues mark SocNavBench-family rows `unavailable/excluded`, not success | **Met** | Row policy in [`issue_1584_socnav_unavailable_row_policy.md`](../issue_1584_socnav_unavailable_row_policy.md) (PR #1596); consumed by #1353/#1454 alignment (PR #1486). |

### Appended `agent-exec-spec:v1` acceptance criteria (agent-executable slice)

| Criterion | Status | Evidence |
| --- | --- | --- |
| Missing vs present assets inventoried with provenance + restore path documented | **Met** | `scripts/tools/manage_external_data.py` explain/check/stage/provenance for `socnavbench-control` + `socnavbench-s3dis-eth` (PR #1924, closing #1873); `docs/socnav_assets_setup.md`; #2397 inventory note. |
| Availability reporting is fail-closed; placeholders never counted as restored | **Met** | PR #3755 (`87086a515`) fail-closed readiness vs placeholder shells; PR #4526 (`64cae2774`) tightened the `socnavbench-control` contract to also require the S3DIS/SBPD groups (wayptnav-only no longer validates). 43 focused tests pass (reproduced below). |

### Contributing merged PRs

| PR | Commit | Contribution |
| --- | --- | --- |
| #1924 | (closes #1873) | External-data assistant `manage_external_data.py`: explain/check/stage/provenance for the SocNavBench asset families |
| #2400 | `1514bc79b` | Durable #2397 status note + tracked evidence sidecar |
| #1596 | (closes #1584) | `unavailable/excluded` downstream row policy |
| #1486 | — | Aligned #1353 SocNav row contract to the policy |
| #3755 | `87086a515` | Fail-closed asset readiness vs placeholder shells |
| #4526 | `64cae2774` | Tightened `socnavbench-control` contract to require S3DIS/SBPD groups (fail-closed, tests inverted) |

## Reproduced validation (2026-07-06, `origin/main` @ `405eb5b5a`)

```bash
uv run python scripts/tools/manage_external_data.py --json check socnavbench-control
# exit 2; status=incomplete, ok=false (wayptnav_data + S3DIS/SBPD groups missing)

uv run python scripts/tools/manage_external_data.py --json check socnavbench-s3dis-eth
# exit 2; status=incomplete, ok=false (ETH mesh dir + ETH traversible pickle missing)

uv run python scripts/tools/prepare_socnav_assets.py --report-json output/tmp/issue1456_socnav_report.json
# exit 2; status=MISSING_REQUIRED_ASSETS
#   - wayptnav_data: missing (required)
#   - sbpd_dataset: missing (required)      -> sd3dis/stanford_building_parser_dataset
#   - sbpd_traversibles: missing (required) -> sd3dis/stanford_building_parser_dataset/traversibles

uv run python -m pytest tests/ -k "socnav and (map or asset)" -q
# 43 passed, 11705 deselected
```

The vendored SocNavBench code tree exists under `third_party/socnavbench`, but the required external
**data** assets are absent, and every readiness path fails closed (exit `2` / `MISSING_REQUIRED_ASSETS`)
rather than reporting a placeholder shell as available. This matches the #2397 snapshot and confirms
the PR #3755 / #4526 fail-closed behavior still holds after those merges.

## Closure decision

**Keep #1456 open, `state:blocked-external-input`.** Every agent-executable acceptance criterion —
the tooling/inventory layer and the downstream-row-policy layer — is met and reproducibly validated
on `origin/main`. The three core asset criteria (assets staged, `prepare_socnav_assets.py` clean,
re-entry probe clean) are genuinely blocked on **maintainer-staged licensed external data**, which is
not agent-executable and not merely compute-gated, so `COMPLETE-FIRST` does not convert the issue to
complete. The fail-closed tooling already tracks the exact missing paths, so keeping the issue open
loses no tracking trail.

### Residual (the single remaining gate)

- Stage the licensed SocNavBench control-pipeline assets (`wayptnav_data`, S3DIS/SBPD meshes +
  traversibles, ETH mesh/traversible) under `third_party/socnavbench` via the manual path in
  [`docs/socnav_assets_setup.md`](../../socnav_assets_setup.md), then rerun the `#562` focused re-entry
  probe. The issue should auto-unblock once the private data-distribution mechanism (tracked in the
  #1456 thread) ships these assets to the compute hosts. No further agent-executable step exists until
  those bytes are present.
