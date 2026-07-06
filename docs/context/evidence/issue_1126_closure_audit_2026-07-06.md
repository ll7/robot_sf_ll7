# Issue #1126 Closure Audit

Date: 2026-07-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/1126>
Parent: <https://github.com/ll7/robot_sf_ll7/issues/1091>
Context note: [`issue_1126_sdd_curation_preflight.md`](../issue_1126_sdd_curation_preflight.md)

## Decision

Keep #1126 open. The issue is no longer blocked on an absent local SDD annotation tree in this
checkout: the reviewed BYO annotation tree is pinned in `configs/data/sdd_staging_manifest.yaml`.
A first real scene import and CPU smoke are recorded in
[`issue_1126_real_sdd_smoke_2026-07-06.md`](issue_1126_real_sdd_smoke_2026-07-06.md), but the
candidate is `exploratory_only` because the smoke runs timed out.

No raw SDD files are committed. No benchmark campaign, Slurm/GPU job, or paper-facing claim is made.

## Acceptance Criteria Evidence

| Acceptance criterion | Status | Evidence |
| --- | --- | --- |
| Official/BYO SDD annotations staged locally or clear access/provenance failure recorded | Met for this checkout | `expected_tree_sha256=66dec2c82b0a01b23bf9fa9acef352af86549e7ea749811ea4ef9c47003d4acf` is pinned for the reviewed local BYO tree: 60 `annotations.txt` files, 444959624 bytes. Raw data remains under ignored external-data storage. |
| One scene/video selected with a recorded deterministic selection rule | Met for exploratory slice | `annotations/bookstore/video0/annotations.txt` selected; probe found 200021 usable `Pedestrian` points and 116 usable tracks. |
| Import command, source identity, checksums, license/source URL, and scale assumptions recorded | Met for exploratory slice | Real smoke note records tree checksum, selected annotation checksum, source URL/license, and `--meters-per-pixel 0.0417` scale assumption. |
| Generated scenario/map artifacts pass repository loading/parser validation | Met for exploratory slice | Import wrote map/scenario/provenance to ignored `output/sdd_curation/issue_1126_real_smoke`; YAML/JSON load assertions passed. |
| At least one representative CPU smoke run succeeds, or output is explicitly rejected with reasons | Met for exploratory slice | `simple_policy` CPU runs at horizons 80 and 384 both wrote one episode with `successful_jobs=1`, `failed_jobs=0`, no collisions, timeout outcome. Classified `exploratory_only`. |
| Documentation distinguishes `benchmark_ready` from `exploratory_only` | Met | The real smoke note classifies this candidate as `exploratory_only`; no benchmark-ready or paper-facing claim is made. |
| Only small reviewable artifacts or durable pointers are committed | Met | This PR commits checksum, code/test guardrails, and durable evidence notes only. Raw SDD files and generated importer outputs stay in ignored `output/`. |

## Contributing PRs

| PR | Contribution |
| --- | --- |
| #1091 / #1127 | SDD trajectory scenario importer, `scripts/tools/import_sdd_scenarios.py`. |
| #3765 | Fail-closed SDD curation readiness preflight and tests. |
| #4564 | Metadata-only decision packet writer. |
| #4616 | Closure audit plus runnable importer handoff command regression guard. |
| This PR | Pins the reviewed BYO SDD annotation tree checksum, fixes no-probe preflight overclaim, and records the first real SDD import/smoke as `exploratory_only`. |

## Reproduced Validation

Commands use `ROBOT_SF_EXTERNAL_DATA_ROOT=output/external_data` (repo-relative; run from the
repo root) so the worktree reads the existing ignored local BYO data without copying it.

```bash
scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/manage_external_data.py --json sdd-mode
# exit 0 after this PR: mode dataset_backed_prior, dataset_backed true

scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/sdd_curation_preflight.py --json
# exit 0: staging mode dataset_backed_prior; benchmark_promotion_allowed false without annotation probe

scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/sdd_curation_preflight.py \
  --annotation "${ROBOT_SF_EXTERNAL_DATA_ROOT}/sdd/annotations/bookstore/video0/annotations.txt" \
  --min-track-points 8 \
  --require-benchmark-ready \
  --json
# exit 0: selection_satisfiable true, benchmark_promotion_allowed true
```

## Remaining Checklist

- [x] Pin the staged SDD annotation directory checksum so `resolve_sdd_scenario_prior_mode` can
      report `dataset_backed_prior` for the reviewed tree.
- [x] Select one scene/video and record identity, source URL/license, selected-file checksum, and
      `--meters-per-pixel` assumption.
- [x] Run the importer command and load the generated scenario/map.
- [x] Run one CPU smoke path; mark `benchmark_ready` or `exploratory_only`, or reject with reasons.
- [ ] Decide the next benchmark-ready action: tune or choose another selected scene, calibrate scale,
      or intentionally accept this candidate as exploratory-only evidence.
