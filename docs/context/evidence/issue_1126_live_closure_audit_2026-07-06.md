# Issue #1126 Live Closure Audit

Date: 2026-07-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/1126>

## Decision

Keep #1126 open.

Merged PRs delivered the fail-closed Stanford Drone Dataset (SDD) curation gate,
bring-your-own annotation-tree pin, one real `bookstore/video0` import, CPU smoke execution, and
post-reopen classifier correction. They do not yet deliver a `benchmark_ready` SDD-derived scenario
set. The only real SDD candidate with recorded CPU smoke is still `exploratory_only` because both
representative smoke horizons timed out.

This audit was produced after reading the live issue body, all issue comments, and merged PRs
#3765, #4564, #4616, #4638, #4647, and #4668. It is a closure audit and integration report only.
It adds no raw SDD files, generated scenario/map/provenance files, benchmark campaign output,
Slurm/GPU submission, or paper/dissertation claim.

## Criterion Table

| Criterion | Status | Evidence | Closure impact |
| --- | --- | --- | --- |
| SDD source staged or blocked with provenance. | Recorded, not live here. | E1 | Needs pinned local tree present. |
| Scene/video selected by rule. | Met for exploratory slice. | E2 | Candidate still not benchmark-ready. |
| Import command and provenance recorded. | Met for exploratory slice. | E3 | Provenance covers attempted candidate. |
| Scenario/map load validation passed. | Met for exploratory slice. | E4 | Artifacts remain ignored and exploratory. |
| Representative smoke succeeds or rejects explicitly. | Rejected for closure. | E5 | No `benchmark_ready` smoke exists. |
| `benchmark_ready` vs `exploratory_only` documented. | Met. | E6 | Keeps issue open unless boundary changes. |
| Only small reviewable artifacts committed. | Met. | E7 | Artifact policy is satisfied. |

## Evidence Notes

- E1: PR #4638 pinned the reviewed bring-your-own SDD annotation tree in
  `configs/data/sdd_staging_manifest.yaml`. The pin records tree SHA-256
  `66dec2c82b0a01b23bf9fa9acef352af86549e7ea749811ea4ef9c47003d4acf`, 60
  `annotations.txt` files, and 444959624 bytes.
- E2: PR #4638 selected `annotations/bookstore/video0/annotations.txt`, with 200021 usable
  `Pedestrian` points and 116 usable tracks at `--min-track-points 8`.
- E3: PR #4564 added decision-packet support. PR #4616 fixed the generated command to use
  importer's real `--annotations`, `--out-dir`, and `--meters-per-pixel` arguments. PR #4638
  recorded selected annotation checksum, official SDD URL/license, and `--meters-per-pixel 0.0417`.
- E4: PR #4638 recorded generated ignored map/scenario/provenance outputs under
  `output/sdd_curation/issue_1126_real_smoke` and load assertions for the generated scenario/map.
- E5: PR #4638 recorded two `simple_policy` CPU smoke runs, horizons 80 and 384. Both had
  `successful_jobs=1`, `failed_jobs=0`, no collisions, and timeout outcomes. PR #4647 classified
  this as `exploratory_only`; PR #4668 corrected the next action to
  `tune_or_select_benchmark_ready_candidate`.
- E6: PR #4647 added the final integration audit. PR #4668 corrected it after the issue was
  reopened, preserving `exploratory_only` while preventing accidental closure.
- E7: PRs #3765, #4564, #4616, #4638, #4647, and #4668 committed scripts, tests,
  manifest/checksum pointers, and compact evidence notes only. Raw SDD annotations and generated
  SDD scenario artifacts remain untracked.

## Reproduced Live Checks

Fresh isolated worktree:

```bash
scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/manage_external_data.py list
# sdd: status missing
# expected path:
# /home/luttkule/git/robot_sf_ll7.worktrees/issue-1126-closure-audit-verify-acceptance-criteria-v2/output/external_data/sdd  # allow-abs-path: verbatim recorded tool output (session worktree)
```

Pinned-tree root probe against the primary checkout output root also failed closed in this session:

```bash
ROBOT_SF_EXTERNAL_DATA_ROOT="$HOME"/git/robot_sf_ll7/output/external_data \
  scripts/dev/run_worktree_shared_venv.sh -- \
  python scripts/tools/sdd_curation_preflight.py --json
# staging_mode=proxy_schema_smoke
# dataset_backed=false
# benchmark_promotion_allowed=false
# output_classification=blocked
```

The absence of local SDD bytes here is not a contradiction of PR #4638's recorded evidence. It
means this closure-audit session cannot run the next real-data empirical slice without restaging
the licensed BYO tree.

## Remaining Work

- Restage or point `ROBOT_SF_EXTERNAL_DATA_ROOT` at the pinned SDD annotation tree.
- Run the next CPU-only empirical action: tune/calibrate `bookstore/video0` or select another SDD
  scene/video, then import and smoke-test it.
- Close #1126 only when the resulting evidence is `benchmark_ready`, or when the maintainer
  explicitly accepts the existing candidate as exploratory-only closure.
