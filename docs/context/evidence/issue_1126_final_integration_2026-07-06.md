# Issue #1126 Final Integration Audit

Date: 2026-07-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/1126>

## Decision

#1126 remains open after PR #4647. Merged PRs cover most first real Stanford Drone Dataset
(SDD) curation evidence: local bring-your-own SDD annotations pinned, one real
`bookstore/video0` candidate selected, and importer/smoke evidence recorded. The candidate is
`exploratory_only`, not benchmark-ready, because both representative smoke horizons timed out.

PR #4647 treated that ambiguity as `accept_exploratory_only`, but the issue was reopened because
#1126's benchmark-ready scenario-set ask remains unmet. The executable classifier now preserves the
`exploratory_only` classification while returning
`recommended_next_action: tune_or_select_benchmark_ready_candidate`.

No raw SDD files or generated scenario/map/provenance outputs are committed. No full benchmark
campaign, Slurm/GPU submission, or paper/dissertation claim is made.

## Criterion To Evidence

| Acceptance criterion | Status | Evidence |
| --- | --- | --- |
| #1497 staged official/BYO SDD source annotations locally, or recorded clear access/provenance failure keeps issue blocked. | Met | PR #4638 pins `expected_tree_sha256=66dec2c82b0a01b23bf9fa9acef352af86549e7ea749811ea4ef9c47003d4acf` for reviewed local BYO SDD annotation tree: 60 `annotations.txt` files, 444959624 bytes. |
| One scene/video selected recorded deterministic selection rule. | Met | PR #4638 selects `annotations/bookstore/video0/annotations.txt`; the probe found 200021 usable `Pedestrian` points and 116 usable tracks at `--min-track-points 8`. |
| Import command, source identity, source checksums, license/source URL, scale assumptions recorded. | Met | `issue_1126_real_sdd_smoke_2026-07-06.md` records source URL, Creative Commons Attribution-NonCommercial-ShareAlike 3.0 license, tree checksum, selected annotation checksum, import command, and `--meters-per-pixel 0.0417` scale assumption. |
| Generated scenario/map artifacts pass repository loading/parser validation. | Met | PR #4638 generated ignored files under `output/sdd_curation/issue_1126_real_smoke/`; structured load assertions passed for scenario, map, provenance, four pedestrians, and `meters_per_pixel=0.0417`. |
| At least one representative smoke run succeeds, or output explicitly rejected with reasons. | Met as exploratory-only; not closure-ready | PR #4638 ran CPU `simple_policy` smoke at horizons 80 and 384: both had `successful_jobs=1`, `failed_jobs=0`, no collisions, and timeout outcomes. `classify_smoke_decision` now classifies this shape as `exploratory_only` with `recommended_next_action: tune_or_select_benchmark_ready_candidate`. |
| Documentation states `benchmark_ready` vs `exploratory_only` status. | Met | `issue_1126_real_sdd_smoke_2026-07-06.md` states the candidate is `exploratory_only`. `classify_smoke_decision` only emits `benchmark_ready` for loaded, successful, no-timeout, no-collision smoke summaries and points timeout smoke toward another benchmark-ready curation action. |
| Only small reviewable artifacts or durable pointers committed. | Met | Raw SDD annotations and generated SDD-derived files remain untracked under ignored local data/output paths. Committed surfaces are compact evidence notes, staging checksum pin, and code/tests for the fail-closed/exploratory classification contract. |

## Closure Boundary

This does not resolve the issue. The next closure path is to tune/calibrate the selected candidate
or select another SDD scene/video that reaches benchmark-ready smoke criteria. This note does not
claim benchmark-ready SDD scenario set evidence.
