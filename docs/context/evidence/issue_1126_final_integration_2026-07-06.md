# Issue #1126 Final Integration Audit

Date: 2026-07-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/1126>

## Decision

#1126 is ready for closure after this PR merges. Merged PRs now cover the first real Stanford Drone
Dataset (SDD) curation contract: local bring-your-own SDD annotations are pinned, one real
`bookstore/video0` candidate is selected, importer/smoke evidence exists, and the candidate is
intentionally accepted as `exploratory_only` rather than benchmark-ready.

The remaining live ambiguity after PR #4638 was the next action: tune/select another scene, calibrate
scale, or accept exploratory-only. This PR resolves that as `accept_exploratory_only` and makes the
choice executable through `scripts/tools/sdd_curation_preflight.py::classify_smoke_decision`. Future
benchmark-ready tuning can be a new enhancement; it is not required to close #1126's first-real-SDD
scenario curation contract.

No raw SDD files or generated scenario/map/provenance outputs are committed. No full benchmark
campaign, Slurm/GPU submission, or paper/dissertation claim is made.

## Criterion To Evidence

| Acceptance criterion | Status | Evidence |
| --- | --- | --- |
| #1497 staged official/BYO SDD source annotations locally, or recorded clear access/provenance failure keeps issue blocked. | Met | PR #4638 pins `expected_tree_sha256=66dec2c82b0a01b23bf9fa9acef352af86549e7ea749811ea4ef9c47003d4acf` for a reviewed local BYO SDD annotation tree: 60 `annotations.txt` files, 444959624 bytes. |
| One scene/video selected with recorded deterministic selection rule. | Met | PR #4638 selects `annotations/bookstore/video0/annotations.txt`; the probe found 200021 usable `Pedestrian` points and 116 usable tracks at `--min-track-points 8`. |
| Import command, source identity, source checksums, license/source URL, and scale assumptions recorded. | Met | `issue_1126_real_sdd_smoke_2026-07-06.md` records the source URL, Creative Commons Attribution-NonCommercial-ShareAlike 3.0 license, tree checksum, selected annotation checksum, import command, and `--meters-per-pixel 0.0417` scale assumption. |
| Generated scenario/map artifacts pass repository loading/parser validation. | Met | PR #4638 generated ignored files under `output/sdd_curation/issue_1126_real_smoke/`; structured load assertions passed for scenario, map, provenance, four pedestrians, and `meters_per_pixel=0.0417`. |
| At least one representative smoke run succeeds, or output explicitly rejected with reasons. | Met as exploratory-only | PR #4638 ran CPU `simple_policy` smoke at horizons 80 and 384: both had `successful_jobs=1`, `failed_jobs=0`, no collisions, and timeout outcomes. `classify_smoke_decision` now classifies this exact shape as `exploratory_only` with `recommended_next_action: accept_exploratory_only`. |
| Documentation states `benchmark_ready` vs `exploratory_only` status. | Met | `issue_1126_real_sdd_smoke_2026-07-06.md` states the candidate is `exploratory_only`. `classify_smoke_decision` only emits `benchmark_ready` for loaded, successful, no-timeout, no-collision smoke summaries. |
| Only small reviewable artifacts or durable pointers committed. | Met | Raw SDD annotations and generated SDD-derived files remain untracked under ignored local data/output paths. Committed surfaces are compact evidence notes, the staging checksum pin, and code/tests for the fail-closed/exploratory classification contract. |

## Closure Boundary

This closes the first-real-SDD curation issue as exploratory evidence. It does not claim a
benchmark-ready SDD scenario set. A future benchmark-ready follow-up should start from a new issue
that tunes the selected candidate or compares alternate SDD scenes against a stated success criterion.
