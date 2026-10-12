# Prediction beam-search freeze diagnostic (#10318)

**Historical first revision.** The [guarded safety revision](revision2/README.md) supersedes this result for the current PR head.

Development diagnostic only. Three seeds do not establish a release result or a general safety claim. No fallback/degraded rows are admitted. Fifteen of the eighteen affected episodes still time out; their global feasibility has not been established.

## Confirmed defect

The beam search admitted stage candidates using a different objective from the objective used to rank its sequences. Single-action scoring adds speed-dependent safety margins and omits the phase costs. At near-field states it can rank five stationary turns above every moving candidate, so the beam cannot consider a sequence that its own objective would prefer.

Code evidence: `robot_sf/planner/socnav_prediction.py:1595` builds the admission ranking, `:1617` freezes the candidate subset, and `:1637` expands sequences only from that subset. Compare the single-action speed margin at `:1086` and `:1207` with sequence scoring at `:1450` and its phase cost at `:1521`.

The exact seed-1001 following-human snapshot at step 100 reproduces this with the released predictor, actual differential-drive settings and static map geometry. Its old single-action top five all have zero speed; sequence scoring prefers 0.32 m/s. The old search returns 0.0 m/s; the fix returns 0.16 m/s, with 0.910 m predicted pedestrian surface clearance and 1.006 m swept static surface clearance. [Before](native_probe_before.json) and [after](native_probe_after.json) preserve the scored snapshot.

The minimal fix uses sequence scoring for the admission ranking. It changes no physical radii, safety thresholds, prediction cadence, forecast horizon, or candidate lattice. It does expose candidates previously pruned by the different single-action speed margin; the comparison therefore records lower-clearance episodes explicitly. It is one confirmed contributing defect, not a claim that the short-horizon planner should complete every scenario.

## Reproduction and test value

Comparator: `53721be3248600e08641a9816d094b605e3c93e4` (fetched origin/main). Runtime fix: `386e1f500d3fa8cdbdf490bd10b3b298ca3905f3`. The final delivery adds only the test-name refinement and diagnostic artifacts to that runtime fix.

The deterministic regression calls the real search using the released 0.0.8 configuration and a fixed eight-frame forecast approximating the audit state. It asserts positive command speed and independently checks physical body separation; a touching-body control must still stop. It needs neither a model download nor a simulation run.

```sh
scripts/dev/run_worktree_shared_venv.sh -- uv run python -m pytest \
  tests/planner/test_socnav_prediction_module.py::test_sequence_search_preserves_clear_near_field_progress -q
```

Run from the baseline checkout with only the new test copied in: **1 failed, 1 passed**, exit 1. The failure is `assert 0.0 >= 0.1` with `Clear forward progress was pruned out of the sequence search`. Run from the fix: **2 passed**, exit 0. Focused module: **36 passed**. Ruff check and format pass.

Test value: it catches safe motion being pruned out of the beam; the baseline failure demonstrates sensitivity; it asserts command/clearance behavior rather than which scorer is called; its two deterministic cases take approximately 0.04 seconds each before optional-environment startup overhead. Existing sequence-search coverage only checks determinism and a no-pedestrian progress escape, so it misses this near-field admission failure. No production test seam was added.

## Affected scenarios

The full 0.0.8 scenario matrix and prediction configuration were used through the canonical `_run_map_episode` path (reproducer: [run_affected.py](run_affected.py)), with 400 steps at 0.1 s and seeds 1001, 1002, 1003. Each pair has identical robot reset state. The registry predictor digest is `a28aed6d6ad7e1ebf597277ade1cf908efa6da038d0a9fcfdf80c7c31d8d1be1`; every row records loaded predictive foresight, eligible declared adapter execution and no predictor fallback.

[All 18 paired episodes and classifications](affected.csv): zero successes become three successes, zero collisions remain zero, and all minimum pedestrian surface clearances remain positive. The two following-human completions and entering-elevator seed-1003 completion demonstrate reachable episodes affected by the defect. The remaining timeouts are disclosed, not declared globally infeasible.

The original linked U2 audit report, review README, dossier and complete underlying three development traces were inspected. The dossier excludes its curated recordings from admitted evidence because analysis trace fields are incomplete. These source recordings are diagnostic leads; no source evidence or release admission status is changed.

## Empty-world behaviour gate

Both revisions completed all 102 expected actor-free episodes across the main suite and the three-width slice on seeds 1001–1002, with complete step traces. In both: main 94/96 successes, width 6/6, zero collisions. No new failures. This sweep covers the affected prediction arm; other planner implementations are unchanged.

[All 102 paired episodes](empty_world.csv) classify 96 changed traces: 73 changed trajectories retaining success, 21 changed commands with identical plant trajectories, and two existing physical-infeasibility timeouts (one with trajectory changes, one with command changes only). Six successes are unchanged. [Summary](empty_summary.json). Maximum paired position difference at a common step is 0.735 m.

```sh
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_empty_world_sweep.py \
  --head-sha <tested-commit> --suite both --seeds 1001 1002 \
  --workers 6 --arms prediction_planner --output-dir <private-artifact-directory>
```

The runtime comparison is pinned to the revisions above; later commits change only test naming and diagnostic artifacts.

The original narrow doorway has walls ending at y=4 and restarting at y=6 (`maps/successor_svg_maps/issue_9762_francis2023_narrow_doorway_goal_zone_entry_v2.svg:15`). A radius-1.0 body has zero surface gap in the 2.0 m throat; the positive swept-clearance requirement rejects contact. Timeout in that actor-free case is correct, rather than a target for tuning.

## Remaining risks

- The existing sequence objective lacks the single-action speed inflation. Using it consistently is a behavior change, and measured clearance reductions require independent review.
- Residual short-horizon safety/yield equilibria remain in fifteen affected episodes. These need separate diagnosis under #10318, not a forced positive-speed escape.
- No held-out evaluation, release benchmark, cross-planner ranking, model retraining or general collision-safety claim is made.
- Independent review of the final head is required; this lane does not merge.

Raw traces and command logs are preserved privately. These compact tables, scored snapshots and provenance are tracked for review and survive worktree cleanup.

Reproduce the affected cells from each frozen checkout with the same driver:

```sh
scripts/dev/run_worktree_shared_venv.sh -- uv run python docs/context/evidence/issue_10318/run_affected.py \
  --seeds 1001 1002 1003 --out <private-artifact-file>
```

[Manifest](manifest.json) records input, predictor and compact artifact digests. The driver wraps the same canonical call used for the comparison and rejects other seed bands before dispatch.
