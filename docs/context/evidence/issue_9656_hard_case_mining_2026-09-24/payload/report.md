# Issue #9656: historical benchmark hard-case mining

Evidence tier: `diagnostic_only`. The artifacts describe one checksum-pinned historical simulator release and four bounded same-scenario reruns. They do not establish planner safety, benchmark generalization, or real-world safety.

## Source

- Public Release 0.0.2 bundle SHA-256: `64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90`.
- Campaign: `paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316`; source commit `f7ebdcae2375d085e925213197a75a386e26a79c`.
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`; SHA-256 `d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5`.
- Selector: `benchmark-showcase.v1` at PR #9662 head `477f14c1b4b052ad407a34a71caace6618a75eeb`; selected summary SHA-256 `c2f0b4c0b85303e3547e4ce13f5676b45c886b6b9593a78a7a4d6016fdb39c1f`.
- Materializer commit: `a752661b3518a0d5776b785516dec614e7035f27`; the four replay receipts retain their actual evaluator commit `5cccee50be333adceee4c978b54bf63d32454cc9`. The final resume/materialization did not rerun simulations.
- Historical denominator: 987/987 episode rows present; 0 missing, 0 duplicate, and 0 malformed rows.
- Existing camera-ready analyzer status: `passed`; 7 planner-level findings were retained.

The source selector summary was rerun twice at the same output path and produced byte-identical SHA-256 `c2f0b4c0b85303e3547e4ce13f5676b45c886b6b9593a78a7a4d6016fdb39c1f`. A second run at a different output path produced the same case identities but different analyzer report hashes because that diagnostic records absolute paths.

## Selection and materialization

- 36 unique cases from 7 scenario IDs, 6 families, and 7 planners.
- The nine named event/metric groups selected 15 cases each; overlapping groups resolve to 36 unique stable case IDs. There is no opaque composite score.
- Planner counts: `{"goal": 8, "orca": 4, "ppo": 5, "prediction_planner": 5, "sacadrl": 4, "social_force": 6, "socnav_sampling": 4}`.
- Scenario-family counts: `{"accompanying_peer": 11, "blind_corner": 9, "bottleneck": 9, "circular_crossing": 4, "cross_trap": 2, "crowd_navigation": 1}`.
- Full source contains 241 canonical collision events with collision termination; both `metrics.collisions` and `metrics.total_collision_count` are present and non-positive for all 241 rows. The slice preserves and flags 17 such rows without rewriting source events or metrics.
- One-row, one-source-seed scenario matrices and planner configuration snapshots were materialized for 36/36 cases; their hashes are in `summary.json`.
- Existing showcase renderer status: `{'unavailable': 36}`. The release rows contain no `replay_steps`, so no trajectory renderer was called and no source animation is available.

## Bounded replay

Four distinct cases produced one replay episode each, within the five-case ceiling. An initial `--scenario-id` invocation failed for those same four cases: each command scheduled three jobs but wrote zero episode rows (12 failed setup jobs total) because map resolution ran from `scenario_path='.'`. The corrected explicit one-row matrix path then produced one episode per case; the initial failures are preserved separately in `summary.json` and are not planner outcomes. The final slice refresh reused the corrected receipts and did not start another replay. Five selected PPO cases were unavailable because archived model files were absent; the remaining 27 selected cases were not attempted under the bounded replay budget.

All four replay episode identities, planner configuration hashes, execution modes, and canonical event flags match their selected source cases. Every replay was run at `5cccee50be333adceee4c978b54bf63d32454cc9`, while the source campaign is at `f7ebdcae2375d085e925213197a75a386e26a79c`; numeric comparisons are therefore classified `mismatch_different_revision`, not exact historical replay.

| Case | Planner | Scenario / seed | Execution | Episode outcome | Event flags | Metric mismatches | Episode SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `case-2fa6f8a28dafddc0` | orca | francis2023_accompanying_peer / 111 | `adapter → adapter` | `failure` | collision_event=match, route_complete=match, timeout_event=match | clearing_distance_min | `2df8e8697022e5bf1afb6036bebea10eae7c7e81ec2465d6de27159018c791ac` |
| `case-4a01770a7c387ea0` | goal | francis2023_circular_crossing / 111 | `native → native` | `collision` | collision_event=match, route_complete=match, timeout_event=match | clearing_distance_min, collisions, comfort_exposure, force_exceed_events, total_collision_count | `a1bc1c9df1277aa98f48d66d7f5005451038f775ecd7b2f0b73ce84efc51d478` |
| `case-5a33e6eb8e4a4d12` | orca | francis2023_circular_crossing / 111 | `adapter → adapter` | `collision` | collision_event=match, route_complete=match, timeout_event=match | clearing_distance_min, collisions, comfort_exposure, force_exceed_events, total_collision_count | `24a69c12b7eb39792a4e403996e9c95cdb2604cef7b034c689147fa58b0d4c1d` |
| `case-b08371fb5cd09502` | goal | francis2023_accompanying_peer / 111 | `native → native` | `failure` | collision_event=match, route_complete=match, timeout_event=match | clearing_distance_min | `e5eef6e52003b17c15f3fb789ad161433ad35b35ed278dee0d9d7041c5b19737` |

The four replay rows had these source/replay revision-divergent metric differences: two accompanying-peer cases differed in minimum clearance only; both circular-crossing cases differed in minimum clearance, collision count, comfort exposure, force-exceed count, and total collision count. All other compared named metrics matched. Revision difference is recorded, but no causal attribution for the metric changes is made.

The original replay outputs predate output hashing in the first receipt. Their SHA-256 values were first calculated while resuming the preserved local output and verified on subsequent copies; this is custody from that capture onward, not an independently signed checksum from the initial runner process. Source environment identity was absent from the release bundle; replay Python/platform and lockfile digest are retained in `summary.json`.

## Reproduction and limits

The machine-readable summary contains the exact input digests, row hashes, selected case inventory, per-case replay command, outcomes, named metrics, checksums, and availability state. Reproduction commands are also embedded in the summary. Failed-command paths are normalized to repository-root-relative paths; run those receipts from the repository root. The raw bundle and case/replay output trees are not copied into git; source cases are recoverable from the public release and per-row source references.

This produces an evidence-backed historical challenge slice only. It does not run a new benchmark campaign or search, render new trajectories, admit cases into issue #9652’s versioned corpus, or support planner-ranking/safety claims. Cases remain historical regression candidates pending the corpus owner’s admission policy.
