# Auditor preparation for issue #9952

## AUD2 RV10 repair (current)

The comparison below supersedes the initial AUD results retained later in this
note. Comparator: reviewed head `2e0d42e4505cce1be7c2ba5180b650fe42424eb9`. Source is the identical
verified public 0.0.7 bundle; both comparisons are **full raw-row scans**, not
projections. All producer and retained artifact hashes were verified. Compact
[comparison](evidence/issue_9952_auditor/aud2/comparison.json),
[before receipt](evidence/issue_9952_auditor/aud2/before-receipt.json), and
[after receipt](evidence/issue_9952_auditor/aud2/after-receipt.json) retain all four
signal-status denominators. The new channels were absent in the comparator;
"not registered" is not an observed zero.

Malformed physical Mappings and booleans now return measurement errors. Direct
non-finite measurements use the same error reason; public source inventory still
rejects unexpected non-finite rows before detector evaluation. Only documented
structured records are admitted by name. The scalar physical bounds and
undefined-tracking normalization are unchanged.

The [channel and floor contracts](../scenario_review/audit_scan.md#scalar-and-systematic-sensitivity-engine-v12)
add unconditioned planner/scenario outcome incidence and cross-planner feature
medians, and lower MAD floors to numerical-noise guards (absolute `1e-5`, relative
`1e-6`). All six timeouts in the reviewer's 24-success / 6-timeout single-planner
cohort retain an incidence flag with denominator 30 and timeout rate 0.2. Uniform
per-planner failures and uniform metric shifts have separate cross-planner
controls. The absolute incidence floor is zero: every adverse episode is a
candidate even without other planners; a cross-planner excess above 0.1 flags
its cell. This is intentionally sensitive descriptive triage, not statistical
significance or proof of unexpected failures. Exact scenario cells preserve the
production scenario-shard boundary; planner config hashes and outcomes do not
partition these controls.

`horizon_consistency` independently catches recorded timeout/steps/horizon and
reason/outcome mismatches. The unresolved parallel-traffic seed-126 bottleneck
cell is **flagged** by both horizon consistency and outcome incidence. Its raw
row records steps 400, runner horizon 600, simulator `max_episode_steps=400`,
termination reason `terminated`, timeout true and zero stall windows. The
smaller simulator limit explains the early timeout and is explicitly reported;
this establishes a configured runner/simulator discrepancy, not a causal
termination, goal-zone or deadlock defect. Its outcome-matched outliers remain
unavailable, while the cross-planner median-shift channel is clear.
The [fixture provenance](evidence/issue_9952_auditor/aud2/fixture-provenance.json)
pins archive/member/raw-line digests and was independently checked against archive
bytes, member line 1336. No held-out episodes were executed.

Both scans admit 20,160/20,160 rows, use 48 serial scenario shards and one process.
Before: 494.30 s; after: 935.33 s. Both have **zero detector errors**.

| Detector | Before flagged | After flagged | Before unavailable | After unavailable |
| --- | ---: | ---: | ---: | ---: |
| actuator_mismatch | 0 | 0 | 20160 | 20160 |
| cohort_multivariate_outlier | 6799 | 9102 | 358 | 358 |
| common_mode_anomaly | 0 | 0 | 20160 | 20160 |
| extreme_measurements | 3430 | 3430 | 0 | 0 |
| goal_adjacent_timeout | 0 | 0 | 2186 | 2186 |
| horizon_consistency | not registered | 1505 | not registered | 0 |
| initial_reset_anomaly | 0 | 0 | 20160 | 20160 |
| oscillation_limit_cycle | 0 | 0 | 20160 | 20160 |
| outcome_incidence | not registered | 11585 | not registered | 0 |
| outcome_metric_contradiction | 0 | 0 | 20160 | 20160 |
| planner_cohort_shift | not registered | 17220 | not registered | 0 |
| planner_disagreement | 0 | 0 | 20160 | 20160 |
| provenance_consistency | 0 | 0 | 0 | 0 |
| seed_outlier | 7651 | 10293 | 358 | 358 |
| stuck_no_progress | 0 | 0 | 20160 | 20160 |
| telemetry_integrity | 0 | 0 | 0 | 0 |
| trajectory_shape_outlier | 0 | 0 | 20160 | 20160 |

| Release-row gate detector | Before | After |
| --- | ---: | ---: |
| impossible_contact_speed | 97 | 97 |
| orbit_zero_progress | 4796 | 4796 |
| pedestrian_free_baseline_regression | 6 | 6 |
| same_step_all_planners | 26 | 26 |
| short_collision | 456 | 456 |
| universal_failure_unannotated | 74 | 74 |


Release-row gate findings remain 5,455. Preflight remains unrecorded (zero
`invalid_run_preflight_mismatch` findings does not establish preflight validity).
Existing trace/initial-state contract unavailability remains explicit. The broad
new median-shift channel flags 17,220 rows; intended planner/config differences
can explain these candidates. No population precision or reduced-noise claim is
made for this head, and no thresholds were tuned against this campaign.

Regression proof: **28 final new nodes fail on the reviewed head; 28 pass on the
fixed implementation**. This includes the four imported reviewer public-scan
nodes, subgroup/uniform-shift/noise/horizon sensitivity and non-finite/boolean
physical corruption. The [test-value gate](evidence/issue_9952_auditor/aud2/test_value.md)
records behavior, credible regression, coverage gap and no production seams;
[red output](evidence/issue_9952_auditor/aud2/red.txt),
[green output](evidence/issue_9952_auditor/aud2/green.txt),
[focused output](evidence/issue_9952_auditor/aud2/focused.txt): **365 passed,
2 skipped**. The focused paths add `test_aud2_refutations.py` and
`test_aud2_detectors.py` to the initial AUD command below. Ruff check/format and
`git diff --check` pass. The
[840-row shard proof](evidence/issue_9952_auditor/aud2/new-shards-proof.json)
compares each new channel's indexed peers with complete-campaign peer lists,
with zero differences in complete signal records; it is structural equality
proof, while the literal sensitivity tests establish the new detector behavior.

The [queue validation](evidence/issue_9952_auditor/aud2/queue_validation.json)
loads the 1,238,250,933-byte full queue through the CLI and its top
100 has 100 distinct campaign/scenario/seed cells. All 20,160 input candidates
remain retained.

Full suite/readiness pipeline, hosted-CI acceptance, 0.0.8 scanning, planner
execution and trace adjudication remain unrun. No subagents, merges, force
pushes, release actions or claim admission. Large artifacts remain reconstructable
local diagnostics in `/home/luttkule/aud2-evidence/`; the public bundle remains
the durable input. Delivery requires a new review at the pushed head.

## Initial AUD delivery (historical, reviewed head 2e0d42e4)

This is offline tooling validation on the published **0.0.7** rows, not a 0.0.8
campaign audit or a release certificate. No planner episodes, Slurm jobs, merges,
or subagents were used. Main comparator: `46809b6bc18caef2a9cd82686e9670c224459c42`.

## Fixes and boundaries

- Orbit uses typed `deadlock_stall.stall_window_count` with an `ok` status and
  non-completion. Completed episodes with transient stalls stay clear on this channel.
- `time_to_collision_min` previously missed the TTC aliases and fell through to
  the collision-count bound of 1.0. It now uses the TTC minimum in seconds (0.0).
  Negative TTC and collision counts exceeding their bound still flag.
- Outliers use outcome-matched cohorts and a MAD floor of
  `max(MAD, 0.01 feature units, 0.05 * abs(median))`; robust z divides by 1.4826.
  Floors are configurable and disclosed. Small outcome cohorts are unavailable.
  These scores remain uncalibrated triage signals.
- A top-level `failure` with explicit timeout/non-collision/non-completion evidence
  is a terminal outcome. Explicit execution failures remain unsupported.
- Documented positive-infinity disabled-tracking and pedestrian-free NaN separation
  sentinels become null with path/token/reason receipts. Unexpected non-finite
  values still fail admission; original source bytes remain unchanged.
- Verified loader-marked release rows keep planner and episode config hashes in
  separate namespaces, use stable planner config for seed cohorts, and qualify
  shared episode IDs by arm. Source-member and episode-hash disagreements fail.
- Scan cap: 1 GiB. Queue cap: 2 GiB / 50 million JSON nodes. Existing depth,
  path, symlink, individual string, and row-count protections remain.
- Cohorts are indexed once. The release entry point scans 48 scenarios serially,
  retains the bundle provenance, and withholds the completion receipt if producer
  source changes. The first 100 queue slots represent distinct cells; duplicates
  remain later in the queue. With fewer than 100 eligible cells, repeats fill it.
- `common_mode_failure` signal reasons map to the highest benchmark-defect band.
  Detector/method and queue-policy revisions distinguish the changed semantics.

Optional new reset/goal-clearance, early-wall-contact, and horizon catchers were
not added. Published rows lack the initial-state/trace contracts needed by several
existing detectors; those remain explicitly unavailable rather than inferred.
Preflight was not loaded (`require_preflight=False` for this diagnostic replay).

## Input and custody

[Published release](https://github.com/ll7/robot_sf_ll7/releases/tag/paper-matrix-v2-h600-s30-2026-09-07f7e8d43084de748915e1b1eb8b2a1603357c6e),
asset `issue9431_release_benchmark_data_0_0_7_07f7e8d43084_20260922_publication_bundle.tar.gz`:

- Bundle SHA-256: `684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`.
- Manifest SHA-256: `3976f5e0e4f5d8d8b601099e363475214706516dde42949b5673f379d626dc5d`.
- 14 arms × 48 scenarios × 30 seeds = 20,160 rows; raw JSONL totals 604,837,162 bytes.
- The canonical loader verifies archive, manifest, and member digests.
- [Vendored samples](../../tests/fixtures/analysis_workbench/aud_release_0_0_7.json)
  retain exact raw lines, line/member/archive SHA-256, source commit and location.
  Counterfactual fixture edits are named explicitly in the tests.
- Small results, test output, comparator and hand judgments are tracked under
  [evidence/issue_9952_auditor](evidence/issue_9952_auditor/).
  Large bundle, shards and queue remain local in `/home/luttkule/aud-evidence/`;
  they are reproducible diagnostics, not durable public benchmark evidence.
  The public bundle is the durable input. No runtime depends on these local outputs.
  Text-log/CSV presentation copies normalize line endings/trailing whitespace;
  their sidecars retain the original source SHA. Receipts retain exact bytes.
  Generated evidence is SHA-linked and marked `AI-GENERATED NEEDS-REVIEW`.

## Full-row comparison

Parent BA-01 cannot directly admit the published rows (source cap, scoped hash
aliases, sentinels and terminal failure status). Therefore detector comparisons
use the **identical finite scalar/outcome projection** on both revisions, with
terminal status omitted; this isolates detector semantics from admission fixes.
The [comparator](evidence/issue_9952_auditor/replay_projection.py) preserves arm
identity and planner config, and invents no initial-state or trace equivalence.
The fixed raw-row scan independently admits all 20,160 rows; its detector counts
match the fixed projection. Both runs have zero detector errors.
The final 48-shard scan took 501.90 s with one process. All retained artifact and
producer digests were verified. The real queue input is 824,506,173 bytes; its CLI
loads all candidates and its first 100 entries cover 100 distinct cells.

| Detector | Before flagged | After flagged | After unavailable |
| --- | ---: | ---: | ---: |
| extreme_measurements | 18202 | 3430 | 0 |
| seed_outlier | 11203 | 7651 | 358 |
| cohort_multivariate_outlier | 10221 | 6799 | 358 |
| goal_adjacent_timeout | 0 | 0 | 2186 |
| provenance_consistency | 0 | 0 | 0 |
| telemetry_integrity | 0 | 0 | 0 |
| actuator_mismatch | 0 | 0 | 20160 |
| common_mode_anomaly | 0 | 0 | 20160 |
| initial_reset_anomaly | 0 | 0 | 20160 |
| oscillillation_limit_cycle | 0 | 0 | 20160 |
| outcome_metric_contradiction | 0 | 0 | 20160 |
| planner_disagreement | 0 | 0 | 20160 |
| stuck_no_progress | 0 | 0 | 20160 |
| trajectory_shape_outlier | 0 | 0 | 20160 |

Gate counts on the unmodified published rows:

| Detector | Before | After |
| --- | ---: | ---: |
| orbit_zero_progress | 4796 | 4796 |
| short_collision | 456 | 456 |
| impossible_contact_speed | 97 | 97 |
| universal_failure_unannotated | 74 | 74 |
| same_step_all_planners | 26 | 26 |
| pedestrian_free_baseline_regression | 6 | 6 |
| invalid_run_preflight_mismatch | 0 | 0 (preflight unavailable) |

Total gate findings: 5455 both ways. The frozen bundle's Boolean already equals
the typed stall/non-completion predicate. The counterfactual regression proves
the changed binding; no count reduction is claimed for orbit.

## Hand review

[20 recorded judgments](evidence/issue_9952_auditor/hand_sample.csv) cover five
positive-TTC removals, five zero-MAD removals, three short collisions, three
implausible contact speeds, two orbit/stall findings and two same-step cells.
I read the outcome, event ledger, metrics and cohort summaries. Precision as
**row-supported actionable triage candidates** is 9/20 (45%) before, 9/10 (90%)
among retained flags after. This intentionally targeted sample is not a random
estimate of detector/population precision, and confirms no causal defects.
One retained PPO orbit case has 95.5% progress and only 0.2 s stationary time;
its stall windows are a weak candidate and need trace review. No tuning against
the held-out campaign or planner execution was performed.

## Regression proof

Parent command, from the clean parent checkout, using the shared fixed dependency
environment and the new tests by explicit path:

```bash
scripts/dev/run_worktree_shared_venv.sh --venv /home/luttkule/aud-work/.venv -- \
  pytest /home/luttkule/aud-work/tests/analysis_workbench/test_aud_9952.py -q --no-cov
```

[Parent output](evidence/issue_9952_auditor/parent-failures-final.txt): **15 failed**,
each for its intended assertion/admission bound, not an import or fixture error.
All node IDs have prefix `tests/analysis_workbench/test_aud_9952.py::`:

```text
test_published_positive_ttc_uses_seconds_bound
test_orbit_uses_recorded_stall_windows_not_failure_boolean
test_outlier_mad_floor_rejects_numerical_noise[seed_outlier]
test_outlier_mad_floor_rejects_numerical_noise[cohort_multivariate_outlier]
test_outlier_cohorts_match_terminal_outcome[seed_outlier]
test_outlier_cohorts_match_terminal_outcome[cohort_multivariate_outlier]
test_release_failure_timeout_is_admitted_but_execution_failure_is_not
test_disabled_tracking_infinity_is_missing_not_row_corruption
test_verified_release_arm_keeps_config_scopes_and_episode_identity
test_scan_shards_cohorts_without_changing_signals
test_scan_accepts_20160_rows_above_old_source_cap
test_common_mode_failure_has_benchmark_defect_priority
test_top_100_represent_distinct_cells_and_retain_all_episodes
test_queue_loads_20160_candidates_above_old_byte_and_node_caps
test_release_seed_cohorts_use_planner_config_not_seeded_episode_hash
```

[Fixed output](evidence/issue_9952_auditor/focused-final.txt): **337 passed, 2 skipped**.
Ruff check, Ruff format check and `git diff --check` passed. Explicit paths:

```bash
scripts/dev/run_worktree_shared_venv.sh -- pytest \
  tests/analysis_workbench/test_aud_9952.py \
  tests/analysis_workbench/test_audit_detectors.py \
  tests/analysis_workbench/test_audit_scan.py \
  tests/analysis_workbench/test_audit_queue.py \
  tests/analysis_workbench/test_release_row_anomalies.py \
  tests/analysis_workbench/test_audit_service_adapters.py \
  tests/analysis_workbench/test_audit_service_next.py \
  tests/analysis_workbench/test_audit_coverage.py -q --no-cov
```

The broad `pr_ready_check`/full suite was not run. This draft delivery reports
focused offline proof, with no readiness stamp or merge-ready claim.

## Test-value gate

No new or changed test needs a production-only test seam. The shard observer
monkeypatches the existing detector call and compares against complete cohorts.
For new test families the other three answers are:

| Protected behavior | Credible regression | Existing coverage gap |
| --- | --- | --- |
| Seconds-based TTC | Restore collision-name fallthrough | Extreme tests cover ttc_s, not published time_to_collision_min |
| Typed stall/non-completion | Reuse flat deadlock or flag completed transient stalls | Old release fixture has both geometric orbit and Boolean |
| Floors in both outlier families | Restore zero-MAD infinite shortcut | Statistical contract edge test expected huge scores |
| Outcomes in both cohorts | Mix timeout and success termination | Compatible-key test mixed outcomes |
| Terminal timeout admission | Treat top failure as execution failure | Scan tests cover collision labels, not published timeout |
| Known undefined diagnostics | Reject all sentinels or excuse arbitrary nonfinite telemetry | Nonfinite test covers corrupt promised telemetry |
| Scoped hashes and arm identity | Treat planner/episode hashes as aliases | Synthetic scan has one config namespace |
| Stable seed cohorts | Key peers by seeded episode hash | Synthetic peers have constant config_id |
| Complete indexed shards | Give each call the entire campaign | Scan tests check accounting, not partition size/equivalence |
| Full-size source admission | Restore 64 MB cap | Existing campaign fixtures are small |
| Common-mode defect priority | Fall through to safety band | Fixed-priority test uses benchmark/config text |
| Diverse head, all rows retained | Remove representative window | Existing rank tests have few cells |
| Full-size queue admission | Restore byte or node cap | Checked-in queue fixture has four candidates |

Changed existing tests keep these contracts:

- `test_cross_planner_and_cohort_detectors_use_compatible_keys`: matched outcomes
  preserve the seed comparison; the new outcome test catches cross-outcome mixing.
- `test_statistical_detectors_cover_contract_edges_and_missing_features`: a
  hand-derived finite score `1 / (1.4826 * 0.05)` replaces the shortcut expectation;
  the new floor cases reject numerical noise. No oracle calls the implementation.
- `test_checked_in_fixture_supports_offline_rank_and_select`: packet policy version
  is now v1.1; restoring stale policy identity fails. New ranking tests verify behavior.
- `test_updated_legacy_report_keeps_collision_gate_opt_in`: intentional orbit
  method/missingness golden revision, while collision-contract activation remains
  opt-in. Restoring accidental activation fails its explicit contract assertions;
  the new orbit counterfactual independently catches the wrong input channel.

## Reproduce

Use a verified copy of the public bundle and fresh output directories. In each
parent/fixed checkout run the same tracked comparator with that checkout on
`PYTHONPATH`. It writes counts and flagged-row summaries without running episodes:

```bash
PYTHONPATH="$PWD" /path/to/shared/venv/bin/python \
  /path/to/fixed/docs/context/evidence/issue_9952_auditor/replay_projection.py \
  /path/to/publication_bundle.tar.gz /new/comparator-output
scripts/dev/run_worktree_shared_venv.sh -- python scripts/analysis/scan_release_audit.py \
  --bundle /path/to/publication_bundle.tar.gz --output-dir /new/private-scan-output
scripts/dev/run_worktree_shared_venv.sh -- python -m robot_sf.analysis_workbench.audit_queue \
  rank --input /new/private-scan-output/queue.json --policy fixed --limit 100 \
  --output /new/top100.json
```

The [comparison](evidence/issue_9952_auditor/comparison.json) retains full status
denominators. The tracked completion receipt pins source/producer/artifact digests;
it is diagnostic replay evidence and does not attest missing preflight or traces.
