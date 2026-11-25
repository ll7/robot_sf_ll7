# EXSEL-7.5-GUARD-PAIR: trace-example screening

**Development-seed diagnostic. New current-version execution on an experimental rebuild
of the 0.0.8 release code. Not release evidence. Merge only after an independent
exact-head review.** The claim ceiling is the guard/escape replication's:
within-controller development evidence (PROVISIONAL).

## Question

The dissertation's frozen trace-example selection protocol (ll7/diss
`docs/context/trace_example_selection_protocol.yaml`, commit `b95d41b524f1`, decision
EXSEL-TRACE-SELECTION-PROTOCOL-2026-10-28) names claim **EXSEL-7.5-GUARD-PAIR**:
within hybrid v4 (continuous) with the escape bundle on, does switching the hard
guard off and on turn a wall collision into a completion on the same seed?
Which development seed, taken in the committed screening order, is the first to show
that pattern in the frozen cell `classic_cross_trap_high`, inside the frozen band?

The protocol is the preregistration. [contract.yaml.txt](contract.yaml.txt) copies its
identity, rules, outcome predicates, sources, the claim entry and the acquisition
spec byte for byte, and adds an `execution:` block that records how this lane ran
it. The contract was committed as `689959b7a412` at 07:48:43 UTC and pushed at
07:48:46 UTC, before the first screening episode started at 07:48:52 UTC.
The suffix preserves the YAML bytes while separating the copied dissertation
references from repo-local path checks; the original filename was `contract.yaml`.

## Result

**ACCEPTED seed 1188 at screening index 1 of 40.**

| Arm | Outcome | Collision kind | Termination | Steps | Time to goal | Band check |
| --- | --- | --- | --- | ---: | ---: | --- |
| G0E1 (guard off, escape on) | C | obstacle (ped 0, obstacle 1) | collision | 155 | n/a | none frozen |
| G1E1 (guard on, escape on) | S | n/a | success | 347 | 34.7 s | inside [33.2, 41.6] |

Only one seed was screened, because the first seed was accepted
([screening_ledger.csv](screening_ledger.csv),
[screening_result.json](screening_result.json)). The claim-keyed report prints these
base rates for captions. They come from the chosen cell's 30 replication pairs
(seeds 1031–1060), not from this trace:

- joint pattern (G0E1 obstacle collision and G1E1 success on the same seed): **27/30**
- unconditional: G0E1 30/30, G1E1 27/30; conditional G1E1 given G0E1: 27/30
- replication seeds that would also pass the frozen G1E1 time band: 13/30
- group frequency over the eight static-constriction cells: 93/240

## Rebuild of the execution source (checks a–e, all pass)

The replication ran at `0035fba05c9b`. The rebuild applies the first ablation packet's
`acquisition-source.patch.gz` to release commit `373dbfde4f39` and commits the result
as `011c24045d67` on branch `exscr2-guardpair-exec`.
[provenance/rebuild_checks.json](provenance/rebuild_checks.json) records:

- (a) patch digest `1314f3c8…a1dfca2` equals the SHA256SUMS entry at `acd878b4`.
- (b) `git apply --check` succeeds on a fresh worktree at `373dbfde`. The patch touches
  only `robot_sf/planner/hybrid_rule_local_planner.py` (+15/−5) and
  `tests/planner/test_guardesc_ablation.py` (+114). The rebuilt tree `7fa66da1`
  equals the tree of `0035fba05c9b`.
- (c) controller config SHA-256 `044931a3…eff94888` matches. The arm configs are
  byte-identical to the replication's `inputs/G0E1.yaml` and `inputs/G1E1.yaml`.
- (d) Python 3.13.14, NumPy 2.4.6, PyYAML 6.0.3, `uv.lock` `def82098…`, CPU only and
  one numerical thread all match `provenance/environment.json` of the replication.
  Workers (2 instead of 4) and host differ; seeding is per episode.
- (e) parity on `classic_cross_trap_high`, seeds 1031–1035, G0E1 and G1E1: all 10
  rows equal the replication's compact rows on outcome class, collision kind,
  success, collision and timeout events, steps, termination reason, pedestrian and
  obstacle collision counts, the guard factor, mode and fallback count
  ([provenance/parity_e.json](provenance/parity_e.json)).

The claim-keyed report (`diss_worked_example_nomination.py --claims-protocol`) exited 0
with "Frozen record: MATCH"
([provenance/claim_keyed_report_guard_pair.txt](provenance/claim_keyed_report_guard_pair.txt)).
Its screening order equals an independent recomputation: the 1161–1200 band sorted
by SHA-256 of `exsel-7.5-guard-pair-2026-10-28:<seed>`
([screening_order.txt](screening_order.txt)).

## Acceptance as applied

The class and band logic copies the protocol script's guard-row projection
(`guard_record`, `class_matches`):

- outcome S if `metrics.success`, else C if `outcome.collision_event`, else T if
  `outcome.timeout_event`;
- collision kind pedestrian if `metrics.ped_collision_count > 0`, else obstacle if
  `metrics.obstacle_collision_count > 0`;
- time to goal `round(steps × 0.1, 6)` s, for successes only;
- bands inclusive; null bands not checked.

Only G1E1's time-to-goal band is frozen. One test per rule:
[tools/test_rules.py](tools/test_rules.py) (13 tests).

Secondary reading (release-row obstacle predicate, also counting
`metrics.wall_collisions > 0`): G0E1 has `wall_collisions` 1.0 and G1E1 0.0, so the
verdict is the same.

## Mode and fallback

Every raw row in parity (10), screening (2), trace rerun (2) and reconstruction (2)
has `planner_kinematics.execution_mode = adapter` (`HybridRuleLocalPlannerAdapter`).
Each also has `planner_runtime.fallback_count = 0` and no runtime fallback/degraded
marker (`runtime_fallback_or_degraded_marker`, the same check as the replication's
`project.py`). The ledger carries `mode`, `fallback_count` and
`fallback_or_degraded_marker` per row. The screening driver stops on any violating row;
none occurred.

## Trace (acquisition step 3)

Seed 1188 was rerun for both arms with `record_simulation_step_trace: true` and
`record_planner_decision_trace: false`
([inputs/exsel75_trace_s1188.yaml](inputs/exsel75_trace_s1188.yaml); it differs from
the screening config only in name, scenario-file name and that flag). Both rerun rows
equal the screening rows exactly on outcome class, termination reason, steps,
pedestrian and obstacle collision counts, `near_misses` and minimum pedestrian
separation, so the trace is accepted
([provenance/trace_compare_s1188.json](provenance/trace_compare_s1188.json)).
The full rerun rows differ from the screening rows only in:

- the trace payload;
- the trace-dependent `paired_effect_*` and `robot_force_samples` metadata;
- configuration hashes and episode ids (the trace flag enters the hash);
- timestamps.

No metric value differs.

Compact trace files:

- [trace/trace_G0E1.csv](trace/trace_G0E1.csv) (155 steps) and
  [trace/trace_G1E1.csv](trace/trace_G1E1.csv) (347 steps). Recorded per step: time,
  robot pose, heading and velocity, selected and applied commands, obstacle and
  pedestrian collision flags, terminated/truncated, minimum pedestrian surface clearance.
- `obstacle_envelope_distance_m_derived` is **derived, not recorded**: the distance
  from the robot centre to the nearest map obstacle or boundary segment, minus the
  1.0 m robot radius. Its sign agrees with the recorded obstacle-collision flag on
  155/155 and 347/347 steps.
- [trace/trace_meta.json](trace/trace_meta.json): map file and SHA-256, terminal events
  and the raw-row SHA-256s.

The guard's rejected commands were not recorded. The planner decision trace stayed
off, so behaviour was not put at risk. The full raw trace rows are in the preserved
bundle.

## Comparison contract for reconstruction

A rerun at execution commit `011c24045d67` with the same config must match the
screening rows exactly on every field except `timestamps.start`, `timestamps.end`,
`wall_time_sec` and `timing.steps_per_second`. This was verified with a 1-seed
reconstruction rerun of seed 1188: both raw rows differ in exactly those four fields
([provenance/recon_compare_s1188.json](provenance/recon_compare_s1188.json)). Paths,
host, commit of the packet branch and wall times legitimately differ. The ledger and
the trace CSV/JSON regenerate byte-identically from the raw rows under
`PYTHONHASHSEED` 1 and 999 (ledger) and default/7 (trace export).

`SHA256SUMS` covers every tracked packet file except `SHA256SUMS` itself.

Reproduce (from a worktree of `exscr2-guardpair-exec` with `uv sync --frozen` on
Python 3.13.14, this packet's `inputs/` copied to the same path in that worktree):

```bash
EXSEL_WT=<execution-worktree> tools/run.sh \
  docs/context/trace_example_selection/exsel-7.5-guard-pair/inputs/exsel75_screen_s1188.yaml \
  <out> exsel75_screen_s1188
python tools/screen.py <execution-worktree> <runs-dir> <ledger-dir>   # whole screening, resumable
```

## Preservation

Raw outputs (parity, screening, trace rerun, reconstruction, logs, checks) are in the
W&B artifact `wandb://ll7/robot_sf/campaign-exscr2_exsel75_guard_pair_20261029:v0`
(digest `0c308f63d3afea599d66572f09c6ef12`; the remote entry table matches 314/314
files). A private host archive holds a copy with a SHA-256 manifest
(`c178b903…6663d283c`) ([provenance/preservation.json](provenance/preservation.json)).

## Limits

- The trace is one seed. It illustrates the 27/30 pattern; it does not estimate a
  rate. Captions quote the base rates above, never a rate read off the trace.
- The guard intervention removes the named collision, clearance and braking rejection
  bundle together. The trace does not identify which part of it produced the
  completion.
- Development seed band 1161–1200 only. The execution source is an experimental
  modification of release `373dbfde`, not the unmodified release.

Provenance: execution commit `011c24045d6703a4a1e8b4428d6a9e931d220310`
(branch `exscr2-guardpair-exec`, tree equal to `0035fba05c9b`); contract commit
`689959b7a4123dc5ad9c4c89a9d1f003d61f48fa`; runtime patch
[provenance/acquisition-source.patch.gz](provenance/acquisition-source.patch.gz).
