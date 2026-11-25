# Guard with crossing pedestrians: robot outcome and pedestrian passage

**Diagnostic evidence, development seeds 1121–1130 and 1161–1200.** This is a
new current-version execution of the 0.0.8 release code plus a rebuilt
diagnostic guard switch. It is not release evidence and changes no thesis text
or frozen release artifact. Its admission is conditional on an independent
exact-head MERGE review.

**Question.** Does the continuous hybrid v4 hard-guard bundle still increase
collision-free completion when pedestrians pass the same restriction as the
robot, and how does it affect their passage?

## Provenance and design

- Release base: `373dbfde4f39667cf9e8732dabe7df5118bdeab1`. Runtime switch
  reconstruction commit: `a1c07ce529e1cff3c860fa571b8d26a68ff4a798`.
  Acquisition and preregistration commit:
  [`d0e2aa1d9363c7fe6e1eb62e225c4721e7fa7271`](https://github.com/ll7/robot_sf_ll7/commit/d0e2aa1d9363c7fe6e1eb62e225c4721e7fa7271).
  The retained remote branch `xped-guard-crossing` contains this commit; the
  packet-only PR is based on current main. Its exact packet commit is the PR
  head, also obtainable with `git rev-parse HEAD` in that checkout.
- [`provenance/diagnostic-guard-switch.patch`](provenance/diagnostic-guard-switch.patch)
  is the zero-context runtime-file diff from the release base to the acquisition
  commit. Apply it to that exact base with `git apply --unidiff-zero` and verify
  the resulting planner blob; it is not a runtime change in the packet PR. The original preregistered contract
  is preserved byte-for-byte in
  [`provenance/preregistered-contract.yaml.gz`](provenance/preregistered-contract.yaml.gz)
  and at the acquisition commit. [`contract.yaml`](contract.yaml) is a corrected
  transcription: its scientific design and outcome rule are unchanged, while
  the wrapper timing and commit-attribution descriptions are corrected.
- All 1,200 robot rows record the acquisition commit. The 400 no-robot
  trajectories have no embedded commit identifier; the archive's environment
  manifest and acquisition logs document their run context, but do not provide
  per-row commit attestation. All 1,200 robot/no-robot pairs match on initial
  pedestrian positions and simulator-configuration digest. The archived
  environment field named `main_rows_git_hash` should be read as robot rows
  only.
- `G1E1` is hybrid v4, guard and escape on; `G0E1` disables the entire hard
  guard with escape on; `orca` is the release reference. No-robot controls use
  the same scenario and seed. Four authored doorway/bottleneck cells each run
  with traverse and outside routes. Robot arms comprise 1,200 episodes; there
  are 400 no-robot runs. The pilot is separate and never pooled.
- Pedestrian trajectories are observed for 50 s. After robot termination, a
  successful robot is removed; a failed robot remains stopped. The wrapper
  continues the simulator after the robot step loop and **before** the runner
  assembles its episode record. The listed robot-field invariance below is
  measured, not guaranteed by an earlier record write.
- All 1,200 robot rows report mode `adapter`. The 800 hybrid rows have recorded
  `fallback_count = 0` and no fallback/degraded marker. The 400 ORCA rows have
  no fallback/degraded marker, but **lack** the `fallback_count` field; this is
  not a recorded zero. The historical `fallback_or_degraded_rows` summary counts
  a blank counter as clean, so it must be interpreted with this limitation.

## Measures and weighting

Passage is a designed-direction crossing of a gate in the restriction. For
each cell/seed, no-robot passers are identified by pedestrian index. The
cell/seed delay is the mean of their first-passage delays, treating a missing
passage as 50 s; the cell/seed blocked share is the fraction of those passers
who do not pass with the robot. Censoring makes delay a lower bound. Pooling
first averages these cell/seed means or fractions over the four cells for each
seed, then averages the seed means: **equal cell and seed weight**, not equal
weight per individual passer. Confidence intervals resample the 50 whole seed
blocks (10,000 bootstrap draws), preserving pairing. These quantities exclude
pedestrians who do not pass within 50 s in the paired no-robot run.

## Result

| Traverse, equal cell and seed weight | G1E1 guard on | G0E1 guard off | ORCA |
| --- | ---: | ---: | ---: |
| Collision-free completion, % | 100.0 | 45.0 [40.5, 49.5] | 90.0 [86.0, 93.5] |
| Wall contact, % | 0.0 | 48.5 [45.0, 52.0] | 1.0 [0.0, 2.5] |
| Pedestrian contact, % | 0.0 | 6.5 [3.5, 10.0] | 7.0 [4.0, 10.0] |
| Cell/seed blocked-share mean, % | 18.1 [15.6, 20.6] | 36.4 [31.5, 41.2] | 17.4 [14.8, 20.0] |
| Cell/seed censored-delay mean, s | 7.6 [6.9, 8.3] | 9.8 [8.8, 10.8] | 5.8 [5.3, 6.2] |

- Guard effect on completion, traverse: **+55.0 pp [50.5, 59.5]**;
  outside: +21.0 pp [14.0, 28.0]; interaction: +34.0 pp [26.0, 42.5].
- Main parking rule: guard minus no guard gives −2.2 s [−2.9, −1.5] in
  the equally weighted cell/seed delay means and −18.3 pp [−22.9, −13.6]
  in the equally weighted cell/seed blocked shares. The preregistered rule
  returns O1; its “preserved” label is relative to guard-off and does not imply
  zero absolute cost.
- Against ORCA, the guarded arm's weighted blocked share is similar, but its
  weighted cell/seed mean delay is 1.8 s greater [1.1, 2.6].
- **Post-hoc sensitivity:** removing failed robots instead changes the guard
  contrast to +2.3 s [1.5, 3.1] delay and +12.5 pp [9.8, 15.1] blocked share,
  under the same cell/seed weighting; this would meet O2. Among the 400 rerun
  robot episodes, 400/400 match the main run on steps, termination, success,
  pedestrian and wall contacts, timeout, and closest robot-pedestrian distance.
  Complete projected rows differ in execution commit and source hashes.
  The policy-specific effects are estimable; this does not isolate a
  parking-independent contemporaneous guard effect. See
  [`evidence/sensitivity_remove_on_failure/`](evidence/sensitivity_remove_on_failure/).
- Without a robot, 366/750 doorway pedestrians and 200/200 bottleneck
  pedestrians pass within 50 s. The 384 non-passing doorway pedestrians are
  excluded from the passage-cost denominator. Pairing does not remove the
  fixture's crowd jamming. Results apply to these cells and development seeds,
  the whole guard bundle with escape on, and the stated parking rules.

The full tables are [`evidence/rates.csv`](evidence/rates.csv),
[`evidence/paired_effects.csv`](evidence/paired_effects.csv), and
[`evidence/summary.json`](evidence/summary.json); the compact episode rows are
[`evidence/episodes.csv.gz`](evidence/episodes.csv.gz).

## Reproduction and comparison contract

From the repository root, re-run the deterministic analysis and verify packet
checksums (the manifest paths are relative to the packet directory):

```bash
python docs/context/guard_crossing_pedestrians/analyze.py analyze \
  --episodes docs/context/guard_crossing_pedestrians/evidence/episodes.csv.gz --out /tmp/xped-check
(cd docs/context/guard_crossing_pedestrians && sha256sum -c SHA256SUMS)
```

For a one-seed acquisition replay, use a separate worktree at the acquisition
commit, Python 3.13.14, and `uv sync --frozen --all-extras` with that commit's
lockfile. From that worktree, run:

```bash
docs/context/guard_crossing_pedestrians/tools/run_campaign.sh "$PWD/.venv/bin/python" /tmp/xped-one 1121 8
.venv/bin/python docs/context/guard_crossing_pedestrians/analyze.py project --raw /tmp/xped-one --out /tmp/xped-one-projected
```

Run the following comparator **from the packet checkout**, supplying its
archived projection and the replay projection:

```bash
python docs/context/guard_crossing_pedestrians/tools/compare_replay.py \
  --archived docs/context/guard_crossing_pedestrians/evidence/episodes.csv.gz \
  --replay /tmp/xped-one-projected/episodes.csv.gz --seed 1121
```

The contract requires the complete 32 distinct `(arm, cell, condition, seed)`
keys for this one seed, the same CSV schema and **exact string equality for
all projected fields except `source_line_sha256`**. Thus mode, fallback
fields, guard flag, status, recorded commit, steps, termination, parking,
success, contacts, timeout, all passage/delay/throughput measures, closest
distance, and paired reset/config checks must match exactly. The source hashes
are retained for raw custody but may differ: robot raw JSON includes new
run timestamps, wall time, and checkout/output paths; no-robot hashes cover
trajectory files. These raw fields are not otherwise projected or silently
normalized. The comparator reports both normalized and full-row equality and
fails on missing, duplicate, unexpected, or differing required fields. The
[recorded one-seed receipt](provenance/replay-1121.json) reports 32/32 normalized
rows and 8/32 complete rows equal; its [replay projection](provenance/replay-1121-episodes.csv.gz)
retains all raw custody hashes. A replay is a deterministic comparison of the
projected contract, not byte-identical raw files or independent validation of
every raw field.

Raw logs, per-episode JSONL, and trajectory sidecars are preserved in the W&B
artifact and private host archive recorded in
[`provenance/preservation.json`](provenance/preservation.json), with a SHA-256
manifest. This packet contains compact projections and its own checksums.
