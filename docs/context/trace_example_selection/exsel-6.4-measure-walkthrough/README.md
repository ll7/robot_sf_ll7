# EXSEL-6.4-MEASURE-WALKTHROUGH: screening result

**Result: ACCEPTED seed 1181 at screening index 4 of 40.** Trace rerun matches the
screening row on every match field; trace accepted.

- **Question.** Which development seed is the protocol's walkthrough episode: the first
  seed in the committed order on which social_force, in `classic_doorway_high`, succeeds
  with at least one exposure step and lies inside all three frozen bands?
- **Protocol.** ll7/diss `docs/context/trace_example_selection_protocol.yaml`, claim
  EXSEL-6.4-MEASURE-WALKTHROUGH (decision EXSEL-TRACE-SELECTION-PROTOCOL-2026-10-28). The
  claim block, rules and acquisition spec are copied verbatim into `contract.yaml`
  (commit 3e60a0924303eaa639b1871fb9bd71743306a80e, pushed before the first screening
  episode).
- **Source commit.** Unmodified 0.0.8 release code at
  373dbfde4f39667cf9e8732dabe7df5118bdeab1 (branch `exscr1-exec-373dbfde`), release
  campaign config, release scenario matrix, release social-force planner entry, authored
  horizon 500 steps, dt 0.1 s. Tier: development-seed diagnostic, new current-version
  execution; not release evidence.
- **Base rate the caption must quote.** Frozen pattern **15/30** (social_force successes
  with at least one exposure step among the 30 release episodes of
  `classic_doorway_high`; the claim-keyed report prints the same 15/30 unconditional
  rate). Never a rate read off this trace.

## Screening (all screened seeds; `screening_ledger.csv`)

| index | seed | outcome | exposure steps | min clearance (m) | time to goal (s) | verdict |
|---|---|---|---|---|---|---|
| 1 | 1197 | S | 13 | 0.3908 | 30.6 | reject: time to goal below band |
| 2 | 1199 | S | 24 | 0.3077 | 45.5 | reject: all three bands missed |
| 3 | 1194 | S | 0 | 0.6209 | 31.2 | reject: class (no exposure step) and all three bands |
| 4 | 1181 | S | 8 | 0.4306 | 34.7 | **accepted** |

Frozen bands: clearance [0.353, 0.459] m, exposure steps [8.0, 13.5], time to goal
[33.25, 39.2] s, all inclusive. Seed 1181's exposure count, 8, lies exactly on the lower
bound; the band is inclusive, so it is inside. Rows are `mode = native`; see
`../provenance/fallback_audit.md`.

## Trace (`trace/`)

- `social_force_seed1181_step_trace.json.gz`: the rerun row's
  `algorithm_metadata.simulation_step_trace` (schema simulation-step-trace.v2, 347 steps,
  dt 0.1) with row identity.
- `trace_check.json`: the rerun matches the screening row on outcome, collision kind,
  termination reason, steps, the three collision counts, near_misses and minimum
  pedestrian separation. Trace versus row: 347 = 347 steps, trace minimum pedestrian
  surface clearance equals the row value (difference below 1e-9 m), 8 trace steps
  with clearance below the 0.5 m near-miss distance against row near_misses 8.

## Comparison contract (reconstruction)

A rerun at the execution commit must match a screening row exactly on every serialized
field except `timestamps.start`, `timestamps.end`, `timing.steps_per_second` and
`wall_time_sec`; file paths, campaign ids and wall time of the surrounding campaign
outputs also differ. Checked with a 1-seed rerun (seed 1197): 813 flattened fields, only
those four differ (`../provenance/reconstruction.json`). `config_hash` is per episode,
so it differs across seeds but must match for the same seed.

## Rerun

```
cd <robot_sf_ll7 worktree at 373dbfde4f39667cf9e8732dabe7df5118bdeab1> && uv sync --frozen
python <packet>/tools/screen.py --contract <packet>/exsel-6.4-measure-walkthrough/contract.yaml \
    --wt <worktree> --runs <out>
python <packet>/tools/screen.py ... --trace-rerun
python <packet>/tools/build_ledger.py --contract ... --runs <out> --out screening_ledger.csv
python <packet>/tools/extract_trace.py --contract ... --runs <out> --out-dir trace
```
