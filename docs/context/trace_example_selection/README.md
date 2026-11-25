# Trace-example screening (exsel protocol), robot_sf side

Screening runs for two claims of the dissertation's frozen trace-example selection
protocol (ll7/diss `docs/context/trace_example_selection_protocol.yaml`, decision
EXSEL-TRACE-SELECTION-PROTOCOL-2026-10-28). Development-seed diagnostic on the
unmodified 0.0.8 release code (new current-version execution); never release evidence.
This packet admits nothing into the thesis. Whether a figure is made is decided on the
dissertation side.

| claim | cell, arms | result | base rate for captions |
|---|---|---|---|
| EXSEL-6.4-MEASURE-WALKTHROUGH | classic_doorway_high; social_force | ACCEPTED seed 1181 at screening index 4 of 40 | 15/30 |
| EXSEL-7.2-NARROW-DOORWAY | francis2023_narrow_doorway; orca, goal | ACCEPTED seed 1179 at screening index 4 of 40 | 30/30 |

Both trace-capable reruns match their screening rows on every match field.

## Layout

- `<claim>/contract.yaml.txt`: the claim block, rules and acquisition spec copied verbatim
  from the protocol, plus the execution binding, screening order, measure definitions,
  trace-match fields, mode rule and outcome rule. The original file was committed and
  pushed in 3e60a0924303eaa639b1871fb9bd71743306a80e before the first screening episode
  (`provenance/execution.json`).
  The suffix preserves the YAML bytes while separating the copied dissertation
  references from repo-local path checks. The execution record uses the original
  filename, `contract.yaml`.
- `<claim>/screening_order.txt`: the committed order with each seed's SHA-256 key.
- `<claim>/screening_ledger.csv`: one row per screened (seed, arm), rejected ones
  included, with the acquisition-spec step 2 fields, the `mode` column and the verdict
  per class predicate and per band. `screening_summary.json` gives the result.
- `<claim>/trace/`: the accepted seed's step traces (gzip, one per arm) and
  `trace_check.json` (rerun versus screening row; trace versus its own row).
- `<claim>/SHA256SUMS`: digests of that claim's files.
- `tools/`: config derivation (`make_config.py`), single-run wrapper (`run_one.sh`),
  screening driver (`screen.py`), acceptance logic (`exsel.py`), contract writer, ledger
  builder, trace extractor, row comparator, W&B upload and tests (`test_exsel.py`).
- `provenance/`: execution identity and timeline, reconstruction check, preservation
  record, fallback and mode audit.

## Execution route

Each screened seed is one camera-ready campaign run by
`scripts/tools/run_camera_ready_benchmark.py` at 373dbfde4f39667cf9e8732dabe7df5118bdeab1.
The config is the release campaign config with only these keys changed:
`seed_policy` (fixed list holding the one screened seed), `scenario_candidates`
(the frozen cell), `planners` (the claim's release planner entries, verbatim),
`export_publication_bundle` (off), `workers` (1), and `snqi_v2_spec` (off). The release
bound its SNQI-v2 anchors for its 48-cell schedule, and the loader refuses a one-cell
schedule. SNQI enters no class predicate or banded measure. The trace rerun also sets
`record_simulation_step_trace`. No robot_sf code was changed and no runtime patch was
applied.

Qualification before screening: goal in `classic_doorway_medium` on development seeds
1004 and 1005 reproduced an earlier independent re-execution at the same commit (other
runner entry point). Steps, termination, collision count, near_misses and minimum
pedestrian separation were equal to full float precision. A runner smoke ran on
development seed 1001. Neither is a screening row.

## Comparison contract

A rerun at the execution commit must reproduce every serialized episode-row field
exactly, except `timestamps.start`, `timestamps.end`, `timing.steps_per_second` and
`wall_time_sec`. Campaign ids, output paths and surrounding campaign timing also differ.
A 1-seed rerun per claim, under a different Python hash seed, met this contract
(`provenance/reconstruction.json`). All derived files (ledgers, summaries, traces,
contracts) regenerate byte-identically under PYTHONHASHSEED 0 and 1. The trace-rerun
match fields are listed in each contract (`trace_rerun.match_fields`).

The root `SHA256SUMS` covers every tracked file of this packet—its two claim
directories, root files, `tools/` and `provenance/`—except itself. Each claim
manifest covers every tracked file in its claim directory except itself.

## Rerun everything

```
git worktree add <wt> 373dbfde4f39667cf9e8732dabe7df5118bdeab1 && (cd <wt> && uv sync --frozen)
for c in exsel-6.4-measure-walkthrough exsel-7.2-narrow-doorway; do
  <wt>/.venv/bin/python tools/screen.py --contract $c/contract.yaml.txt --wt <wt> --runs <out>/$c
  <wt>/.venv/bin/python tools/screen.py --contract $c/contract.yaml.txt --wt <wt> --runs <out>/$c --trace-rerun
  <wt>/.venv/bin/python tools/build_ledger.py --contract $c/contract.yaml.txt --runs <out>/$c --out $c/screening_ledger.csv
  <wt>/.venv/bin/python tools/extract_trace.py --contract $c/contract.yaml.txt --runs <out>/$c --out-dir $c/trace
done
```

Environment: Python 3.13.14, `uv.lock` SHA-256
def82098b23281e7c49f1f05e052e412c2de54ae2da53f1708bc0ddc2a30d023, rvo2 importable,
single-threaded numerical libraries. Measured cost: about 10-12 s per screened seed
(one campaign).

## Preservation

Raw outputs (smoke, qualification, screening, trace reruns and reconstruction reruns;
762 files plus manifest):
`wandb://ll7/robot_sf/campaign-exsel_trace_example_screening_373dbfde4:v0`
(artifact digest cc2828ac799fee5aeabaebc7afc06844, COMMITTED). A lab-host archive has the
same files under SHA-256 manifest 4eaf89f5…613f (`provenance/preservation.json`).

## Executed tool versions

The screening and trace reruns used the tools committed in 3e60a092. Later changes
linted and formatted `exsel.py`, `screen.py` and `write_contract.py` and changed one
writer assertion. At the reviewed head, the derived files regenerated byte-identically
with those tools. This repair updates file references in documentation and tests.
