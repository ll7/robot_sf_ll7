# Issue #2432 AMMV Trace Selection Decision

Date: 2026-06-06

Related issues: <https://github.com/ll7/robot_sf_ll7/issues/2432>,
<https://github.com/ll7/robot_sf_ll7/issues/2159>,
<https://github.com/ll7/robot_sf_ll7/issues/2168>,
<https://github.com/ll7/robot_sf_ll7/issues/2227>,
<https://github.com/ll7/robot_sf_ll7/issues/2428>,
<https://github.com/ll7/robot_sf_ll7/issues/2430>

Status: diagnostic-only fail-closed trace-selection result; not benchmark-strength or
paper-facing evidence.

## Goal

Issue #2430 showed that the first promoted #2428 default/AMMV Social Force trace pair is
telemetry-rich but numerically identical over all recorded per-frame fields. Issue #2432 checked
whether the same local, regenerable Issue #2168 slice contains another seed pair that can support
AMMV behavioral-difference annotation.

## Local Sweep

The sweep regenerated the Issue #2168 `classic_head_on_corridor_low` matrix for seeds `111..113`
with step traces enabled for both planners:

- default Social Force;
- AMMV-aware Social Force via `configs/baselines/social_force_ammv_aware.yaml`.

Both commands used `--record-forces`, `--record-simulation-step-trace`, horizon `100`, `dt=0.1`,
`--workers 1`, and no video. Raw regenerated JSONL outputs remain under `output/issue_2432/` and
are not tracked; the evidence bundle records their checksums.

## Result

All three default/AMMV seed pairs are still frame-identical across the comparison fields needed for
trace-review selection:

| Seed | Frames per planner | Default status | AMMV status | Max per-frame delta | Behavioral delta found |
| --- | ---: | --- | --- | ---: | --- |
| `111` | 100 | failure | failure | 0.0 | no |
| `112` | 100 | failure | failure | 0.0 | no |
| `113` | 100 | failure | failure | 0.0 | no |

The comparison covered robot state, pedestrian state, planner event, selected action, and
`ammv.pedestrian_force_vectors`. The aggregate episode metrics also matched per seed, including
success status, collisions, minimum clearance, and average speed.

## Decision

No non-identical default-vs-AMMV trace pair was found in the local Issue #2168
`classic_head_on_corridor_low` adapter-mode seed slice. The #2159/#2227 AMMV lane should stop
spending annotation or panel effort on this slice if the goal is behavioral-difference evidence.

The next smallest useful proof step is one of:

- select a different scenario or scenario family where AMMV-aware Social Force can plausibly alter
  the benchmark-adapter rollout;
- build a direct planner-mode mechanism trace source from the controlled #2168 mechanism probe;
- add instrumentation that distinguishes default and AMMV terms before benchmark-adapter action
  selection, if the current adapter path intentionally collapses them.

Until one of those paths exists, keep this lane diagnostic-only and classify the current
head-on-corridor seed slice as a blocked behavioral-difference source.

## Evidence

Tracked compact evidence:

- `docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/summary.json`
- `docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/candidate_pair_comparison.csv`

The summary records raw-output checksums, commands, candidate rows, frame counts, and the
per-field maxima. Raw benchmark JSONL outputs are deliberately not tracked.

## Validation

```bash
scripts/dev/run_worktree_shared_venv.sh -- robot_sf_bench validate-config --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml
scripts/dev/run_worktree_shared_venv.sh -- robot_sf_bench preview-scenarios --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml
scripts/dev/run_worktree_shared_venv.sh -- robot_sf_bench run --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml --out output/issue_2432/default_social_force_seed111_3x_h100.jsonl --base-seed 111 --repeats 3 --horizon 100 --dt 0.1 --record-forces --record-simulation-step-trace --no-video --video-renderer none --algo social_force --workers 1 --no-resume --structured-output json
scripts/dev/run_worktree_shared_venv.sh -- robot_sf_bench run --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml --out output/issue_2432/ammv_social_force_seed111_3x_h100.jsonl --base-seed 111 --repeats 3 --horizon 100 --dt 0.1 --record-forces --record-simulation-step-trace --no-video --video-renderer none --algo social_force --algo-config configs/baselines/social_force_ammv_aware.yaml --workers 1 --no-resume --structured-output json
python -m json.tool docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/summary.json
scripts/validation/check_docs_proof_consistency.py --path docs/context/issue_2432_ammv_trace_selection.md --path docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/README.md --path docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/summary.json
git diff --check
```
