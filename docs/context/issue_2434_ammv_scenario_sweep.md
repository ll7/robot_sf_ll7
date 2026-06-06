# Issue #2434 AMMV Scenario Sweep Decision

Date: 2026-06-06

Related issues: <https://github.com/ll7/robot_sf_ll7/issues/2434>,
<https://github.com/ll7/robot_sf_ll7/issues/2159>,
<https://github.com/ll7/robot_sf_ll7/issues/2227>,
<https://github.com/ll7/robot_sf_ll7/issues/2428>,
<https://github.com/ll7/robot_sf_ll7/issues/2430>,
<https://github.com/ll7/robot_sf_ll7/issues/2432>

Status: diagnostic-only multi-scenario AMMV/default frame-level and episode-metric screen; not
benchmark-strength or paper-facing evidence.

## Goal

Issue #2432 showed that the local Issue #2168 `classic_head_on_corridor_low` seed slice remained
frame-identical between default and AMMV-aware Social Force. Issue #2434 checked the next smallest
question: whether a compact set of classic close-interaction scenario families exposes any
non-identical AMMV/default pair worth promoting into behavioral-difference trace work.

## Local Sweep

The sweep added `configs/scenarios/issue_2434_ammv_trace_selection.yaml`, selecting five existing
classic scenario IDs:

- `classic_bottleneck_medium`
- `classic_cross_trap_low`
- `classic_doorway_high`
- `classic_head_on_corridor_medium`
- `classic_overtaking_medium`

Both default and AMMV-aware Social Force runs used horizon `120`, `dt=0.1`, `--workers 1`,
`--record-forces`, `--record-simulation-step-trace`, no video, and JSON structured output. The AMMV
arm used `configs/baselines/social_force_ammv_aware.yaml`. Raw regenerated JSONL outputs remain
under `output/issue_2434/` and are not tracked; the evidence bundle records their checksums.

## Result

The two arms each wrote 15 rows: three seeds for each selected scenario family. All matched
scenario/seed pairs recorded 120 frames and were identical across robot state, pedestrian state,
selected action, planner event, `ammv.pedestrian_force_vectors`, status, outcome, steps, success,
collisions, clearance, speed, force metrics, and every numeric metric present in both JSONL rows.

| Scenario | Seeds | Pairs | Max per-frame delta | Max episode-metric delta | Non-identical pairs |
| --- | --- | ---: | ---: | ---: |
| `classic_bottleneck_medium` | `131, 132, 133` | 3 | 0.0 | 0.0 | 0 |
| `classic_cross_trap_low` | `101, 102, 103` | 3 | 0.0 | 0.0 | 0 |
| `classic_doorway_high` | `141, 142, 143` | 3 | 0.0 | 0.0 | 0 |
| `classic_head_on_corridor_medium` | `111, 112, 113` | 3 | 0.0 | 0.0 | 0 |
| `classic_overtaking_medium` | `121, 122, 123` | 3 | 0.0 | 0.0 | 0 |

The maximum per-frame absolute delta and maximum episode-metric absolute delta across all 15 pairs
were both `0.0`. Wall-clock runtime differed slightly between arms, with maximum absolute wall-time
delta `0.4253122806549072` seconds, but that is not interpreted as a behavioral metric.

## Decision

No non-identical default-vs-AMMV frame-level or episode-metric pair was found in this compact
classic-family benchmark-adapter slice. The #2159/#2227 AMMV trace-panel lane should not spend more
annotation or panel effort on this adapter-mode scenario selection if the goal is
behavioral-difference evidence.

The next smallest useful proof step is one of:

- add or expose direct AMMV mechanism activation fields before benchmark-adapter action selection;
- build a direct planner-mode mechanism trace source from the controlled AMMV probe;
- test a deliberately more sensitive, higher-uncertainty scenario family and keep it labeled as
  diagnostic until a non-identical trace pair is proven.

This result should be read as a scoped negative screen: no detectable frame-level or episode-level
difference under this compact matrix/config, not a general claim that AMMV-aware Social Force is
equivalent or ineffective.

## Evidence

Tracked compact evidence:

- `docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/summary.json`
- `docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/candidate_pair_comparison.csv`

The comparison covers nested `algorithm_metadata.simulation_step_trace` robot, pedestrian,
selected-action, event, and AMMV force-vector fields from the ignored local JSONL rows. The raw JSONL
outputs remain worktree-local; the tracked evidence preserves compact deltas, checksums, and the
candidate table.

## Validation

```bash
scripts/dev/run_worktree_shared_venv.sh -- robot_sf_bench validate-config --matrix configs/scenarios/issue_2434_ammv_trace_selection.yaml
scripts/dev/run_worktree_shared_venv.sh -- robot_sf_bench preview-scenarios --matrix configs/scenarios/issue_2434_ammv_trace_selection.yaml
scripts/dev/run_worktree_shared_venv.sh -- robot_sf_bench run --matrix configs/scenarios/issue_2434_ammv_trace_selection.yaml --out output/issue_2434/default_social_force_scenarios_seed111_h120.jsonl --base-seed 111 --repeats 1 --horizon 120 --dt 0.1 --record-forces --record-simulation-step-trace --no-video --video-renderer none --algo social_force --workers 1 --no-resume --structured-output json
scripts/dev/run_worktree_shared_venv.sh -- robot_sf_bench run --matrix configs/scenarios/issue_2434_ammv_trace_selection.yaml --out output/issue_2434/ammv_social_force_scenarios_seed111_h120.jsonl --base-seed 111 --repeats 1 --horizon 120 --dt 0.1 --record-forces --record-simulation-step-trace --no-video --video-renderer none --algo social_force --algo-config configs/baselines/social_force_ammv_aware.yaml --workers 1 --no-resume --structured-output json
python -m json.tool docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/summary.json
scripts/validation/check_docs_proof_consistency.py --path docs/context/issue_2434_ammv_scenario_sweep.md --path docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/README.md --path docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/summary.json
git diff --check
```
