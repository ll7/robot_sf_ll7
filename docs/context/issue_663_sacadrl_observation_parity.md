# Issue 663 SACADRL Observation Parity Note

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#659` headless upstream reproduction
- `robot_sf_ll7#661` model-level parity for the GA3C-CADRL checkpoint
- `robot_sf_ll7#663` Robot SF observation parity for the SACADRL adapter

## Goal

Determine whether the current Robot SF `SACADRLPlannerAdapter` constructs the same GA3C-CADRL
network input that the upstream source expects when both are driven from comparable crowd states.

This issue isolates the observation-to-vector mapping. It does **not** claim that SACADRL is a
strong planner on the Robot SF benchmark.

## Canonical probe artifacts

- JSON report:
  `output/benchmarks/external/sacadrl_observation_parity/report.json`
- Markdown report:
  `output/benchmarks/external/sacadrl_observation_parity/report.md`

Generated with:

```bash
uv run python scripts/tools/probe_sacadrl_observation_parity.py \
  --repo-root output/repos/gym-collision-avoidance \
  --side-env-python output/benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python \
  --output-json output/benchmarks/external/sacadrl_observation_parity/report.json \
  --output-md output/benchmarks/external/sacadrl_observation_parity/report.md
```

## Method

The probe compares three cases:

1. `live_upstream_two_agents_reset`
   - capture a real native GA3C-CADRL observation from the upstream headless side environment
   - reconstruct a comparable Robot SF SocNav observation
   - run `SACADRLPlannerAdapter._build_network_input(...)`
   - compare the full flattened vector against the upstream-native state

2. `controlled_rotated_multi_agent`
   - construct a non-trivial upstream-style state with rotated heading, multiple agents, and
     non-zero velocities
   - reconstruct the comparable Robot SF observation
   - compare the full flattened vector component-by-component

3. `robot_sf_socnav_observation_fusion`
   - use the actual `robot_sf/sensor/socnav_observation.py` builder on a fake simulator snapshot
   - verify that pedestrian velocities are emitted in ego frame and reconstructed back into the
     correct goal-frame SACADRL features

## Current result

Verdict: `adapter observation mapping reproduced in controlled cases`

Observed probe artifact:

- `output/benchmarks/external/sacadrl_observation_parity/report.md`

Case summary:

- `live_upstream_two_agents_reset`
  - max abs diff: `0.00000000`
- `controlled_rotated_multi_agent`
  - max abs diff: `0.00000000`
- `robot_sf_socnav_observation_fusion`
  - max abs diff: `0.00000000`

## Interpretation

What this now supports:

- the current SACADRL adapter can reproduce the upstream GA3C-CADRL network input exactly on the
  tested live native reset and on a controlled rotated multi-agent case,
- the Robot SF SocNav observation builder emits pedestrian velocities in ego frame,
- the adapter's velocity rotation path is therefore consistent with the actual Robot SF observation
  contract.

What this does **not** support:

- that SACADRL is a strong planner on the Robot SF benchmark,
- that every possible upstream environment configuration is now covered,
- that benchmark weakness should be ignored.

Current judgment:

- the main remaining SACADRL question is planner quality / scenario transfer, not the
  observation-to-network-input mapping,
- current SACADRL benchmark numbers can be treated as CADRL-family evidence more confidently than
  before, subject to the usual caveat that benchmark competitiveness is still separate from source
  faithfulness.

## Recommendation

- keep `sacadrl` as benchmarkable CADRL-family evidence,
- do not spend more time re-litigating checkpoint loading or basic observation algebra unless a new
  failing parity case appears,
- if SACADRL remains weak on the benchmark, treat that as a planner-performance question and move
  effort toward stronger external candidates rather than more low-level adapter debugging.
