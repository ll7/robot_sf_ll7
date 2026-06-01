# Issue #1953 Intersection-Wait Speed-Grid Trace

Issue: [#1953](https://github.com/ll7/robot_sf_ll7/issues/1953)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Depends on: [PR #1952](https://github.com/ll7/robot_sf_ll7/pull/1952) (merged)
Predecessor: [#1951](issue_1951_intersection_wait_phase_grid.md)

## Goal

Inspect the trace-level mechanism behind the strongest #1951
`francis2023_intersection_wait` speed-grid response:
`francis2023_intersection_wait_speed_h1_p050`, a
`single_pedestrian_speed_offset` with `speed_delta_m_s: +0.5`.

This note is diagnostic local evidence only. It is not benchmark-strength or paper-facing evidence.

## Scope

The fixed boundary matches #1951:

- scenario: `francis2023_intersection_wait`;
- target variant: `francis2023_intersection_wait_speed_h1_p050`;
- family: `single_pedestrian_speed_offset`;
- planners: `goal`, `orca`, `scenario_adaptive_hybrid_orca_v2_collision_guard`;
- seeds: `[240, 241, 242]` via `--seed-limit 4`;
- horizon: `80`, dt: `0.1`, closest-approach slice window: `3`.

## Runner Boundary

The literal issue command was run against
`configs/scenarios/perturbations/issue_1610_intersection_wait_phase_grid_v1.yaml`. The current
trace runner selects the first matching variant for a family, so that run produced the
`francis2023_intersection_wait_speed_h1_m025` row. To inspect the requested `+0.5 m/s` row without
changing code, schema, tests, or tracked configs, I generated an ignored manifest under `output/`
filtered from the tracked #1951 manifest to only:

- `francis2023_intersection_wait_noop`;
- `francis2023_intersection_wait_speed_h1_p050`.

The tracked `p050` evidence therefore comes from the same runner and same #1951 manifest content,
but through this ignored narrowed manifest. Treat the broad-run `m025` file as runner-behavior
evidence, not as the #1953 mechanism conclusion.

## Commands

Literal family command, which selected the first speed variant (`m025`):

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python \
  scripts/validation/run_scenario_perturbation_trace_response.py \
  configs/scenarios/perturbations/issue_1610_intersection_wait_phase_grid_v1.yaml \
  --materialized-output-dir \
  output/scenario_perturbations/issue1953_intersection_wait_speed_grid_trace/materialized \
  --output \
  docs/context/evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/closest_approach_trace_slices.json \
  --markdown-output \
  docs/context/evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/report.md \
  --source-scenario-id francis2023_intersection_wait \
  --perturbed-family single_pedestrian_speed_offset \
  --seed-limit 4 --horizon 80 --dt 0.1 --slice-window 3 \
  --planner goal --planner orca \
  --planner scenario_adaptive_hybrid_orca_v2_collision_guard
```

Targeted `p050` command, using the ignored filtered manifest:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python \
  scripts/validation/run_scenario_perturbation_trace_response.py \
  output/scenario_perturbations/issue1953_intersection_wait_speed_grid_trace/target_manifest_speed_h1_p050.yaml \
  --materialized-output-dir \
  output/scenario_perturbations/issue1953_intersection_wait_speed_grid_trace/materialized_speed_h1_p050 \
  --output \
  docs/context/evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/closest_approach_trace_slices_speed_h1_p050.json \
  --markdown-output \
  docs/context/evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/report_speed_h1_p050.md \
  --source-scenario-id francis2023_intersection_wait \
  --perturbed-family single_pedestrian_speed_offset \
  --seed-limit 4 --horizon 80 --dt 0.1 --slice-window 3 \
  --planner goal --planner orca \
  --planner scenario_adaptive_hybrid_orca_v2_collision_guard
```

Both trace runs emitted the known `uni_campus_big.svg` invalid obstacle warning during combined
scenario loading. No code change was needed.

## Aggregate Result

All 9 `p050` no-op-versus-perturbed trace pairs completed. Every paired rollout ended with
`max_steps` on both sides, so this trace did not introduce a success, collision, timeout, or
termination-reason delta.

Mean closest-approach deltas over completed `p050` pairs:

| Planner | Pairs | Clearance Delta | Time Delta | Progress Delta |
|---|---:|---:|---:|---:|
| `goal` | 3 | `-4.530414 m` | `0.0 s` | `0.0 m` |
| `orca` | 3 | `-3.464482 m` | `-0.2 s` | `+0.399855 m` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 3 | `-3.592846 m` | `-0.233333 s` | `+0.46649 m` |
| all | 9 | `-3.862581 m` | `-0.144444 s` | `+0.288782 m` |

The closest-pedestrian index stayed `0` for every no-op and `p050` closest frame. The trace records
indices rather than pedestrian IDs, but this is consistent with the targeted `h1` perturbation row
and argues against a nearest-pedestrian identity switch.

## Per-Seed Closest Frames

Positions are `[x, y]`. `term` is `noop -> p050`.

| Planner | Seed | Term | No-op closest | `p050` closest | Delta |
|---|---:|---|---|---|---|
| `goal` | 240 | `max_steps -> max_steps` | `t=8.0`, ped `0`, robot `[12.314855, 13.840521]`, ped `[14.493091, 20.909155]`, clearance `5.996641`, progress `-2.708502` | `t=8.0`, ped `0`, robot `[12.314855, 13.840521]`, ped `[14.116768, 16.003982]`, clearance `1.415573`, progress `-2.708502` | clearance `-4.581068`, time `0.0`, progress `0.0` |
| `goal` | 241 | `max_steps -> max_steps` | `t=8.0`, ped `0`, robot `[12.103211, 13.798736]`, ped `[14.493091, 20.909155]`, clearance `6.101305`, progress `-2.299511` | `t=8.0`, ped `0`, robot `[12.103211, 13.798736]`, ped `[14.111735, 16.002385]`, clearance `1.58165`, progress `-2.299511` | clearance `-4.519655`, time `0.0`, progress `0.0` |
| `goal` | 242 | `max_steps -> max_steps` | `t=8.0`, ped `0`, robot `[12.267694, 14.146749]`, ped `[14.493091, 20.909155]`, clearance `5.719166`, progress `-2.618941` | `t=8.0`, ped `0`, robot `[12.267694, 14.146749]`, ped `[14.125976, 16.005935]`, clearance `1.228647`, progress `-2.618941` | clearance `-4.490519`, time `0.0`, progress `0.0` |
| `orca` | 240 | `max_steps -> max_steps` | `t=6.4`, ped `0`, robot `[17.109513, 13.904221]`, ped `[14.480614, 21.89268]`, clearance `7.00991`, progress `-7.501221` | `t=6.2`, ped `0`, robot `[16.709539, 13.899653]`, ped `[14.257133, 18.20601]`, clearance `3.555705`, progress `-7.101312` | clearance `-3.454205`, time `-0.2`, progress `+0.399909` |
| `orca` | 241 | `max_steps -> max_steps` | `t=6.5`, ped `0`, robot `[17.122186, 13.882936]`, ped `[14.482421, 21.829694]`, clearance `6.973729`, progress `-7.315195` | `t=6.3`, ped `0`, robot `[16.722225, 13.877344]`, ped `[14.244857, 18.087695]`, clearance `3.485121`, progress `-6.915331` | clearance `-3.488608`, time `-0.2`, progress `+0.399864` |
| `orca` | 242 | `max_steps -> max_steps` | `t=6.3`, ped `0`, robot `[16.867301, 14.145737]`, ped `[14.478652, 21.955862]`, clearance `6.767233`, progress `-7.21758` | `t=6.1`, ped `0`, robot `[16.467358, 14.152489]`, ped `[14.267439, 18.32462]`, clearance `3.3166`, progress `-6.817788` | clearance `-3.450633`, time `-0.2`, progress `+0.399792` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 240 | `max_steps -> max_steps` | `t=6.7`, ped `0`, robot `[17.045011, 13.893816]`, ped `[14.485585, 21.704319]`, clearance `6.819162`, progress `-7.436827` | `t=6.4`, ped `0`, robot `[16.445064, 13.885888]`, ped `[14.230038, 17.969871]`, clearance `3.245994`, progress `-6.837008` | clearance `-3.573168`, time `-0.3`, progress `+0.599819` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 241 | `max_steps -> max_steps` | `t=6.8`, ped `0`, robot `[17.038323, 13.868987]`, ped `[14.486947, 21.641932]`, clearance `6.780965`, progress `-7.23151` | `t=6.5`, ped `0`, robot `[16.438405, 13.859082]`, ped `[14.211885, 17.852852]`, clearance `3.172482`, progress `-6.631788` | clearance `-3.608483`, time `-0.3`, progress `+0.599722` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 242 | `max_steps -> max_steps` | `t=6.5`, ped `0`, robot `[16.625772, 13.89374]`, ped `[14.482421, 21.829694]`, clearance `6.820299`, progress `-6.975561` | `t=6.4`, ped `0`, robot `[16.425911, 13.901202]`, ped `[14.230038, 17.969871]`, clearance `3.223411`, progress `-6.775632` | clearance `-3.596888`, time `-0.1`, progress `+0.199929` |

## Interpretation

The larger negative clearance response appears primarily to be a smooth speed-magnitude/phase
effect for the same nearest pedestrian, not a nearest-pedestrian identity switch. In the `goal`
planner rows, the robot pose and progress are identical at closest approach while the faster
pedestrian is much farther along its path and closer to the robot. ORCA and the hybrid guard show a
small planner-response component: closest approach happens `0.1-0.3 s` earlier and the robot is
about `0.2-0.6 m` different in progress, but the same pedestrian-index and same signed clearance
shift remain across all seeds.

I would not call this a planner-specific route-progress artifact. Route-progress deltas are present
for ORCA and the hybrid guard, absent for `goal`, and smaller than the pedestrian-phase clearance
change. Confidence in this mechanism read is about 0.8 because the trace runner stores pedestrian
indices rather than stable pedestrian IDs and this is only a 3-seed, 80-step local diagnostic.

## Evidence Boundary

Tracked compact evidence:

- [closest_approach_trace_slices_speed_h1_p050.json](evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/closest_approach_trace_slices_speed_h1_p050.json)
- [focus_speed_h1_p050_summary.json](evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/focus_speed_h1_p050_summary.json)
- [report_speed_h1_p050.md](evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/report_speed_h1_p050.md)
- [README.md](evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/README.md)
- [SHA256SUMS](evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/SHA256SUMS)

Also tracked for transparency:

- [closest_approach_trace_slices.json](evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/closest_approach_trace_slices.json)
- [report.md](evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/report.md)

Those two files are the literal-family run and correspond to `speed_h1_m025`, not the #1953 target
row.

Ignored local outputs:

- generated manifest:
  `output/scenario_perturbations/issue1953_intersection_wait_speed_grid_trace/target_manifest_speed_h1_p050.yaml`;
- materialized scenario matrices under
  `output/scenario_perturbations/issue1953_intersection_wait_speed_grid_trace/materialized*/`.

## Routing

Any downstream controller or search-policy change should be a separate issue with its own executable
proof path.
