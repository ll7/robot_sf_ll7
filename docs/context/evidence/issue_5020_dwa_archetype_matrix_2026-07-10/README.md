<!-- AI-GENERATED (robot_sf#5020, 2026-07-10) - NEEDS-REVIEW -->
# Issue #5020 — DWA Across the Standard Archetype Matrix (Executed Evidence)

Date: 2026-07-10

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/5020>

Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/4983> (add a DWA classical baseline)

Implementation PR: <https://github.com/ll7/robot_sf_ll7/pull/5012> (made `algo=dwa` runnable)

This is the executed-evidence packet for #5020: it runs the classical Dynamic Window
Approach (DWA) planner across the **standard classic archetype matrix** with the canonical
config, fixed per-scenario seeds, and native (adapter) execution, then records the resulting
status rows. It is **smoke/nominal benchmark evidence only within the executed contract** — it
is **not** a comparative claim (no comparator planner was run here) and it does **not** promote
the DWA row to benchmark-strength success evidence.

## Contract

| Field | Value |
| --- | --- |
| Planner / algorithm | `dwa` (classical Dynamic Window Approach; unicycle `(v, omega)`) |
| Algorithm config | [`configs/algos/dwa_classic.yaml`](../../../../configs/algos/dwa_classic.yaml) |
| Algorithm config hash (recorded in rows) | `22f1993f86c57945` |
| Scenario matrix | [`configs/scenarios/classic_interactions.yaml`](../../../../configs/scenarios/classic_interactions.yaml) (the standard classic archetype matrix) |
| Matrix composition | 11 archetype configs → **23 graded scenario rows** (see [`classic_density_tier_index.yaml`](../../../../configs/scenarios/archetypes/classic_density_tier_index.yaml)) |
| Seeds | Fixed, per-scenario declared seeds (3 each), honored by the map runner. Distinct declared sets: `[101,102,103]`, `[111,112,113]`, `[121,122,123]`, `[131,132,133]`, `[141,142,143]`, `[151,152,153]`, `[161,162,163]`, `[171,172,173]`, `[201,202,203]`, `[301,302,303]`. |
| Execution mode | **native/adapter** — `benchmark_availability.availability_status = available`, `execution_mode = adapter`, `readiness_status = adapter`, `benchmark_success = true` |
| Observation | level `tracked_agents_no_noise`, mode `socnav_state`, no observation noise |
| Horizon / dt | `100` steps at `dt = 0.1` (run defaults) |
| Repo commit | `42ae06340ed35f62985209f1dbc5f2155642944a` |
| Total jobs | **69** (23 rows × 3 seeds); **written = 69**, **failed = 0** |

### Exact command

```bash
DISPLAY= SDL_VIDEODRIVER=dummy MPLBACKEND=Agg uv run robot_sf_bench --quiet run \
  --matrix configs/scenarios/classic_interactions.yaml \
  --out output/benchmarks/dwa_archetype_matrix_issue5020/episodes.jsonl \
  --algo dwa \
  --algo-config configs/algos/dwa_classic.yaml \
  --benchmark-profile experimental \
  --workers 4 --no-video --no-resume \
  --structured-output json
```

The archetype matrix declares its own fixed per-scenario seeds; the map runner honors the
scenario `seeds:` list first, so no `--base-seed`/`--repeats` override is needed to obtain the
canonical fixed seeds. The raw episode JSONL (69 rows) was written to worktree-local
`output/benchmarks/dwa_archetype_matrix_issue5020/episodes.jsonl` and is **not** a durable
dependency; this directory keeps the compact, reviewable status-row table derived from it.

## Result classification

**Smoke / nominal benchmark evidence only within the executed contract.**

- The matrix executed to completion with **zero simulation/runtime failures** (69/69 episodes
  written, exit code 0). Availability is `available` and execution is native/adapter — there
  were **no fallback or degraded exclusions** in this run.
- Behaviorally, DWA with this canonical config did **not** reach the goal on any of the 69
  episodes: `route_complete = 0/69`. Episodes terminated by `max_steps` timeout (56/69) or by
  `collision` (13/69). This is a genuine behavioral result for the canonical config on these
  archetypes (the reactive window gets stuck in local minima / oscillates toward the timeout,
  and collides on a minority of rows), **not** a crash or planner-unavailable condition.
- This packet makes **no comparative claim** (no comparator planner was executed here) and does
  not promote the DWA row past its existing `degraded` / `counts_as_success_evidence: false`
  readiness status. Per the readiness matrix, a benchmark-strength claim for DWA still requires
  a comparator and broader campaign evidence.

### Aggregate outcome

| Outcome | Episodes |
| --- | --- |
| Total written | 69 |
| Route completed | 0 |
| Timeout (`max_steps`) | 56 |
| Collision-terminated | 13 |
| Runtime/simulation failure | 0 |
| Fallback / degraded exclusion | 0 |

### Collision-terminated rows

`classic_station_platform_medium` (1/3), `classic_cross_trap_high` (1/3), `classic_doorway_low`
(1/3), `classic_doorway_medium` (1/3), `classic_doorway_high` (1/3), `classic_overtaking_low`
(1/3), `classic_overtaking_medium` (1/3), `classic_t_intersection_low` (3/3),
`classic_t_intersection_medium` (3/3).

## Status rows

The retained compact table is:

- [`dwa_archetype_matrix_status_rows.csv`](dwa_archetype_matrix_status_rows.csv)

It records each of the 23 archetype rows with its archetype, density tier, fixed seeds, episode
count, route-completions, timeouts, collision-terminated count, and median step count / average
speed / minimum clearing distance across the 3 seeds.

## Interpretation and caveats

- **Local-minimum / progress failure is the dominant failure mode.** DWA selects commands from a
  one-period dynamic window and scores a short constant-velocity rollout; on bottleneck, doorway,
  cross-trap, head-on, merging, and group-crossing geometries this reactive scoring stalls short
  of the goal within the 100-step horizon. This is expected behavior for an untuned classical DWA
  and is reported as-is, not as a defect of the simulator.
- **T-intersection rows collision-terminate on all three seeds.** These are the only fully
  collision-terminated rows; they warrant a follow-up look at spawn geometry / first-step clearance
  if DWA is ever promoted toward headline use, but that is out of scope for this evidence packet.
- **`ped_density` semantics differ by spawn mode.** Four marker-spawn rows
  (`classic_bottleneck_*`) carry `ped_density: 0.0` as a placement-mode placeholder, not an empty
  scene (see the density tier index). Their low `min_clearance` reflects marker placement, not
  absence of pedestrians.
- **Not a comparator run.** To make any "DWA vs planner X" claim, a paired comparator run under
  the identical matrix/seeds/horizon must be added in a separate, scoped piece of work.
- **Horizon sensitivity.** All rows used the run default horizon of 100 steps. A longer horizon
  would change timeout fractions and is a separate contract.

## What this resolves for #5020

Maps to the #5020 acceptance criteria:

- [x] Run the declared standard archetype matrix with `algo=dwa`, its canonical config, fixed
      seeds, and native execution.
- [x] Record config, commit, seeds, status rows, and fallback/degraded exclusions in a durable
      report (this README + the CSV).
- [x] Classify the result as smoke/nominal benchmark evidence only within the executed contract;
      no comparative claims are made (no comparator).
