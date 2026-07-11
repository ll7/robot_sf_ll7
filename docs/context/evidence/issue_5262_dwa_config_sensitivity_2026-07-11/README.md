<!-- AI-GENERATED (robot_sf#5262, 2026-07-11) - NEEDS-REVIEW -->
# Issue #5262 — DWA Configuration-Sensitivity Diagnostic

This bounded CPU-only diagnostic tests whether small, reversible Dynamic Window Approach (DWA)
configuration changes can explain its prior 0/69 goal-completion result.

## Claim boundary and status

- **Evidence status:** diagnostic-only.
- **Claim boundary:** 27 executions only: three classic archetypes, their fixed three-seed sets,
  and three predeclared DWA points. This does not alter DWA roster status, benchmark metric
  semantics, or the frozen v0.1 suite.
- **Major caveats:** the runner used experimental-profile adapter execution; no fallback or
  degraded execution was counted as success. This is neither a comparator run nor a full
  benchmark campaign.
- **Uncertainty:** about 85% confidence that the tested small retunes are insufficient for this
  failure slice. That conclusion would change if a larger, preregistered search or an
  implementation-mechanism trace isolates a viable untested parameter region.

## Verdict

**`needs-implementation-change`** for the tested slice. All 27 episodes completed without a
runtime failure, yet none reached the goal. The mobility/tolerance point shortened the
T-intersection collision median from 78 to 55 steps, but did not change its three collisions;
the progress-weight point also preserved every failure outcome. Thus the dominant observed
failure is not the canonical values of these bounded configuration axes: bottleneck remains a
timeout and T-intersection remains collision-terminated under every point.

This verdict does not prove that no DWA retune can work. The next empirical action is a
mechanism-level DWA decision trace on the bottleneck timeout and T-intersection first collision,
then a separately scoped implementation change if that trace identifies a concrete defect or
missing recovery behavior.

## Contract

| Field | Value |
| --- | --- |
| Source matrix | [`configs/scenarios/classic_interactions.yaml`](../../../../configs/scenarios/classic_interactions.yaml) |
| Selected archetypes | `classic_bottleneck_medium` (timeout-dominated), `classic_cross_trap_high` (mixed), `classic_t_intersection_low` (collision-dominated) |
| Seeds | `131;132;133`, `101;102;103`, `161;162;163` respectively |
| Planner | `algo=dwa`, experimental profile, adapter execution |
| Base config | [`configs/algos/dwa_classic.yaml`](../../../../configs/algos/dwa_classic.yaml) |
| Declared sweep | [`configs/benchmarks/issue_5262_dwa_config_sensitivity.yaml`](../../../../configs/benchmarks/issue_5262_dwa_config_sensitivity.yaml) |
| Horizon / timestep | 100 steps / 0.1 s |
| Repo commit at execution | `cb520dc0e972e17eccc40827fcb6832b993189ca` |
| Total | 3 config points × 3 archetypes × 3 fixed seeds = 27 episodes |

### Configuration points

| Point | Config hash | Bounded change |
| --- | --- | --- |
| `canonical` | `85fe7a8764671c83` | Canonical DWA config from the preceding 0/69 observation. |
| `mobility_and_goal` | `1be09d2da145af87` | `v_max=1.6`, `omega_max=1.8`, linear/angular acceleration `1.6/2.5`, goal tolerance `0.5`. |
| `progress_weight_400` | `a025ed6417cdb533` | Heading/clearance/velocity/progress weights `0.2/0.6/0.2/4.0`. |

## Results

All points produced the same coarse result: **0/9 route completions, 5 timeouts, and 4
collisions**. Across the 27 executed rows, there were 15 timeouts, 12 collisions, zero route
completions, and zero runtime/simulation failures.

- Per-cell (archetype × config point) table with effective configuration axes:
  [`dwa_config_sensitivity_per_cell_rows.csv`](dwa_config_sensitivity_per_cell_rows.csv)
- Per-episode status rows:
  [`dwa_config_sensitivity_episode_rows.csv`](dwa_config_sensitivity_episode_rows.csv)

## Reproduction

```bash
DISPLAY= SDL_VIDEODRIVER=dummy MPLBACKEND=Agg uv run python \
  scripts/benchmark/run_dwa_config_sensitivity_issue_5262.py \
  --out-dir output/benchmarks/issue_5262
```

The command writes raw JSONL and generated effective configs only under the disposable
`output/benchmarks/issue_5262/` directory. This packet retains the compact derived CSVs needed
to review the diagnosis.

## Acceptance mapping

- [x] Three representative archetypes cover timeout-dominated and collision-dominated failures.
- [x] A ≤30-run CPU-only sweep changes velocity/acceleration bounds, goal tolerance, and objective
      weights through predeclared config points.
- [x] Per-cell and per-episode tables retain config hashes, seeds, and status outcomes.
- [x] The diagnostic verdict is explicit and does not make comparative, roster, paper, or
      frozen-suite claims.
