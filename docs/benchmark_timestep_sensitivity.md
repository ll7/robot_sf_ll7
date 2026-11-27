# Development time-step comparison

The [time-step study](https://github.com/ll7/robot_sf_ll7/issues/10516) pairs each of five scenarios
with ten development seeds and the 14 planners from the benchmark-data template. The three
`issue_10516_timestep_dev_v1_*ms.yaml` campaign configs use the same scenario selection, seeds, robot
motion model, and planner inputs. Each step size has a scenario file that adds its step budget to
the shared selection. The campaign binds that limit in both the runner and the simulator.

| Campaign step (s) | Episode cap (steps) | Episode cap (s) | Planned episodes |
| --- | --- | --- | --- |
| 0.05 | 1,200 | 60 | 700 |
| 0.10 | 600 | 60 | 700 |
| 0.20 | 300 | 60 | 700 |

The controller reads an observation and issues a command once per simulation step, so changing
the step also changes the command interval. This is a closed-loop comparison. The learned
pedestrian predictors keep their released grid of eight samples at 0.1 s, covering 0.8 s.
The sampled-control planner uses the simulation step for its fixed 2 s look-ahead. The manifest
records the resolved timing for each planner and scenario, including the separate rollout grids
used by the other planners.

### Planner timers

The study keeps the planners' step counts. The manifest records each count and its duration on the
grid where it advances. As the command interval grows, the hybrids remember a longer span of
commands and native ORCA takes longer to reach its stall and side-persistence thresholds.

| Planner timing tied to the command interval | Steps or samples | Seconds at 0.05 / 0.10 / 0.20 s |
| --- | --- | --- |
| Four hybrids, oscillation history with weight 0.5 | 6 previous commands | 0.30 / 0.60 / 1.20 |
| Four hybrids, heading smoothness with weight 0.4 | 1 previous command | 0.05 / 0.10 / 0.20 |
| Four hybrids, reaction before braking | 1 step | 0.05 / 0.10 / 0.20 |
| Four hybrids, storage for goal and route progress | 256 samples spanning 255 steps | 12.75 / 25.50 / 51.00 |
| Native ORCA, progress sample used to assess stalls | 1 step | 0.05 / 0.10 / 0.20 |
| Native ORCA, stall count before committing to a side | 3 consecutive stalled calls | 0.15 / 0.30 / 0.60 |
| Native ORCA, side persistence | 10 non-stalled calls | 0.50 / 1.00 / 2.00 |
| SocNav Sampling, look-ahead | 40 / 20 / 10 | 2.00 / 2.00 / 2.00 |

The hybrid progress windows use timestamps at 1, 3 and 5 s. Even the shortest buffer span covers
all three windows. The hybrid route-trace recovery hold is disabled in this study. Its two calls
would span 0.10, 0.20 or 0.40 s. ORCA's reverse-entry and reverse-clear thresholds each count three
calls, or 0.15, 0.30 or 0.60 s. The selected drive settings keep both thresholds inactive.
Social Force filters its target with a 0.5 s time constant and switches turn direction based on
heading and speed. Goal and PPO issue commands at the campaign interval. SA-CADRL also uses that
interval when converting its heading increment to a command.

| Planner grid with a fixed interval | Steps × seconds per step | Seconds at each campaign step |
| --- | --- | --- |
| Prediction Planner and Predictive MPPI, forecast and sampled controls | 8 × 0.10 | 0.80 |
| Prediction Planner, full sequence segment | 3 × 0.10 | 0.30 |
| Predictive MPPI and Guarded PPO, first clearance check | 1 × 0.10 | 0.10 |
| Guarded PPO, command check before its braking tail | 12 × 0.10 | 1.20 |
| RiskDWA and the Guarded PPO alternate controller, command rollout before braking | 16 × 0.10 | 1.60 |
| RiskDWA and the Guarded PPO alternate controller, reachable-command interval | 1 × 0.10 | 0.10 |
| Four hybrids, nominal sampled controls | 8 × 0.20 | 1.60 |

ORCA's pedestrian and obstacle horizons are 6 and 3 s. The hybrid control segment and hard and
soft collision horizons are set in seconds. Stopping tails depend on speed and deceleration.
Counts of sampled candidates and optimization rounds describe search work.

## Compare outcomes

Use route completion without collision for success and the episode collision flag for contact.
Close-clearance time is the number of post-step samples in the benchmark's near-miss distance
band multiplied by the simulation step in seconds. Compare by matched scenario and seed. Use
the same seed blocks at each step size when estimating uncertainty in outcomes or planner order.

## Inspect the grid before running it

Run from the repository root with the benchmark dependencies installed.

```bash
uv run python -m robot_sf.benchmark.timestep_sensitivity \
  --output-dir output/timestep_dev_v1_expansion
```

The command writes `timestep_manifest.json`. Add `--witness` and use a fresh output directory to
run one 0.4 s episode per planner at each step size on the same urban-crossing scenario and seed.
Before shortening an episode, the witness checks that the effective planner step and the runner
and simulator limits for all five scenarios match the campaign. A mismatch produces a refusal in
the report. In the short episode, it checks the observed step and cap, the planner and model
identity, and the learned forecast grid where used. Each witness includes the resolved planner
timing so it can be compared with the manifest. The short run checks these bindings and records
the timers from their settings. A timer may need a longer episode to activate.

Run the existing campaign preflight for each config before acquisition. For example:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_10516_timestep_dev_v1_100ms.yaml \
  --mode preflight --checkpoint-preflight-mode enforced_staged
```

The expansion and witness command is report-only and runs separately from campaign admission
and pull-request readiness. Its output qualifies bindings for development runs. Outcome estimates
and planner order require the matched campaign.
