# Development time-step comparison

The [time-step study](https://github.com/ll7/robot_sf_ll7/issues/10516) pairs each of five scenarios
with ten development seeds and the 14 planners from the benchmark-data template. The three
`issue_10516_timestep_dev_v1_*ms.yaml` campaign configs use the same scenario manifest, seeds, robot
motion model, and planner inputs. The scenario manifest permits the longest step budget. Each
campaign binds the exact limit in both the runner and the simulator.

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
The report includes the observed time grid, planner execution status, and any refusal or
degradation. The native safety decisions of Guarded PPO are part of that planner's configured
behavior. Runtime fallback or degraded execution stays visible in the report.

Run the existing campaign preflight for each config before acquisition. For example:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_10516_timestep_dev_v1_100ms.yaml \
  --mode preflight --checkpoint-preflight-mode enforced_staged
```

The expansion and witness command is report-only and runs separately from campaign admission
and pull-request readiness. Its output qualifies bindings for development runs. Outcome estimates
and planner order require the matched campaign.
