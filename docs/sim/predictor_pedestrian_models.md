# Alternative pedestrians for predictor development

Load `configs/scenarios/predictor_pedestrian_models.yaml` with
`robot_sf.training.scenario_loader.load_scenarios`, then pass a selected entry
to `build_robot_config_from_scenario(entry, scenario_path=path)`. The three entries
use the same corridor, crowd settings, and development seed `1001`:

| Scenario | `simulation_config.pedestrian_model` |
| --- | --- |
| `predictor_social_force_control` | `social_force_default` |
| `predictor_orca_alternative` | `orca_pedestrians_v1` |
| `predictor_zanlungo_alternative` | `hsfm_zanlungo_collision_prediction_v1` |

Use **ORCA** to evaluate a Social Force predictor against different motion
equations. It reuses the repository's native `rvo2` solver and bypasses SFM
force calculation and integration entirely. Existing spawn/speed sampling and
route/goal controllers remain shared. Preferred velocity points toward the
current goal, capped by the pedestrian speed limit and distance to that goal
over one timestep. Native ORCA chooses and applies a collision-avoidance velocity
directly; it does not integrate a Social Force acceleration.

For an existing scenario, select the alternative with:

```yaml
simulation_config:
  pedestrian_model: orca_pedestrians_v1
```

The versioned ORCA mode uses a 10 m neighbor range, all actors within range as
eligible neighbors, and 2.5 s agent/wall horizons. Static walls use existing map
segments (unless pedestrian obstacle interaction is disabled). Robots enter as
discs at their current position and world velocity; their solver outputs are
discarded, preserving actual robot dynamics. Native pedestrians retain standard
half-responsibility reciprocity, while the robot follows its own controller;
this assumption and instantaneous pedestrian velocity changes do not guarantee
clearance with non-reciprocal or acceleration-limited actors. The small native
scene is rebuilt each step, so reset/respawn never retains stale solver state.
Missing `rvo2` raises an error; there is no SFM fallback or new dependency.

ORCA does not apply SFM obstacle/robot/group forces or residual adversaries.
Social Force oracle tracing/residuals and ego-pedestrian environments are
explicitly rejected for this first mode. `last_ped_forces` holds effective
velocity-change acceleration for ORCA, not decomposable SFM force components;
robot-force component outputs are empty/zero. Use motion-based metrics for this
diagnostic option. Omitting `pedestrian_model` retains the default SFM path.

The [Zanlungo implementation](../context/issue_4973_zanlungo_collision_prediction.md)
remains an additional SFM-related comparison. It replaces the pedestrian–pedestrian
Social Force term with earliest-approach projection, but retains goal, obstacle,
group and robot force equations and a total-force heading update. It only partly
addresses predictor/simulator equation sharing; ORCA is the independent choice.

These are diagnostic development options for #10407, outside standard campaigns
and uncalibrated for Robot SF. Tests prove selection, SFM bypass, deterministic
trajectory differences, and exact default-path equivalence; they do not prove
predictor gains or realism. Future evaluations must keep robot dynamics,
scenarios, and compute budgets matched and use dev seeds `1001–1200` only.

Run the implementation check with:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/sim/test_predictor_pedestrian_models.py tests/ped_npc/test_orca_pedestrians.py -q
```
