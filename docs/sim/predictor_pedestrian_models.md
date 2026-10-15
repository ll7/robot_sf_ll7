# Alternative pedestrians for predictor development

Load `configs/scenarios/predictor_pedestrian_models.yaml` with
`robot_sf.training.scenario_loader.load_scenarios`, then pass a selected entry
to `build_robot_config_from_scenario(entry, scenario_path=path)`. The two entries
use the same corridor, crowd settings, and development seed `1001`:

| Scenario | `simulation_config.pedestrian_model` |
| --- | --- |
| `predictor_social_force_control` | `social_force_default` |
| `predictor_zanlungo_alternative` | `hsfm_zanlungo_collision_prediction_v1` |

The alternative reuses the existing [Zanlungo implementation](../context/issue_4973_zanlungo_collision_prediction.md).
It replaces the default pedestrian–pedestrian Social Force term with repulsion
based on neighbors projected to each actor's earliest eligible closest-approach
time. Goal, obstacle, group, and robot forces remain shared. It also uses the
existing total-force heading update; the default derives heading from velocity.
The shipped pair keeps the same semi-implicit integration scheme.

For an existing scenario, select the alternative with:

```yaml
simulation_config:
  pedestrian_model: hsfm_zanlungo_collision_prediction_v1
```

The selector enables the nested `zanlungo_collision_prediction` configuration;
the example pair pins its parameters explicitly. Omitting `pedestrian_model`
retains the default Social Force path. There is no new solver or dependency.

This is a diagnostic development option for #10407. It creates a mismatch in
pedestrian interaction equations for evaluating a Social Force predictor, but
shares other SFM terms and does not establish independence from all simulator
assumptions. The parameters are not calibrated to Robot SF, and the pair is not
part of standard campaigns. Tests prove selection, deterministic trajectory
differences, and exact default-path equivalence; they do not prove predictor
gains or realism. Future evaluation must keep robot dynamics, scenarios, and
compute budgets matched and use development seeds `1001–1200` only.

Run the implementation check with:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/sim/test_predictor_pedestrian_models.py -q
```
