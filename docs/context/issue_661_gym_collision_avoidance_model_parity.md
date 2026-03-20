# Issue 661 gym-collision-avoidance Model Parity Probe

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#659` headless upstream reproduction
- `robot_sf_ll7#661` side-environment CADRL-family wrapper parity

## Goal

Check whether the current local `robot_sf.planner.socnav._SACADRLModel` and discrete action table
already match the upstream GA3C-CADRL checkpoint on a real native upstream observation.

This issue does **not** yet claim Robot SF wrapper parity. It isolates the model-level question first.

## Canonical probe artifacts

- JSON report:
  `output/benchmarks/external/gym_collision_avoidance_model_parity/report.json`
- Markdown report:
  `output/benchmarks/external/gym_collision_avoidance_model_parity/report.md`
- Captured native upstream payload:
  `output/benchmarks/external/gym_collision_avoidance_model_parity/payload.json`

Generated with:

```bash
uv run python scripts/tools/probe_gym_collision_avoidance_model_parity.py \
  --repo-root output/repos/gym-collision-avoidance \
  --side-env-python output/benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python \
  --output-json output/benchmarks/external/gym_collision_avoidance_model_parity/report.json \
  --output-md output/benchmarks/external/gym_collision_avoidance_model_parity/report.md
```

## Current result

Verdict: `native-model parity reproduced`

Observed parity on one live upstream GA3C-CADRL observation:
- upstream argmax: `4`
- local argmax: `4`
- probability max abs diff: `0.00000000`
- discrete action-table max abs diff: `0.00000001`
- native observation shape: `[1, 132]`
- states used:
  - `num_other_agents`
  - `dist_to_goal`
  - `heading_ego_frame`
  - `pref_speed`
  - `radius`
  - `other_agents_states`

Interpretation:
- the current local TensorFlow checkpoint wrapper is not the source of CADRL-family drift on this tested native observation,
- the local discrete action table also matches the upstream action set up to float noise,
- the remaining open gap is Robot SF observation mapping and downstream benchmark behavior.

## Why this matters

This meaningfully narrows the CADRL-family uncertainty.

What is now supported by evidence:
- upstream example completion is possible in the isolated side environment under explicit headless shims,
- the local `_SACADRLModel` reproduces upstream checkpoint inference on native source input,
- future CADRL-family work should focus on observation/adapter parity rather than re-litigating checkpoint loading.

What is still **not** established:
- Robot SF benchmark parity,
- end-to-end wrapper fidelity on Robot SF observations,
- benchmark quality or paper-surface competitiveness.

## Recommendation

Recommendation: `adapter-parity work is now the right next step`

Reason:
- model-level parity is no longer the main uncertainty,
- the next meaningful risk is how Robot SF observations are converted into the upstream state contract,
- that is the layer that should now be tested or corrected.
