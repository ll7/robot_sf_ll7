# Issue 1369 SAGE MPC-Transfer Planner Reproducibility Assessment

Date: 2026-05-20

Related issue:
- `robot_sf_ll7#1369` research: assess SAGE MPC-transfer planner reproducibility

Related context:
- `docs/context/external_planner_reuse_checklist.md`
- `docs/benchmark_planner_family_coverage.md`
- `robot_sf_ll7#1355` external learned-policy candidate matrix
- `robot_sf_ll7#1365` graph observation adapter planning
- `robot_sf_ll7#1367` CrowdNav-family verdict

## Goal

Assess whether SAGE, the graph-based MPC-transfer navigation planner from
`TIB-K330/drl_planner`, is reproducible enough to justify a Robot SF wrapper, benchmark entry, or
follow-up implementation issue.

## Sources Checked

- Upstream repository: https://github.com/TIB-K330/drl_planner
- Local source checkout: `output/repos/drl_planner`
- Source revision checked: `debbace` (`2025-05-31`, `Update graph_sac.py`)
- Paper DOI linked by upstream README: https://doi.org/10.1109/lra.2024.3412610
- License file: MIT, copyright 2025 TIB-K330

The upstream README identifies the project as the official code for "Sample-efficient
learning-based dynamic environment navigation with transferring experience from
optimization-based planner" and states that the core code is released, but the training part is
not yet fully organized.

## Source Contract

The released code is useful as a source-side reference, but it is not a drop-in Robot SF planner.

Observed runnable surfaces:

- `train.py` is the only top-level executable entry point found.
- It imports legacy `gym`, creates `gym.make("CrowdSim-v0")`, configures a bundled `crowd_sim`
  environment, trains `MpcDroQ("GraphPolicy", ...)` for `10000000` steps, then evaluates the saved
  best model.
- There is no `requirements.txt`, `pyproject.toml`, `setup.py`, or environment file in the checked
  source tree.
- No checkpoints, trained policy archives, `data/mpc_rollouts.pkl`, or other source-buffer data
  were present in the checked tree.
- The MPC-transfer hook exists as `MpcDroQ.load_source_buffer(...)`, but the only call site in
  `train.py` is commented: `model.load_source_buffer("data/mpc_rollouts.pkl")`.

Observation and graph shape:

- The default config uses a differential robot, `human_num = 5`, `obstacle_num = 3`,
  `wall_num = 4`, `time_step = 0.25`, ORCA humans, and circle-crossing train/test scenarios.
- `modules/crowd_graph.py` constructs a DGL graph over robot, human, obstacle, and wall nodes.
- Node features are not raw Robot SF structured observations. They are transformed through an
  RVO-derived, robot-frame representation with velocity-obstacle features, obstacle/wall distances,
  expected collision-time terms, and source-simulator normalization.
- A Robot SF adapter would therefore need the graph-observation adapter surface tracked by
  `robot_sf_ll7#1365`, plus explicit parity checks against the source harness.

Action and kinematics:

- The default source config sets `robot.kinematics = "differential"`.
- The learning stack acts through a two-dimensional Gym `Box` action space and feeds source
  `ActionDiff` values to the source environment.
- This is not the same public contract as Robot SF `unicycle_vw`; any wrapper would need explicit
  action scaling, bounds, and projection proof before benchmark use.

## Source Smoke

Command:

```bash
PYTHONPATH=output/repos/drl_planner timeout 120s .venv/bin/python \
  output/repos/drl_planner/train.py --debug --output_dir output/sage_source_smoke
```

Result:

```text
ModuleNotFoundError: No module named 'gym'
```

This smoke used the Robot SF virtual environment, which has Gymnasium-oriented dependencies but not
legacy `gym`. I did not install guessed dependencies into the Robot SF environment because the
upstream repo does not publish a requirements or environment file, and the entry point also depends
on a DGL graph stack and source-specific replay-buffer artifacts.

## Reproducibility And Fairness Risks

- Training is not fully organized by the upstream author's own README statement.
- The published core code does not include dependency pins, checkpoints, or the offline MPC
  experience buffer that makes the SAGE method sample-efficient.
- A Robot SF wrapper without the source MPC buffer would not test the method described by the
  paper; a wrapper with a locally regenerated buffer would need provenance that distinguishes
  source-simulator pretraining from Robot SF benchmark evaluation.
- The graph construction bakes in source-simulator RVO feature engineering. Treating those features
  as equivalent to Robot SF structured state would overclaim compatibility.
- No Robot SF benchmark run was attempted because there is no verified inference policy, adapter,
  or source-harness parity command.

## Verdict

Current tier: `monitor / source-side blocked`.

Do not add a Robot SF planner wrapper, benchmark config, or training issue yet. SAGE is relevant to
the graph-observation and MPC-transfer roadmap, but the available source is not reproducible enough
for benchmark-facing implementation in this repository.

Before revisiting implementation, request or locate:

- exact dependency/environment pins,
- a runnable source-side training and evaluation command,
- the offline MPC source buffer or a reproducible generation command,
- trained checkpoints or a documented inference-only path,
- and source-harness metrics that can be reproduced before any Robot SF adapter work starts.

## Coverage Matrix Recommendation

Track SAGE as an external family anchor only. It should be useful for the external learned-policy
candidate matrix and graph-observation adapter planning, but it must not be counted as current
in-repo benchmark support.
