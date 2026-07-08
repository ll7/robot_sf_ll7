# Issue #4871 — CrowdNav_Prediction_AttnGraph external learned baseline feasibility smoke

**Status:** smoke PASS (go). Checkpoint loads and acts on synthetic Robot SF
observations; per-step inference is trivially fast. Zero-shot transfer into Robot
SF scenarios is **not** scientifically defensible without an explicit
out-of-training-distribution caveat (see Transfer caveats). This note records the
go/no-go data only — **no roster addition, no campaign, no retraining**.

**Upstream:** `Shuijing725/CrowdNav_Prediction_AttnGraph` (ICRA 2023, "Intention
Aware Robot Crowd Navigation with Attention-Based Interaction Graph"), pinned at
commit `390773137be04ed14e27620dc5fd7c5e1a5b1f62` (MIT). Staged via
`scripts/tools/manage_external_repos.py stage crowdnav_pred_attng`.

**Artifact:** `robot_sf/planner/crowdnav_pred_attng.py` (model-only adapter) +
`tests/planner/test_crowdnav_pred_attng.py` (staged-checkout smoke).

## Verdict by issue step

| Step | Gate | Result |
| --- | --- | --- |
| 1. Prior art | No prior assessment of this repo; no dormant in-tree scaffold | PASS. The same author's `CrowdNav_HEIGHT` is already integrated as the `crowdnav_height` planner and was used as the template pattern. `grep crowdnav/attn/graph` in `robot_sf/baselines/` and `algorithm_metadata.py` returned no Prediction_AttnGraph entry. |
| 2. License | MIT permits academic benchmark reuse | PASS. `LICENSE` at the pinned commit is MIT. |
| 3. Pretrained checkpoints | Authors ship a trained RL navigation policy | PASS (no retraining needed). The "Ours" attention-graph SRNN policy ships as `trained_models/GST_predictor_non_rand/checkpoints/41200.pt` (no randomized humans) and `.../GST_predictor_rand/checkpoints/41665.pt` (randomized humans). The `00000.pt` files under `ORCA_no_rand`/`SF_no_rand` are the non-learning ORCA/SF baselines, not learned weights. |
| 4. Stage + inference path | Repo stages; checkpoint loads and runs | PASS. Staged at the pinned SHA. The PyTorch policy path loads and runs (model-only adapter; no OpenAI Baselines / `crowd_sim` / Python-RVO2 / TensorFlow install required). |
| 5. Contract + timing | Thinnest adapter produces actions; per-step wall-clock measured | PASS. Adapter reconstructs the upstream dict observation and calls `policy.act`. Per-step ≈ 1–2 ms on CPU, flat in observed neighbor count. |
| 6. Record verdict | Gate-doc row written | PASS. See `docs/benchmark_experimental_planners.md`. |

A documented negative at any step would have been a complete deliverable; none of
the gates failed.

## What was actually exercised (and what was deliberately not)

**Exercised:** the PyTorch RL navigation policy only. The adapter imports
`rl/networks/model.Policy`, loads `41200.pt` via `torch.load(weights_only=False)`,
pins `base.nenv = 1` (matching the upstream `test.py` entrypoint), reconstructs
the upstream dict observation from world-frame Robot SF state, and calls
`policy.act(..., deterministic=True)`. The import context short-circuits the one
transitive heavy import (`rl.networks.envs`, which otherwise pulls in `gym` +
OpenAI Baselines + `crowd_sim`) with a minimal `VecNormalize` stub, so the network
loads against PyTorch alone.

**Deliberately NOT exercised (follow-up cost classes):**
- The **TensorFlow GST trajectory-predictor** inference path
  (`CrowdSimPredRealGST-v0`, `gst_updated/`). The smoke reconstructs the 5-step
  future positions with a **constant-velocity** model instead, which needs no GST
  model and no TensorFlow.
- The full `crowd_sim` rollout / `test.py` evaluation path (OpenAI Baselines,
  Python-RVO2 for ORCA humans, gym 0.15.7). The smoke is model-only.
- The randomized-humans checkpoint `41665.pt` (same architecture; not needed for
  the go/no-go).

## Checkpoint architecture (state_dict is authoritative)

The bundled `trained_models/GST_predictor_non_rand/configs/config.py` sets
`predict_method='none'` and `arguments.py` sets `env_name='CrowdSimVarNum-v0'`,
which would build the spatial-attention input with width 2. **That does not match
the shipped weights.** `load_state_dict` fails with
`base.spatial_attn.embedding_layer.0` = `Linear(12→128)`, i.e. the checkpoint was
trained with the 5-step **prediction** variant. The adapter therefore uses
`env_name='CrowdSimPred-v0'` (spatial input width `2*(predict_steps+1) = 12`),
which loads cleanly. The directory name `GST_predictor_*` is consistent with the
weights; the bundled test config is simply inconsistent and would need editing to
run upstream `test.py` against this checkpoint.

Confirmed architecture (`human_num=20`, holonomic 2-D action):

- `base.spatial_attn.embedding_layer.0` `Linear(12→128)` → 5-step prediction edges
- `base.robot_linear.0` `Linear(9→256)` → `cat(temporal_edges[2], robot_node[7])`
- `base.spatial_linear.0` `Linear(512→256)` → human-human self-attention (`use_self_attn`)
- `base.attn` `EdgeAttention_M` → robot-human attention (`use_hr_attn`)
- `base.humanNodeRNN` (`EndRNN`, GRU size 128) → recurrent node state
- `dist.fc_mean` `Linear(256→2)` + `dist.logstd` → `DiagGaussian` holonomic `ActionXY(vx,vy)`
- recurrent state: dict with `human_node_rnn` `(1,1,128)` and `human_human_edge_rnn` `(1,21,256)`

## Observation contract mapping (Robot SF → upstream)

Upstream `crowd_sim/envs/crowd_sim_pred.py generate_ob`, reconstructed in the adapter:

| Upstream field | Shape | Source from Robot SF `Observation` |
| --- | --- | --- |
| `robot_node` | `(1,7)` `[px,py,r,gx,gy,v_pref,theta]` | `robot.position`, `robot.radius`, `robot.goal`, `v_pref=1.0`, `theta=atan2(vy,vx)` |
| `temporal_edges` | `(1,2)` `[vx,vy]` | `robot.velocity` |
| `spatial_edges` | `(20,12)` | per closest-20 pedestrian: current + 5 const-velocity future positions, world-frame relative to robot, sorted by current distance, padded with sentinel 15 |
| `detected_human_num` | `(1,)` | `min(agent_count, 20)`, floored at 1 |

Action mapping: upstream emits holonomic `ActionXY(vx,vy)`, clipped to magnitude
`v_pref=1.0` (upstream `clip_action`). This is a valid Robot SF `velocity`-space
command. **Projection to a unicycle `(v, ω)` command is an explicit transfer
caveat, not a silent transform** — the adapter returns `(vx, vy)`.

## Per-step inference wall-clock

CPU (single thread, `torch.set_num_threads(1)`), 30-step mean after 3 warmup
steps, from `tests/planner/test_crowdnav_pred_attng.py`:

| Observed neighbors | ms/step |
| --- | --- |
| 2 | ~2.2 |
| 5 | ~2.2 |
| 10 | ~2.2 |
| 20 | ~2.2 |

Flat in observed neighbor count because `human_num` is fixed at 20 (padding), so
the attention cost is constant. Standalone (outside pytest) ≈ 1.1 ms/step. Either
way this is far below any Robot SF benchmark step budget; **runtime is not the
blocker for this planner family**.

Sanity signal: with the robot at the origin and the goal at `(5, 0)` in open
space, the deterministic action is `≈ (0.98, -0.18)` — strongly toward the goal,
i.e. the loaded weights compute meaningful navigation, not noise.

## Transfer caveats (the real question — zero-shot into Robot SF is NOT defensible)

Any future comparison slice must frame results as **policy evaluated
out-of-training-distribution** until the mismatches below are reconciled:

1. **Kinematics: holonomic vs differential-drive.** Upstream trains/outputs
   holonomic `ActionXY(vx,vy)`. Robot SF default robots are unicycle `(v, ω)`.
   A holonomic→unicycle projection (e.g. align linear speed along heading, derive
   `ω` from heading error) changes the reachable action set and must be a
   deliberate, documented decision, not a silent remap.
2. **Pedestrian model: ORCA vs social-force.** Upstream trains against ORCA
   pedestrians (`humans.policy="orca"`); Robot SF default crowd is social-force.
   The learned interaction graph encodes ORCA avoidance dynamics, so zero-shot
   transfer to social-force crowds is the headline OOD risk. The SICNav smoke
   (#4870) carries the same caveat; a future comparison would need an ORCA
   pedestrian mode in Robot SF and explicit framing.
3. **Prediction variant vs the paper's headline.** The smoke uses a
   **constant-velocity** 5-step prediction (no GST/TensorFlow). The paper's
   "intention-aware" contribution is the GST-inferred prediction. Exercising the
   GST variant is a separate, heavier dependency path (TensorFlow + GST weights).
4. **Sensing radius / FOV.** Upstream `robot.sensor_range=5`, `FOV=2π` (full).
   Robot SF sensor models differ; any radius/FOV mismatch changes which humans
   populate the spatial edges.
5. **Human-count cap.** Upstream `human_num=20` fixed; denser Robot SF scenarios
   truncate to the 20 closest, sparser ones pad with the sentinel. The
   attention mask (`detected_human_num`) handles this correctly, but it is a
   capacity assumption.
6. **Control cadence.** Upstream `env.time_step=0.25s`; the recurrent policy was
   trained at that cadence. The adapter enforces 0.25s and rejects other `dt`.
   Robot SF typically steps at a different cadence; running the recurrent policy
   at the wrong cadence changes its dynamics.
7. **Preferred speed.** Upstream `v_pref=1.0 m/s`; Robot SF robots have different
   speed envelopes. The clip normalizes to 1.0, which may under-use a faster robot
   or over-command a slower one.
8. **Reward / arena.** Upstream is open-space circle-crossing (arena radius 6,
   discomfort dist 0.25, collision penalty −20). Robot SF scenarios include
   static obstacles and doorway geometry that this open-space policy never saw.

## What this issue is NOT

- **Not a roster addition.** The planner is not registered in
  `algorithm_metadata`, `algorithm_readiness`, or the testing-only list. It must
  not enter benchmark sweeps.
- **Not a campaign / benchmark claim.** No success/collision numbers are claimed
  for Robot SF scenarios. The upstream test log (0.92 success, 0.07 collision on
  their 20-human ORCA circle-crossing) is upstream-environment context only.
- **Not a retraining.** The shipped checkpoint is used as-is.

Roster/campaign/retraining decisions are separate maintainer calls after this smoke.
