# Issue #1366 GenSafeNav / SoNIC Conformal Contract Assessment

Date: 2026-05-20

Related issues:

* <https://github.com/ll7/robot_sf_ll7/issues/1366>
* [Issue #626 SoNIC Source-Harness Reproduction Probe](../issue_626_sonic_source_harness_probe.md)
* [Issue #627 SoNIC / GenSafeNav Model-Only Wrapper Follow-Up](../issue_627_sonic_wrapper_followup.md)
* [Issue #1247 Prediction-Aware Safety Shield Contract](../issue_1247_safety_shield_contract.md)
* Issue #1355 External Learned-Policy Candidate Matrix
* Issue #1359 Learned-Policy Reject/Monitor Registry

## Goal

Assess whether GenSafeNav / SoNIC-style adaptive conformal inference (ACI) and constrained RL can
be evaluated in Robot SF without calibration leakage or privileged future-trajectory observations.

## Source Evidence Checked

Primary external sources:

* GenSafeNav repository: <https://github.com/tasl-lab/GenSafeNav>
* GenSafeNav paper: <https://arxiv.org/abs/2508.05634>
* SoNIC repository: <https://github.com/tasl-lab/SoNIC-Social-Nav>
* SoNIC paper: <https://arxiv.org/abs/2407.17460>
* Issue #1393 source-harness rerun:
  [issue_1393_gensafenav_source_harness.md](issue_1393_gensafenav_source_harness.md)

Local source snapshot inspected in the ignored worktree output area:

* `output/repos/GenSafeNav`
* commit: `01baf926a5c77c1a4ab28635658eb014ef4f1767`
* license: MIT
* included pretrained checkpoints: `trained_models/Ours_GST/checkpoints/05207.pt`,
  `trained_models/GST_predictor_rand/checkpoints/05207.pt`, and source-side classical-policy
  checkpoints for `ORCA` and `SF`.

These output paths are not tracked durable artifacts. They record the local inspection surface used
for this note and can be recreated by cloning the upstream repository and rerunning the validation
commands below.

The upstream README states that GenSafeNav augments observations with ACI prediction uncertainty and
uses constrained RL; it also documents a Docker/GPU-oriented quick start and `python test.py` /
`python visualize.py` entrypoints. The arXiv abstract makes the same high-level claim and reports
intrusion counts against ground-truth human future trajectories. Those intrusion labels are valid
for evaluation metrics, but not as deployment-time policy inputs.

## Source-Harness Result

Command:

```bash
uv run python scripts/tools/probe_sonic_source_harness.py \
  --repo-root output/repos/GenSafeNav \
  --model-name Ours_GST \
  --checkpoint 05207.pt \
  --output-json output/benchmarks/external/gensafenav_source_harness_probe/report.json \
  --output-md output/benchmarks/external/gensafenav_source_harness_probe/report.md
```

Result: `source harness blocked`.

Observed blocker:

```text
ModuleNotFoundError: No module named 'gym'
```

The probe still extracted the saved source contract:

* `robot_policy`: `selfAttn_merge_srnn`
* `human_policy`: `orca`
* `robot_sensor`: `coordinates`
* `predict_method`: `inferred`
* `action_kinematics`: `holonomic`
* `env_use_wrapper`: `True`
* default env: `CrowdSimPredRealGST-v0`
* Docker base image: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`

Interpretation: Robot SF can inspect and run model-only wrapper paths, but the source-faithful
GenSafeNav harness is not reproduced in the current repository Python environment. This issue should
not promote the family to benchmark-ready status.

The probe writes `output/benchmarks/external/gensafenav_source_harness_probe/report.json` and
`report.md` as ignored, rerunnable evidence. Do not treat those files as durable dependencies unless
they are intentionally promoted under the repository evidence-bundle policy.

## Prediction And Calibration Boundary

The relevant upstream code paths are:

* `trained_models/Ours_GST/configs/config.py`
  * enables `policy.aci_input = True`
  * enables `policy.constrain_cost = True`
  * sets `sim.predict_method = "inferred"`
  * sets `env.use_wrapper = True`
  * uses `action_space.kinematics = "holonomic"`
* `crowd_sim/envs/crowd_sim_pred_real_gst.py`
  * exposes `spatial_edges` as predicted pedestrian trajectories
  * exposes `conformity_scores` as per-human, per-horizon uncertainty tokens
  * computes `conformal_intrusion_cost` from predicted points and ACI buffers
* `crowd_sim/envs/utils/human.py`
  * updates DtACI using previous realized pedestrian locations and previous predictions
* `crowd_sim/envs/crowd_sim_pred.py`
  * computes ground-truth future trajectories in test mode for intrusion/reward diagnostics
* `rl/networks/selfAttn_srnn_temp_node.py`
  * concatenates `conformity_scores` into the policy input when `policy.aci_input` is true
* `train.py`
  * trains PPO Lagrangian / constrained-RL cost critics using ACI-derived costs

Field classification:

| Field or source | Classification | Benchmark rule |
| --- | --- | --- |
| robot pose, velocity, goal, radius | deployment-observable | Allowed. |
| current tracked pedestrian pose/velocity | deployment-observable only under the declared observation level | Allowed for `tracked_agents_no_noise` or explicit synthetic-noise levels; not real-sensor certification. |
| GST-predicted pedestrian trajectory in `spatial_edges` | deployment-derived prediction | Allowed only when generated from current/past observations by a frozen predictor with no evaluation future labels. |
| `conformity_scores` policy input | deployment-derived online uncertainty | Allowed only if computed from prior realized observations/predictions or from a train/validation-only calibration state; must not consume the current decision's future label. |
| DtACI updates from `gt_locations_aci` | delayed deployment observation | Allowed for online adaptation after the real position is observed; forbidden if used to tune on held-out evaluation seeds before reporting those same seeds. |
| `conformal_intrusion_cost` | training / diagnostic signal | Allowed for training or diagnostics; not a replacement for Robot SF benchmark metrics. |
| cost critic / Lagrange multiplier state | training-only | Allowed only as frozen learned-policy state at evaluation. |
| upstream `human_future_traj` with `method="truth"` | oracle metric / diagnostic | Forbidden as policy input; allowed only to compute source-side intrusion metrics after the fact. |
| evaluation-seed calibration of ACI quantiles or buffers | forbidden | Would leak evaluation labels into policy state and invalidate benchmark evidence. |

## Train-Only Calibration Protocol

If the family is revisited for promotion, the minimum leakage-free protocol is:

1. Freeze the GST predictor, policy checkpoint, and any cost critic before benchmark evaluation.
2. If ACI needs offline calibration, use only training or validation seeds that are excluded from
   the reported Robot SF evaluation split.
3. Permit online ACI updates during an episode only from observations that would have been available
   before the next control decision.
4. Record the calibration state and observation level in the benchmark metadata.
5. Fail closed when a run would need `human_future_traj` truth, current-episode future labels, or
   source evaluation seeds to initialize uncertainty.

Robot SF does not yet have enough source-harness proof to verify that this protocol is implemented
end to end for GenSafeNav.

## Verdict

Verdict: `source-side reproduction first`.

GenSafeNav / SoNIC remains high-value for safety and OOD research, but the current evidence is not
enough for main-table benchmark inclusion. The already implemented Robot SF wrappers should remain
explicitly experimental, model-only, and adapter-level. The conformal/ACI method should not become a
policy-search candidate or benchmark row until source-side execution is reproducible and the ACI
state is proven train-only or online-past-only.

Not `wrapper spike` yet because the model-only adapter does not prove source-faithful uncertainty
or constrained-RL semantics.

Not `monitor only` yet because the inspected source suggests a plausible leakage-free route if the
calibration state is frozen or online-past-only.

Not `reject for now` because the source includes MIT-licensed code, trained checkpoints, a Docker
path, explicit ACI state, and a clear prediction/uncertainty contract worth preserving for future
source-harness work.

## Follow-Up Boundary

Safe follow-ups:

* improve the source-harness environment until `python test.py --model_dir trained_models/Ours_GST`
  runs without local compatibility shims;
* add metadata fields that distinguish model-only adapter, source-faithful runtime, train-only
  calibration, and online ACI adaptation;
* reference this verdict from the external learned-policy matrix and reject/monitor registry.

Unsafe follow-ups:

* treating `ours_gst` or `gensafenav_ours_gst` as source-faithful GenSafeNav evidence;
* using ground-truth future trajectories as Robot SF policy input;
* calibrating conformal intervals on the same seeds used for reported benchmark evaluation;
* merging guarded model-only aliases into the base GenSafeNav claim.
