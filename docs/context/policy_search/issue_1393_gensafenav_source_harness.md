# Issue #1393 GenSafeNav / SoNIC Source-Harness Reproduction

Date: 2026-05-20

Related issues:

* <https://github.com/ll7/robot_sf_ll7/issues/1393>
* [Issue #1366 GenSafeNav / SoNIC Conformal Contract](issue_1366_gensafenav_sonic_conformal_contract.md)
* [Issue #626 SoNIC Source-Harness Reproduction Probe](../issue_626_sonic_source_harness_probe.md)
* [Issue #627 SoNIC / GenSafeNav Model-Only Wrapper Follow-Up](../issue_627_sonic_wrapper_followup.md)
* [Learned-Policy Eligibility Checklist](contracts/learned_local_policy_eligibility.md)

## Goal

Turn the GenSafeNav / SoNIC source-side-first verdict into a fresh executable record for #1393.
The task is source-harness proof, not a Robot SF adapter and not a benchmark claim.

## Source Snapshot

External checkout:

* Repository: <https://github.com/tasl-lab/GenSafeNav>
* Local ignored checkout: `output/repos/GenSafeNav`
* Commit: `01baf926a5c77c1a4ab28635658eb014ef4f1767`
* Docker base image recorded by the source: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`
* Model: `trained_models/Ours_GST`
* Checkpoint: `trained_models/Ours_GST/checkpoints/05207.pt`

The checkout, model files, and generated probe logs remain ignored local artifacts. The compact
tracked evidence summary is
`docs/context/evidence/issue_1393_gensafenav_source_harness_2026-05-20/report.json`.

## Probe Command

```bash
git clone https://github.com/tasl-lab/GenSafeNav output/repos/GenSafeNav
git -C output/repos/GenSafeNav checkout 01baf926a5c77c1a4ab28635658eb014ef4f1767
git -C output/repos/GenSafeNav rev-parse HEAD
uv run python scripts/tools/probe_sonic_source_harness.py \
  --issue 1393 \
  --repo-remote-url https://github.com/tasl-lab/GenSafeNav \
  --repo-root output/repos/GenSafeNav \
  --model-name Ours_GST \
  --checkpoint 05207.pt \
  --timeout-seconds 30 \
  --output-json output/benchmarks/external/gensafenav_source_harness_probe_1393/report.json \
  --output-md output/benchmarks/external/gensafenav_source_harness_probe_1393/report.md
```

The probe exits non-zero when the source harness is blocked. For this run that non-zero exit is the
expected fail-closed result.

## Result

Verdict: `source harness blocked`.

Observed blocker:

```text
Traceback (most recent call last):
  File "output/repos/GenSafeNav/test.py", line 9, in <module>
    from rl.networks.envs import make_vec_envs
  File "output/repos/GenSafeNav/rl/networks/envs.py", line 3, in <module>
    import gym
ModuleNotFoundError: No module named 'gym'
```

The probe also extracted the source contract:

* `robot_policy`: `selfAttn_merge_srnn`
* `human_policy`: `orca`
* `robot_sensor`: `coordinates`
* `predict_method`: `inferred`
* `action_kinematics`: `holonomic`
* `env_use_wrapper`: `True`
* default env: `CrowdSimPredRealGST-v0`

## Field Classification

This run does not advance the ACI/conformal fields beyond the #1366 assessment because the source
entrypoint fails before an episode starts. The #1366 classification remains authoritative:

* current robot and tracked pedestrian state: deployment-observable under the declared observation
  level,
* GST-predicted pedestrian trajectories: deployment-derived predictions only if generated from
  current/past observations by a frozen predictor,
* `conformity_scores`: allowed only as train/validation-calibrated or online-past-only state,
* `conformal_intrusion_cost` and cost critic training state: training or diagnostic only,
* source `human_future_traj` truth and evaluation-seed calibration: forbidden as policy inputs.

## Verdict

Status: `source harness blocked by runtime dependency`.

The source harness has not reproduced a GenSafeNav `Ours_GST` checkpoint in the Robot SF runtime.
This keeps GenSafeNav / SoNIC out of benchmark-ready and source-faithful adapter status. Model-only
compatibility shims remain useful for metadata/prototype work, but they do not prove conformal,
constrained-RL, or source simulator semantics.

## Next Reopen Gate

Revisit source-harness promotion only after a dedicated source environment can run:

```bash
python test.py --model_dir trained_models/Ours_GST --test_model 05207.pt
```

without local fallback, model-only shims, evaluation-seed calibration, or future-trajectory policy
inputs. The next record should capture the pinned Python version, `gym`, `matplotlib`, Torch/CUDA
stack, source commit, checkpoint checksum, and whether the run reaches at least one source episode.
