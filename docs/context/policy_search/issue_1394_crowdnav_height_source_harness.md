# Issue #1394 CrowdNav HEIGHT Source-Harness Proof

Date: 2026-05-20

Related issues:

* <https://github.com/ll7/robot_sf_ll7/issues/1394>
* [Issue #1367 CrowdNav-Family Learned-Policy Verdict](issue_1367_crowdnav_family_verdict.md)
* [Issue #601 CrowdNav Feasibility Note](../issue_601_crowdnav_feasibility_note.md)
* [Issue #770 IGAT / ST2 Attention-Based RL Assessment](../issue_770_igat_st2_attention_assessment.md)
* [Learned-Policy Eligibility Checklist](contracts/learned_local_policy_eligibility.md)

## Goal

Prove or fail closed one CrowdNav-lineage source-harness checkpoint before opening new Robot SF
adapter work. This note uses `CrowdNav_HEIGHT` because it is the current Robot SF family
representative, has an upstream source test path, and advertises pretrained model checkpoints.

## Source Snapshot

External checkout:

* Repository: <https://github.com/Shuijing725/CrowdNav_HEIGHT>
* Local ignored checkout: `output/repos/CrowdNav_HEIGHT`
* Commit: `65451bcdd1f3fbebaf6e96a0de73aaa56d74ca05`
* License: MIT
* README checkpoint pointer: Google Drive model-checkpoint folder
* Model directory tested: `trained_models/HEIGHT`
* Checkpoint tested: `237400.pt`

The cloned repository does not bundle the tested `.pt` checkpoint locally. The source entrypoint
fails before checkpoint loading because the current Robot SF environment does not provide the
legacy `gym` package expected by the upstream harness.

Tracked compact evidence:
`docs/context/evidence/issue_1394_crowdnav_height_source_harness_2026-05-20/report.json`.

## Probe Command

```bash
git clone https://github.com/Shuijing725/CrowdNav_HEIGHT output/repos/CrowdNav_HEIGHT
git -C output/repos/CrowdNav_HEIGHT checkout 65451bcdd1f3fbebaf6e96a0de73aaa56d74ca05
git -C output/repos/CrowdNav_HEIGHT rev-parse HEAD
uv run python scripts/tools/probe_crowdnav_height_source_harness.py \
  --issue 1394 \
  --repo-root output/repos/CrowdNav_HEIGHT \
  --model-dir trained_models/HEIGHT \
  --checkpoint 237400.pt \
  --timeout-seconds 30 \
  --output-json output/benchmarks/external/crowdnav_height_source_harness_probe_1394/report.json \
  --output-md output/benchmarks/external/crowdnav_height_source_harness_probe_1394/report.md
```

The probe exits non-zero when the source harness is blocked. For this run that non-zero exit is the
expected fail-closed result.

## Result

Verdict: `source harness blocked`.

Observed blocker:

```text
Traceback (most recent call last):
  File "<CrowdNav_HEIGHT checkout>/test.py", line 10, in <module>
    from training.networks.envs import make_vec_envs
  File "<CrowdNav_HEIGHT checkout>/training/networks/envs.py", line 3, in <module>
    import gym
ModuleNotFoundError: No module named 'gym'
```

The probe also records `checkpoint_status: missing_local_checkpoint` because
`trained_models/HEIGHT/checkpoints/237400.pt` is not present in the cloned repository. The upstream
README points users to an external Google Drive folder for checkpoints, so a future source-harness
pass must explicitly hydrate and checksum the checkpoint before claiming reproducibility.

## Source Contract

Best-effort config extraction from `crowd_nav/configs/config.py` recorded:

* `env_name`: `CrowdSim3DTbObs-v0`
* `scenario`: `circle_crossing`
* `mode`: `sim`
* `robot_policy`: `selfAttn_merge_srnn_lidar`
* `human_num`: `7`
* `static_obs`: `True`
* `action_space_kinematics`: `turtlebot`

Requirements in the source checkout include legacy `gym`, `tensorflow==2.11.0`,
`numpy<=1.23.1`, and `pybullet==3.2.6`.

## Eligibility Classification

The source entrypoint fails before an episode starts, so no source rollout or checkpoint inference
is proven.

Field classification from the source config and family notes:

* robot pose, velocity, goal, radius: deployment-observable in principle,
* current human pose/velocity/radius: deployment-observable under a tracked-agent observation level,
* static obstacle/lidar tokens: deployment-observable only when produced from declared Robot SF map
  or sensor observations,
* temporal graph/recurrent state: allowed only when updated from past observations inside the
  policy state,
* source simulator normalization, hidden reset semantics, and `turtlebot` action interpretation:
  adapter contract required before Robot SF benchmark use,
* any source test labels, future trajectory truth, or checkpoint-specific evaluation logs:
  diagnostics only, not policy input.

## Verdict

Status: `blocked by runtime and local checkpoint assets`.

This is enough to keep CrowdNav HEIGHT and adjacent CrowdNav-family descendants in
source-harness-first or prototype-only status. The existing Robot SF `crowdnav_height` wrapper
remains a model-only experimental representative, not source-harness parity evidence.

Do not open a new Robot SF adapter issue from #1394. Reopen only after a dedicated source
environment imports `gym`, hydrates a checksummed HEIGHT checkpoint, and runs at least one
source-side episode through `test.py` without fallback or Robot SF observation substitutions.
