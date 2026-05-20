# NavDP And NoMaD Diffusion Navigation Assessment

Date: 2026-05-20

Related issue:

- Issue #1356: <https://github.com/ll7/robot_sf_ll7/issues/1356>

Related source references:

- NavDP paper: <https://arxiv.org/abs/2505.08712>
- NavDP repository: <https://github.com/InternRobotics/NavDP>
- NoMaD paper: <https://arxiv.org/abs/2310.07896>
- GNM/ViNT/NoMaD repository: <https://github.com/robodhruv/visualnav-transformer>

Related Robot SF anchors:

- `docs/context/policy_search/README.md`
- `docs/context/policy_search/2026-04-29_broad_policy_search.md`
- `docs/context/external_planner_reuse_checklist.md`
- `docs/context/issue_691_benchmark_fallback_policy.md`
- `docs/benchmark_planner_family_coverage.md`
- `docs/dev/observation_contract.md`
- `docs/context/issue_1246_observation_levels.md`
- `docs/context/issue_1247_safety_shield_contract.md`

## Goal

Decide whether NavDP or NoMaD-style diffusion navigation should move toward a Robot SF local-planner
adapter, source-side reproduction, monitor-only tracking, or rejection for now. This assessment does
not integrate either method, import assets, train a diffusion model, or make benchmark claims.

## Contract Question

Robot SF local-planner candidates need a fair `observation_t -> action_t` or
`observation_t -> short local trajectory_t` contract using planner-facing state available inside the
benchmark episode. The candidate must not depend on future/global privileged state at evaluation
time, external scene assets that define a different benchmark, hidden visual/topological navigation
assumptions, or a follower/controller that dominates the policy comparison.

This follows `docs/context/external_planner_reuse_checklist.md`: verify provenance, reproduce the
source harness before wrapping, capture observation/action contracts explicitly, and fail closed on
missing assets or incompatible kinematics. The planner-family coverage table currently lists
diffusion / transformer / multimodal social trajectory planners as `missing`, with no in-repo
benchmark-facing implementation.

## Source Checks

### NavDP

Repository metadata checked with:

```bash
gh repo view InternRobotics/NavDP --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt,defaultBranchRef
gh api repos/InternRobotics/NavDP/contents --jq '.[].name'
gh api repos/InternRobotics/NavDP/contents/baselines/navdp --jq '.[].name'
gh api repos/InternRobotics/NavDP/contents/baselines/navdp/navdp_server.py --jq .content | base64 -d
gh api repos/InternRobotics/NavDP/contents/baselines/navdp/requirements.txt --jq .content | base64 -d
```

Observed source facts:

- Repository: `InternRobotics/NavDP`, default branch `master`, pushed `2026-01-12`.
- GitHub license metadata: none detected; `gh api repos/InternRobotics/NavDP/license` returned
  `404`.
- Checkpoint access: README asks users to fill a form to access the latest model checkpoint.
- Runtime path: `baselines/navdp/navdp_server.py` starts a Flask server around `NavDP_Agent`.
- Inputs: server endpoints accept RGB images, depth images, optional image goals, and point/pixel
  goal payloads.
- Outputs: endpoints return selected trajectories plus candidate trajectories and values.
- Evaluation stack: top-level evaluation scripts import IsaacSim/IsaacLab, USD scene assets,
  camera sensors, and an MPC trajectory follower.
- Dependencies: `baselines/navdp/requirements.txt` includes Torch 2.2.2, Diffusers 0.33.1, Flask,
  Gradio, Open3D, OpenCV, and image/video dependencies.

Robot SF implication:

NavDP has a visible trajectory API, but the observed source-side path is not a clean Robot SF local
planner contract. It expects RGB-D camera observations, IsaacSim/IsaacLab scenes, external scene
assets, a GPU-oriented visual stack, checkpoint access outside the repo, and a trajectory follower.
Replacing the visual/RGB-D inputs with Robot SF 2D state or occupancy observations would be a new
method or retraining project, not a fair adapter of the published policy.

### NoMaD / ViNT / GNM

Repository metadata checked with:

```bash
gh repo view robodhruv/visualnav-transformer --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt,defaultBranchRef
gh api repos/robodhruv/visualnav-transformer/contents/README.md --jq .content | base64 -d
gh api repos/robodhruv/visualnav-transformer/contents/deployment/config/models.yaml --jq .content | base64 -d
gh api repos/robodhruv/visualnav-transformer/contents/train/train_environment.yml --jq .content | base64 -d
gh api repos/robodhruv/visualnav-transformer/contents/deployment/src/navigate.py --jq .content | base64 -d
gh api repos/robodhruv/visualnav-transformer/contents/deployment/src/explore.py --jq .content | base64 -d
```

Observed source facts:

- Repository: `robodhruv/visualnav-transformer`, default branch `main`, pushed `2024-09-15`.
- GitHub license metadata: MIT.
- README advertises official code and checkpoint release for GNM, ViNT, and NoMaD.
- Checkpoint path: README links Google Drive model weights; deployment config expects
  `deployment/model_weights/nomad.pth`.
- Training stack: conda environment with Python 3.8.5, CUDA-era Torch, ROS bag packages, Diffusers
  0.11.1, LMDB, and image-processing dependencies.
- Observation path: training data consists of temporally labeled camera images plus odometry; the
  deployment path consumes camera images from ROS topics.
- Navigation path: `deployment/src/navigate.py` loads a topological map of images, predicts
  distances to image nodes, samples diffusion trajectories, and publishes waypoints through ROS.
- Exploration path: `deployment/src/explore.py` uses camera-image context and NoMaD diffusion action
  samples without a Robot SF state-observation input.

Robot SF implication:

NoMaD has clearer open-source licensing and public checkpoint pointers than NavDP, but its runnable
contract is still visual-navigation/topomap-based. A Robot SF adapter would need to replace camera
image context, topomap images, ROS topics, and the waypoint controller with Robot SF state or map
features. That would be a new reduction, not a faithful benchmark row for the published method.

## Verdicts

| Candidate | Verdict | Reason |
| --- | --- | --- |
| NavDP | `monitor only` | Public source exists and the server returns trajectories, but license metadata is absent, checkpoint access is form-gated, and the source-side path depends on RGB-D, IsaacSim/IsaacLab, scene assets, and an MPC follower. |
| NoMaD | `monitor only` | MIT source and checkpoint pointers exist, but the method is a visual/topological navigation stack rather than a Robot SF local social-navigation policy. |

Neither candidate should receive a Robot SF adapter issue yet.

## Source-Side Smoke Decision

No source-side smoke was run in this issue.

That is intentional:

- NavDP requires checkpoint access plus IsaacSim/IsaacLab scene setup and asset downloads before the
  benchmark scripts can run.
- NoMaD requires downloading model weights and running a ROS/topomap/image-observation deployment
  stack that does not answer the Robot SF local-planner contract question by itself.

A source-side smoke would be useful only if maintainers explicitly want to validate upstream demo
health, not as evidence of Robot SF benchmark compatibility.

## Future Entry Criteria

Open a future adapter or source-reproduction issue only if at least one of these becomes true:

- NavDP exposes a documented non-visual or 2D local-observation API, with accessible checkpoint and
  explicit license terms.
- NoMaD exposes a minimal offline inference path that accepts image tensors plus a local goal and
  returns waypoints without ROS/topomap deployment, and the reduction to Robot SF observations is
  stated as a new method rather than a faithful published-policy adapter.
- A Robot SF-specific diffusion policy is proposed with tracked training data, no privileged
  evaluation leakage, fixed scenario/seed splits, and a durable model artifact plan.

## Conclusion

Diffusion navigation stays in the monitor bucket for Robot SF. NavDP and NoMaD are useful modern
references, but neither is currently benchmark-compatible as a fair local-planner candidate. The
right next action is to keep them referenced from learned-policy screening notes and avoid opening an
adapter PR until a clean source-side local-policy contract exists.
