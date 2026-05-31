# Issue #1758 Arena-Rosnav Source-Side Assessment - 2026-05-30

Date: 2026-05-30

Related issues:

- Issue #1758: <https://github.com/ll7/robot_sf_ll7/issues/1758>
- Issue #1620 external learned-policy ranking:
  <https://github.com/ll7/robot_sf_ll7/issues/1620>
- Issue #1617 local-planner repository survey:
  `docs/context/issue_1617_local_planner_repo_survey.md`
- Issue #1618 learned-policy adapter interface:
  `docs/context/issue_1618_learned_policy_adapter_interface.md`

## Goal

Assess Arena-Rosnav as a source-side learned-navigation benchmark stack before any Robot SF adapter,
planner import, or benchmark claim.

## Verdict

Current verdict: `source-side reproduction first`.

Arena-Rosnav is a useful external benchmark/workflow reference, but it is not a Robot SF planner
candidate today. The source checkout exposes ROS Noetic/Gazebo/Flatland launch, benchmark,
training, and Rosnav action-node surfaces, but the current Robot SF environment cannot run even the
smallest source entry points without Arena-specific ROS and workspace dependencies. The shallow
source checkout also did not include durable trained-policy files.

Do not open a Robot SF adapter issue until a source-side Arena workspace or container run proves a
specific Rosnav agent, checkpoint path, observation/action contract, and benchmark command.

## Source Metadata

Primary source checked:

- Repository: <https://github.com/Arena-Rosnav/arena-rosnav>
- Checked commit: `5de9d38` on `master`
- GitHub metadata on 2026-05-30:
  - license: MIT
  - latest release: `v0.2.2`, published 2023-02-16
  - topics: ROS, Python, benchmarking, robotics, simulation, DRL, PPO, PyTorch, navigation
- Alternate historical/source family found by search:
  - <https://github.com/ignc-research/arena-rosnav>, GPL-2.0 metadata, default branch
    `local_planner_subgoalmode`

The `Arena-Rosnav/arena-rosnav` README points to the external installation docs and does not by
itself document a complete source-side benchmark command.

## Dependency And Runtime Shape

Arena-Rosnav is a multi-repository ROS workspace, not a standalone Python policy package.

Observed source surfaces:

- `.repos` pins additional Arena, simulator, message, planner, and third-party repositories,
  including `Arena-Rosnav/rosnav-rl`, `Arena-Rosnav/crowdnav-ros`,
  `Arena-Rosnav/sarl-star`, `Arena-Rosnav/trail`, `Arena-Rosnav/applr`,
  `Arena-Rosnav/flatland`, `pedsim_ros`, and `move_base_flex`.
- `Dockerfile` targets Ubuntu 20.04, ROS Noetic, Gazebo 11, catkin tooling, Poetry, Flatland, and
  ROS package dependencies.
- `pyproject.toml` uses Python 3.8-era Arena dependencies plus optional planner/training groups.
  The planner group includes PyTorch, Stable-Baselines3-era dependencies, legacy `gym`, MPI, and
  OpenCV; the training group references `rosnav_rl` from `../../planners/rosnav`.
- `arena_bringup/configs/benchmark/contests/allplanners.yaml` includes `rosnav`, `applr`, `trail`,
  and other local planners alongside `teb`, `dwa`, `dragon`, and `cohan`.
- `arena_bringup/configs/training/sb_training_config.yaml` describes Stable-Baselines3 PPO
  training, continuous action support, reward shaping, curriculum, W&B logging, and periodic
  evaluation.

This dependency shape is too broad for a direct Robot SF adapter. A future reproduction should use
the upstream container/workspace path or a deliberately isolated source harness, not install ROS
Noetic/Gazebo into the Robot SF virtual environment.

## Checkpoint And Policy Availability

No durable trained-policy files were found in the shallow source checkout:

```bash
find output/repos/arena-rosnav -type f \
  \( -name '*.zip' -o -name '*.pt' -o -name '*.pth' -o -name '*.onnx' \
     -o -name 'best_model*' -o -name 'last_model*' \) -print
```

The command returned no files.

Training configs refer to checkpoint names or machine-local paths, for example `last_model`,
`best_model`, and paths under `/home/tar/catkin_ws/src/planners/rosnav/agents/...`. Those are not
durable Robot SF artifacts and cannot support a benchmark row.

## Observation And Action Fit

Robot SF adapter fit is partial and unproven:

- The testing action node calls the `rosnav_rl/get_action` service and publishes ROS
  `geometry_msgs/Twist` on `cmd_vel`.
- `_publish_action(...)` treats the returned action as `[linear.x, linear.y, angular.z]`.
- Training configs include laser, full-range laser, optional RGB-D, discrete/custom discretized
  action spaces, subgoal mode, reward shaping, and curriculum settings.
- The benchmark stack composes local planners with intermediate planners and simulator-specific
  task generation.

This is not yet Robot SF's learned local-policy contract. A future adapter would need explicit
metadata for observation keys, time alignment, LaserScan/RGB-D/subgoal availability, action
projection into Robot SF kinematics, checkpoint normalization, raw/adapted/post-guard diagnostics,
and fail-closed behavior when ROS services or model files are absent.

## Source-Side Smoke

The source was cloned only into ignored local output:

```bash
gh repo clone Arena-Rosnav/arena-rosnav output/repos/arena-rosnav -- --depth 1
git -C output/repos/arena-rosnav rev-parse --short HEAD
```

Result: `5de9d38`.

Smallest training entry-point probe:

```bash
timeout 30s .venv/bin/python output/repos/arena-rosnav/training/scripts/train_agent.py --help
```

Result:

```text
ModuleNotFoundError: No module named 'rl_utils'
```

Smallest action-node probe:

```bash
timeout 30s .venv/bin/python output/repos/arena-rosnav/testing/scripts/drl_agent_node.py --help
```

Result:

```text
ModuleNotFoundError: No module named 'rospy'
```

These are expected fail-closed blockers in a Robot SF worktree. They show that source-side
reproduction needs the Arena ROS workspace, pinned external repos, and Rosnav dependencies before
policy behavior can be evaluated.

These failed probes are the validation proof for the `source-side reproduction first` verdict:
the source exists and exposes relevant learned-navigation surfaces, but the current checkout does
not satisfy the runtime dependency or checkpoint contract needed for Robot SF adapter work.

## Robot SF Routing

Keep Arena-Rosnav in the learned-policy reject/monitor registry as `source-side reproduction first`.
Do not add it to `docs/context/policy_search/candidate_registry.yaml`.

Candidate-registry absence check:

```bash
rg -n "arena|rosnav" docs/context/policy_search/candidate_registry.yaml
```

Result: no matches.

Issue lifecycle: close Issue #1758 with this assessment after the docs PR lands. Future work should
open a new, narrower source-harness issue only when there is a planned Arena workspace/container
reproduction path.

Reopen for adapter work only if a follow-up proves all of:

- exact upstream branch/commit and installed external repos,
- a runnable source-side command in the Arena workspace or container,
- a specific trained Rosnav agent and durable checkpoint path,
- observation/action metadata mapped to Issue #1618,
- and a fail-closed Robot SF plan that does not count ROS fallback, missing checkpoints, or
  simulator-only behavior as benchmark success.

## Validation

Commands used:

```bash
gh repo view Arena-Rosnav/arena-rosnav --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef,repositoryTopics,latestRelease
gh repo view ignc-research/arena-rosnav --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef,repositoryTopics,latestRelease
gh search repos "Arena Rosnav" --limit 10 --json fullName,url,description,license,updatedAt,stargazersCount
gh repo clone Arena-Rosnav/arena-rosnav output/repos/arena-rosnav -- --depth 1
timeout 30s .venv/bin/python output/repos/arena-rosnav/training/scripts/train_agent.py --help
timeout 30s .venv/bin/python output/repos/arena-rosnav/testing/scripts/drl_agent_node.py --help
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```

Docs validation result for this branch:

```text
OK docs/proof consistency check passed for 5 changed file(s).
```
