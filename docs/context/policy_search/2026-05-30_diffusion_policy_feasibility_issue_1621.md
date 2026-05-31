# Diffusion-Policy Local-Navigation Feasibility - Issue #1621 - 2026-05-30

Related issue:

- Issue #1621: <https://github.com/ll7/robot_sf_ll7/issues/1621>

Related Robot SF context:

- `docs/context/policy_search/2026-05-20_navdp_nomad_diffusion_assessment.md`
- `docs/context/policy_search/2026-05-20_learned_local_navigation_screen.md`
- `docs/context/policy_search/contracts/learned_local_policy_eligibility.md`
- `docs/context/issue_1618_learned_policy_adapter_interface.md`
- `docs/context/policy_search/learned_policy_registry.md`
- `docs/context/policy_search/reject_monitor_registry.md`
- `docs/context/issue_691_benchmark_fallback_policy.md`

## Goal

Assess whether diffusion-based action or trajectory generation is viable as a Robot SF local
navigation policy family. This note generalizes the earlier NavDP/NoMaD-specific assessment from
issue #1356; it does not train a diffusion model, import visual datasets, add an adapter, or claim
benchmark success from any source-side demo.

## Contract Used

A Robot SF local planner candidate must reduce to a non-privileged
`observation_t -> action_t` or `observation_t -> short local trajectory_t` contract. The deployed
policy may use the current robot state, current goal or local subgoal, current visible pedestrian
and obstacle observations, bounded history ending at `t`, and declared map or lidar observations
available to the benchmark observation level.

The candidate is not adapter-ready if it requires future states, hidden scene labels, simulator
outcomes, visual or topological assets that define a different benchmark, private checkpoints, or a
follower/controller that performs the actual local-navigation work without being separated in the
diagnostics. Fallback and degraded execution remain caveats, not successful evidence.

## Candidate Matrix

| Candidate | Observation requirements | Output contract | Runtime cost | Source and checkpoint state | Robot SF fit | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| NavDP | RGB-D observation, point/image/no-goal task payloads, IsaacSim/IsaacLab scene assets, privileged training guidance | Planned trajectories executed by an MPC follower in the source benchmark | GPU-oriented visual diffusion stack plus asynchronous trajectory following | `InternRobotics/NavDP` is public; GitHub license metadata is absent; README gates the latest checkpoint behind a form | Not a fair 2D local social-planner adapter without replacing the observation space and evaluation harness | `monitor only` |
| NoMaD / ViNT / GNM | Camera-image context, visual goal or topological image map, ROS deployment assumptions | Diffusion-sampled waypoint/action chunks for visual navigation | GPU image model plus ROS/topomap deployment path | `robodhruv/visualnav-transformer` is MIT licensed and advertises checkpoint releases | Open and useful as visual-navigation context, but replacing images/topomaps with Robot SF state would create a new method | `monitor only` |
| Diffusion Policy | State or image observations for imitation-learning manipulation tasks | Receding-horizon action sequence | Multi-step denoising at inference; source provides state and vision notebooks | `real-stanford/diffusion_policy` is MIT licensed; task checkpoints/logs are for manipulation-style benchmarks, not Robot SF navigation | Strong policy-representation reference, but no external local-navigation checkpoint or adapter contract exists | `reject current adapter; use as design reference` |
| Consistency Policy | Same task-specific observation schema as the distilled diffusion teacher, commonly state or image visuomotor inputs | Single-step or few-step action sequence distilled from a diffusion teacher | Lower-latency than diffusion teacher; still needs a trained teacher/student pair | `Aaditya-Prasad/consistency-policy` is MIT licensed and provides training/deployment instructions | Useful latency idea for future Robot SF-native diffusion training, but not a standalone navigation policy | `monitor only` |
| Diffuser / trajectory diffusion planning | Offline RL trajectory data, environment state trajectories, reward or constraint guidance | Denoised state/action trajectories with receding-horizon execution | Iterative trajectory denoising; source examples center on offline RL tasks | `jannerm/diffuser` is MIT licensed; public code exists for the ICML 2022 planning setup | Conceptually close to trajectory generation, but current public checkpoints/examples do not define Robot SF local social-navigation inputs | `monitor only` |
| LDP-style local diffusion planner | Local robot-navigation observations plus global-path conditioning; paper-level source identified | Local trajectory/action sequence for collision avoidance | Diffusion planning cost must be checked in source before any adapter work | arXiv paper found for LDP; no public source repository was verified in this pass | Most on-topic navigation shape, but source/checkpoint absence blocks adapter work | `source only / monitor` |
| ComposableNav-style dynamic-navigation diffusion | Dynamic-environment observations plus instruction/motion-primitive conditioning | Composed trajectory distribution over motion primitives | Diffusion over trajectories, likely training-heavy | arXiv/project page found; no Robot SF-compatible checkpoint or local contract verified | Interesting future research lane, but outside current local-planner benchmark inputs | `monitor only` |
| Robot SF-native state/lidar diffusion policy | Robot SF state, map-local, or lidar observation levels only; no visual assets or privileged future fields | Velocity command or short local trajectory with raw/adapted/post-guard diagnostics | Unknown until a small training design exists; consistency distillation may be required for latency | No checkpoint or training dataset exists today | Plausible only as a new Robot SF training project, not an external import | `defer behind dataset and latency gate` |

## Source Checks

Observed with GitHub metadata and upstream README/API checks:

- NavDP: `InternRobotics/NavDP`, default branch `master`, pushed `2026-01-12`, no GitHub license
  metadata, checkpoint access via a form, RGB-D server/evaluation path, IsaacSim/IsaacLab benchmark
  and scene assets.
- NoMaD / ViNT / GNM: `robodhruv/visualnav-transformer`, default branch `main`, pushed
  `2024-09-15`, MIT license, official code and checkpoint release for visual navigation models.
- Diffusion Policy: `real-stanford/diffusion_policy`, default branch `main`, pushed
  `2024-12-24`, MIT license, source centered on visuomotor policy learning with state/vision
  manipulation examples.
- Consistency Policy: `Aaditya-Prasad/consistency-policy`, default branch `main`, pushed
  `2024-07-20`, MIT license, source describes distilling a diffusion teacher to single/few-step
  inference for visuomotor tasks.
- Diffuser: `jannerm/diffuser`, default branch `main`, pushed `2024-07-18`, MIT license, source
  for trajectory diffusion planning in offline RL-style control settings.

Web/source references checked:

- NavDP paper and source: <https://arxiv.org/abs/2505.08712>,
  <https://github.com/InternRobotics/NavDP>
- NoMaD paper and source: <https://arxiv.org/abs/2310.07896>,
  <https://github.com/robodhruv/visualnav-transformer>
- Diffusion Policy paper and source: <https://arxiv.org/abs/2303.04137>,
  <https://github.com/real-stanford/diffusion_policy>
- Consistency Policy paper and source: <https://arxiv.org/abs/2405.07503>,
  <https://github.com/Aaditya-Prasad/consistency-policy>
- Diffuser paper and source: <https://arxiv.org/abs/2205.09991>,
  <https://github.com/jannerm/diffuser>
- LDP paper: <https://arxiv.org/abs/2407.01950>
- ComposableNav project and paper: <https://amrl.cs.utexas.edu/ComposableNav/>,
  <https://arxiv.org/abs/2509.17941>

## Observation-Fit Assessment

State-only diffusion policies are technically compatible with Robot SF only if trained on Robot SF
or a source dataset with the same deployment-observable state contract. No such external checkpoint
was found. Importing a manipulation Diffusion Policy or Diffuser checkpoint would not answer the
local-navigation question.

Map-local or lidar-like diffusion planners are the most plausible Robot SF-native direction because
they can stay within existing observation levels and produce a bounded local trajectory. That path
requires a new dataset manifest, action/trajectory schema, latency budget, and first falsification
test before any training issue should start.

RGB-D, image-goal, and topological-map diffusion navigation remains incompatible with the current
2D local social-navigation benchmark as a direct adapter. The visual policy may be valid research,
but substituting Robot SF state tensors for the released image observations would be a new method.

## Recommendation

Do not implement an external diffusion-policy adapter now.

Recommended status:

- NavDP and NoMaD stay `monitor only` per issue #1356.
- General Diffusion Policy and Consistency Policy stay design references, not adapter candidates.
- Diffuser/LDP/ComposableNav-style trajectory diffusion stays monitor/source-first until a public
  source checkpoint and a Robot SF-compatible local observation contract are verified.
- A Robot SF-native state/lidar diffusion policy is plausible only after a separate launch packet
  defines the dataset, split, action trajectory schema, latency target, fail-closed missing-artifact
  behavior, and diagnostics required by the learned-policy eligibility checklist.

## Follow-Up Boundary

No adapter issue should be opened from this assessment alone.

The smallest useful future issue would be a launch-packet issue, not an implementation issue:

- policy/source: Robot SF-native state or lidar diffusion trajectory policy;
- adapter contract: `observation_t` from a declared Robot SF observation level to a short local
  trajectory or velocity command, with raw/adapted/post-guard diagnostics;
- durable inputs: tracked dataset manifest, train/validation/evaluation seed split, normalizer
  provenance, and checkpoint artifact plan;
- first falsification test: one smoke training or source-side inference command that must fail
  closed when the checkpoint, normalizer, observation level, or action schema is missing, plus a
  latency check showing the policy can run at the intended local-planner control rate.

Until those fields exist, diffusion policy should remain a monitored research family rather than a
runnable Robot SF benchmark candidate.

## Validation

Commands run for this note:

```bash
gh repo view InternRobotics/NavDP --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt,defaultBranchRef
gh repo view robodhruv/visualnav-transformer --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt,defaultBranchRef
gh repo view real-stanford/diffusion_policy --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt,defaultBranchRef
gh repo view Aaditya-Prasad/consistency-policy --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt,defaultBranchRef
gh repo view jannerm/diffuser --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt,defaultBranchRef
gh api repos/InternRobotics/NavDP/contents/README.md --jq .content | base64 -d
gh api repos/Aaditya-Prasad/consistency-policy/contents/README.md --jq .content | base64 -d
```

Final branch validation:

```bash
rg -n "diffusion|Diffusion|Consistency Policy|NavDP|NoMaD|LDP" docs/context/policy_search docs/context/README.md
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
