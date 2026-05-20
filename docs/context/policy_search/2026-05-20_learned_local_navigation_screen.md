# Learned Local-Navigation Candidate Screen

Date: 2026-05-20

Related issue:

- Issue #1355: <https://github.com/ll7/robot_sf_ll7/issues/1355>

Related Robot SF anchors:

- `docs/context/policy_search/README.md`
- `docs/context/policy_search/candidate_registry.yaml`
- `docs/context/policy_search/reports/2026-05-05_best_learning_policy.md`
- `docs/context/policy_search/reports/2026-05-05_learning_hybrid_policy_search.md`
- `docs/benchmark_planner_family_coverage.md`
- `docs/context/issue_742_awesome_robot_social_navigation_mining.md`
- `docs/context/issue_601_crowdnav_feasibility_note.md`
- `docs/context/issue_600_dsrnn_stretch_follow_up.md`
- `docs/context/issue_907_social_jym_sarl_parity.md`
- `docs/context/issue_909_social_jym_policy_provenance.md`
- `docs/context/issue_1190_dreamerv3_checkpoint_import_boundary.md`
- `docs/context/issue_1246_observation_levels.md`
- `docs/context/issue_1247_safety_shield_contract.md`

## Goal

Screen learning-based local-navigation candidates against Robot SF's planner contract and policy
search funnel so future work starts from a current, deduplicated shortlist rather than novelty
alone. This note is assessment-only: it does not add a planner, train a policy, or make
paper-facing claims.

## Contract Used For Screening

Candidates are considered implementable only when they can plausibly reduce to:

- Robot SF planner-facing observation at time `t`, without future leakage or hidden privileged
  scene state at evaluation time;
- a mappable action or short local trajectory that can be executed through the benchmark command
  contract;
- fixed scenario/seed evaluation through episode JSONL;
- explicit source, license, checkpoint, and training lineage;
- fail-closed behavior when source assets, model weights, or dependencies are missing.

Methods that rely on RGB-D foundation-navigation stacks, full embodied-AI simulators, ROS/Gazebo,
or inaccessible trained-policy artifacts are not rejected as research. They are rejected or monitored
as current Robot SF benchmark rows until a smaller source-side reproduction or adapter contract is
proven.

## Ranked Top 10

| Rank | Candidate family | Verdict | Why |
| ---: | --- | --- | --- |
| 1 | ORCA-residual learned local policy | `implement after training-design gate` | Best follow-up to the failed inference-only guarded PPO pass; Issue #1358 already scopes the required training, lineage, and residual-bound work. |
| 2 | Tentabot-style motion-primitive value policy | `source-side assessment first` | Closest external learned method to Robot SF's rollout/scoring style; Issue #1357 should check whether it becomes a learned scorer rather than a ROS/Gazebo port. |
| 3 | PPO issue-791 best policy | `baseline, not safety promotion` | Strongest current learned-only Robot SF candidate, but worse collision rate than ORCA on the scoped paper matrix. |
| 4 | CrowdNav HEIGHT / IGAT-style graph policies | `source-side reproduction first` | Most concrete CrowdNav-lineage delta after the existing CrowdNav/SARL/DSRNN notes; keep checkpoint/source fidelity caveats explicit. |
| 5 | NavDP | `monitor only for Robot SF adapter` | Current and code-backed, but RGB-D diffusion navigation with privileged simulation training does not reduce cleanly to a 2D local planner without creating a new method. |
| 6 | NoMaD / ViNT / GNM visual navigation | `monitor only for Robot SF adapter` | MIT code/checkpoints exist, but topomap/visual-goal assumptions and cross-robot visual navigation are not Robot SF local-planner inputs. |
| 7 | DS-RNN / RGL / CrowdNav graph baselines | `already tracked, assessment only` | Valuable family anchors, but source-simulator state packing, old dependency stacks, and adapter-heavy semantics are already documented as non-benchmark-ready. |
| 8 | SafeCrowdNav / SoNIC / GenSafeNav safety-aware RL | `monitor or source-harness first` | Relevant safety framing, but Robot SF already separates internal guarded PPO from SoNIC-family claims and requires source/checkpoint proof. |
| 9 | DreamerNav / DreamerV3 world-model navigation | `monitor only` | Modern world-model navigation is interesting, but current Robot SF DreamerV3 checkpoint/import boundary is fail-closed and not a local-planner adapter. |
| 10 | SAC/TD3/PPO mapless baselines | `internal baseline lane only` | Useful for ablations and training research, but current SAC evidence is not benchmark-ready; TD3 needs its own source/config lineage before being a candidate. |

## Candidate Matrix

| Candidate | Source URL | Source status checked | Robot SF fit | Fairness or provenance risks | Smallest falsification experiment | Existing Robot SF anchor |
| --- | --- | --- | --- | --- | --- | --- |
| ORCA-residual learned policy | <https://github.com/ll7/robot_sf_ll7/issues/1358>, <https://github.com/ll7/robot_sf_ll7/blob/main/docs/context/policy_search/reports/2026-05-05_learning_hybrid_policy_search.md> | Prior ORCA-prior guarded PPO evidence exists in repo | `implement after training-design gate` | reward leakage, ORCA guard doing all useful work, durable checkpoint lineage, residual bounds | define ORCA command/risk observation extension and prove residual clipping plus guard diagnostics before any training run | `docs/context/policy_search/reports/2026-05-05_learning_hybrid_policy_search.md`, Issue #1358 |
| PPO issue-791 best policy | <https://github.com/ll7/robot_sf_ll7/blob/main/docs/context/policy_search/reports/2026-05-05_best_learning_policy.md>, <https://github.com/ll7/robot_sf_ll7/blob/main/model/registry.yaml> | Existing smoke, nominal, and paper-matrix comparison | `baseline, not safety promotion` | improves success but collision rate is worse than ORCA; predictive-foresight/model artifact provenance must remain explicit | rerun `ppo_issue791_best_v1` smoke/nominal with unchanged registry and compare against ORCA | `docs/context/policy_search/reports/2026-05-05_best_learning_policy.md` |
| Guarded PPO / ORCA-prior guarded PPO | <https://github.com/ll7/robot_sf_ll7/blob/main/docs/context/issue_602_guarded_ppo_profile.md>, <https://github.com/ll7/robot_sf_ll7/blob/main/docs/context/policy_search/reports/2026-05-05_learning_hybrid_policy_search.md> | Implemented and evaluated; promotion rejected | `reject current variants, revise only with training` | inference-only blending raised static collisions or reduced success; threshold tweaking is exhausted | require a trained residual or learned risk scorer; do not open another guard-threshold tweak | `docs/context/issue_602_guarded_ppo_profile.md`, `docs/context/policy_search/reports/2026-05-05_learning_hybrid_policy_search.md` |
| Tentabot motion-primitive value policy | <https://github.com/RIVeR-Lab/tentabot>, <https://arxiv.org/abs/2208.08034> | GitHub repo visible; no GitHub-detected license in metadata; paper advertises open-source implementation | `source-side assessment first` | ROS/Gazebo/3D sensing assumptions; a Robot SF rewrite could become a new method, not a faithful benchmark row | verify license and run a source-side example, then decide if Robot SF should implement a learned scorer over existing rollout candidates | Issue #1357 |
| NavDP | <https://github.com/InternRobotics/NavDP>, <https://arxiv.org/abs/2505.08712> | GitHub repo visible; no GitHub-detected license in metadata; paper describes RGB-D diffusion policy and privileged simulation guidance | `monitor only for adapter` | RGB-D/local observation tokens, privileged training targets, large scene dataset, trajectory follower dominates comparison | source-side demo only; reject Robot SF adapter unless a 2D observation-to-trajectory contract is explicit and weights/assets are accessible | Issue #1356 |
| NoMaD / ViNT / GNM | <https://github.com/robodhruv/visualnav-transformer>, <https://arxiv.org/abs/2310.07896> | GitHub metadata reports MIT license; official code and checkpoint release for visual navigation models | `monitor only for adapter` | visual goal/topomap/ROS-bag-style deployment assumptions, foundation-navigation dataset advantage, not a local social planner | verify checkpoint/demo path and document why replacing visual inputs with Robot SF state would be a new method | Issue #1356 |
| CrowdNav / SARL / social-jym | <https://github.com/vita-epfl/CrowdNav>, social-jym notes | CrowdNav metadata reports MIT; social-jym wrapper/parity work exists but trained SARL/SARL-PPO artifacts are absent | `already tracked, blocked for benchmark` | source simulator state packing, holonomic-to-unicycle projection loss, missing trained artifacts | fail closed until trained policy artifacts and source-policy quality parity are durable | `docs/context/issue_601_crowdnav_feasibility_note.md`, `docs/context/issue_907_social_jym_sarl_parity.md`, `docs/context/issue_909_social_jym_policy_provenance.md` |
| DS-RNN / RGL | <https://github.com/Shuijing725/CrowdNav_DSRNN>, <https://github.com/ChanganVR/RelationalGraphLearning> | DS-RNN metadata reports MIT and active repo updates; RGL repo visible but no GitHub-detected license metadata | `already tracked, assessment only` | graph/history source packing, legacy dependency stack, source action semantics need adapter proof | source-harness reproduction before any Robot SF wrapper; otherwise keep as family context | `docs/context/issue_600_dsrnn_stretch_follow_up.md`, `docs/benchmark_planner_family_coverage.md` |
| CrowdNav HEIGHT / IGAT | <https://github.com/Shuijing725/CrowdNav_HEIGHT>, <https://arxiv.org/abs/2203.01821> | HEIGHT metadata reports MIT; repo is current and has checkpoints per prior intake notes | `source-side reproduction first` | upstream graph/state construction and checkpoint fidelity; discrete action projection; no proven benchmark advantage over current wrapper | run the upstream `test.py`/checkpoint path in source harness and compare observation/action contract against `crowdnav_height` wrapper | `docs/context/issue_742_awesome_robot_social_navigation_mining.md`, `docs/context/760_model_shortcoming_hypothesis.md` |
| SafeCrowdNav / SoNIC / GenSafeNav | <https://github.com/Janet-xujing-1216/SafeCrowdNav>, <https://github.com/tasl-lab/SoNIC-Social-Nav>, <https://github.com/sepsamavi/safe-interactive-crowdnav> | SafeCrowdNav, SoNIC, and SICNav-family repos report MIT licenses in GitHub metadata | `monitor or source-harness first` | safety terminology can hide privileged prediction/training contracts; internal guarded PPO is not SoNIC-family equivalence | verify source-side checkpoint/demo and map the safety signal to Robot SF observation/action metadata before adapter work | `docs/context/issue_602_guarded_ppo_profile.md`, `docs/context/issue_601_crowdnav_feasibility_note.md` |
| DreamerNav / DreamerV3 navigation | <https://pmc.ncbi.nlm.nih.gov/articles/PMC12510832/>, <https://github.com/danijar/dreamerv3> | DreamerNav paper describes Isaac/photorealistic multimodal navigation; Robot SF DreamerV3 import boundary is fail-closed | `monitor only` | world-model checkpoint surgery, multimodal/RGB-D and occupancy-map assumptions, expensive training and artifact lineage | no adapter issue until a stable checkpoint import and Robot SF observation contract exist | `docs/context/issue_1190_dreamerv3_checkpoint_import_boundary.md`, `docs/context/dreamerv3_program_close_out_2026_04_30.md` |
| SAC / TD3 mapless RL | <https://github.com/ll7/robot_sf_ll7/blob/main/docs/context/issue_790_sac_benchmark_transfer_note.md>, <https://github.com/ll7/robot_sf_ll7/blob/main/docs/benchmark_planner_family_coverage.md> | SAC implementation exists but benchmark transfer remains poor; TD3 lane is not registered | `internal baseline lane only` | training-set success is not benchmark evidence; checkpoint promotion and observation transforms must be explicit | SAC/TD3 candidate must pass smoke and nominal policy-search stages before any family claim | `docs/context/issue_790_sac_benchmark_transfer_note.md`, `docs/benchmark_planner_family_coverage.md` |
| Recent review/family index | <https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1658643/full> | Review lists SARL, RGL, DS-RNN, CrowdNav descendants and benchmarking concerns | `source index only` | review breadth is not implementation evidence; individual source/checkpoint checks still required | use only to find candidates, then verify source repos and Robot SF contract one by one | Issue #1355 |

## Decisions

### Implement Now

No external learned-policy candidate should be added directly from this screening pass. The only
implementation-ready learned-policy direction is the already-scoped internal Issue #1358 residual-policy
training issue, and even that starts with a design/config/checkpoint-lineage gate rather than a
quick local code patch.

### Source-Side Reproduction First

- Tentabot-style motion-primitive value policies (Issue #1357).
- CrowdNav HEIGHT or IGAT-family graph policies if maintainers want another CrowdNav-lineage proof.
- SafeCrowdNav/SoNIC/SICNav-family safety-aware policies only after source checkpoints and safety
  signals are proven without fallback.

### Monitor Only

- NavDP and NoMaD diffusion/visual navigation until they expose a clean 2D local trajectory policy
  that can run without RGB-D/topomap/foundation-navigation assumptions.
- DreamerNav/DreamerV3 world-model navigation until Robot SF has a stable checkpoint import and
  observation contract.

### Reject For Now

- Any candidate whose only available path requires replacing Robot SF state observations with
  visual navigation inputs, importing a full external simulator, relying on missing checkpoints, or
  using privileged future/global state at evaluation time.
- Another inference-only guarded PPO threshold tweak. The existing learning-hybrid evidence already
  showed why this is not the next useful experiment.

## Follow-Up Routing

- Issue #1358 remains the right training-heavy follow-up for a bounded ORCA-residual learned policy.
- Issue #1357 remains the right assessment follow-up for motion-primitive value learning.
- Issue #1356 remains the right assessment follow-up for NavDP/NoMaD diffusion navigation, with the likely
  first outcome being monitor/reject unless source evidence proves a clean local-planner reduction.
- Do not create duplicate CrowdNav/SARL/DS-RNN/Dreamer/SAC issues without first updating the
  existing context notes linked above.

## Validation

Source and repo checks used:

```bash
gh repo view vita-epfl/CrowdNav --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt
gh repo view Shuijing725/CrowdNav_DSRNN --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt
gh repo view robodhruv/visualnav-transformer --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt
gh repo view RIVeR-Lab/tentabot --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt
gh repo view InternRobotics/NavDP --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt
gh repo view Janet-xujing-1216/SafeCrowdNav --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt
gh repo view sepsamavi/safe-interactive-crowdnav --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt
gh repo view ChanganVR/RelationalGraphLearning --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt
gh repo view Shuijing725/CrowdNav_HEIGHT --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt
gh repo view tasl-lab/SoNIC-Social-Nav --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt
```

Document validation for this issue should include:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```
