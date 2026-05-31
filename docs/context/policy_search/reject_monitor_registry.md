# Learned-Policy Reject And Monitor Registry

Date: 2026-05-20

This registry keeps negative and monitor-only learned-policy findings in one place so future
candidate-discovery work does not reopen the same lines by accident.

Use this file for rejected, deferred, monitor-only, and source-side-first learned-policy families.
Keep `candidate_registry.yaml` focused on implemented or concrete runnable Robot SF candidates with
config pointers. A method should move from this registry into `candidate_registry.yaml` only after
there is a bounded Robot SF candidate config, runnable command, and proof path.

Ownership boundary:

- `learned_policy_registry.md` owns the current planning state for each learned-policy family and
  should contain at most one current-state row per stable `policy_id`.
- This reject/monitor registry owns source-backed negative evidence, monitor-only rationale,
  historical deferrals, and reopen criteria.
- `candidate_registry.yaml` owns runnable Robot SF candidates. Entries here should not be treated as
  runnable or benchmark-ready unless they graduate to a concrete candidate config with proof.

Status vocabulary:

- `reject for now`: Do not implement without materially new evidence.
- `monitor only`: Track the family, but do not open implementation work yet.
- `source-side reproduction first`: Prove the upstream/runtime path before any Robot SF wrapper.
- `prototype only`: A wrapper or metadata surface may exist, but it is not benchmark-ready.
- `defer`: Revisit after a named prerequisite lands.

Each entry separates source-backed facts from Robot SF synthesis. Source-backed facts come from
linked upstream repositories, existing Robot SF context notes, or issue bodies. Robot SF synthesis is
the repository-specific decision about benchmark fit, fairness, and reopen criteria.

## Entries

### CrowdNav / SARL / RGL Base Family

- Source URL: https://github.com/vita-epfl/CrowdNav
- Current status: `source-side reproduction first`
- Evidence grade: `observed`
- Source-backed facts: CrowdNav is the canonical attention-based crowd-navigation source anchor;
  the upstream repository is MIT licensed and exposes source configs and test code, but public
  pretrained weights are not clearly bundled in the upstream checkout.
- Robot SF synthesis: Treat the family as a reference anchor, not current benchmark support. The
  source simulator controls observation packing, normalization, and policy semantics, and Robot SF
  would need an explicit observation/action adapter plus source-harness parity proof.
- Related Robot SF work: #601,
  `docs/context/issue_601_crowdnav_feasibility_note.md`,
  `docs/benchmark_planner_family_coverage.md`.
- Reopen if: one source-harness policy can be reproduced with durable checkpoint provenance and an
  adapter contract that maps observations and actions without silent fallback.

### DSRNN-Style Graph-Attention Family

- Source URL: https://github.com/Shuijing725/CrowdNav_DSRNN
- Current status: `defer`
- Evidence grade: `observed`
- Source-backed facts: The upstream DSRNN note records an MIT-licensed repository, visible source
  test path, source config surface, and advertised checkpoints.
- Robot SF synthesis: This is a roadmap family behind CrowdNav/SoNIC source-harness work. It adds
  graph and temporal-history reconstruction on top of the CrowdNav-style simulator contract, so it
  is not a small wrapper candidate.
- Related Robot SF work: #600,
  `docs/context/issue_600_dsrnn_stretch_follow_up.md`,
  `docs/benchmark_planner_family_coverage.md`.
- Reopen if: CrowdNav-lineage source-harness reproduction is proven first, then DSRNN runs one
  upstream checkpoint with documented graph/history inputs and Robot SF-compatible action mapping.

### HEIGHT And IGAT Attention Successors

- Source URLs:
  - https://github.com/Shuijing725/CrowdNav_HEIGHT
  - https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph
- Current status: `prototype only`
- Evidence grade: `observed`
- Source-backed facts: HEIGHT has an upstream repository, MIT license signal, source entrypoints,
  and published checkpoints. Issue #1394 cloned the HEIGHT source at
  `65451bcdd1f3fbebaf6e96a0de73aaa56d74ca05`; the source entrypoint remained blocked on
  `ModuleNotFoundError: No module named 'gym'`, and the advertised `237400.pt` checkpoint was not
  bundled in the checkout. IGAT has a public upstream repository and checkpoints.
- Robot SF synthesis: HEIGHT remains the current ceiling representative for this branch, but the
  wrapper path is adapter-heavy and experimental. IGAT is not a better first integration without a
  HEIGHT-vs-IGAT source-harness comparison.
- Related Robot SF work: #760, #770, #1367,
  Issue #1394,
  `docs/context/issue_742_awesome_robot_social_navigation_mining.md`,
  `docs/context/issue_770_igat_st2_attention_assessment.md`,
  `docs/context/policy_search/issue_1394_crowdnav_height_source_harness.md`,
  `docs/benchmark_planner_family_coverage.md`.
- Reopen if: a single attention successor demonstrates source-harness parity and materially better
  Robot SF fit than the existing HEIGHT prototype.

### ST2 Or Unpublished Attention Successors

- Source URL: no public source repository or checkpoint was found in
  `docs/context/issue_770_igat_st2_attention_assessment.md`.
- Current status: `reject for now`
- Evidence grade: `observed`
- Source-backed facts: The prior attention-successor assessment found no public ST2 implementation
  or checkpoint to run.
- Robot SF synthesis: Do not open implementation work for unpublished attention successors. They can
  remain literature context only until there is a source URL, checkpoint, and source-harness path.
- Related Robot SF work: #770, #1367,
  `docs/context/issue_770_igat_st2_attention_assessment.md`.
- Reopen if: a public implementation appears with checkpoint/provenance assets and a bounded
  source-harness command.

### SoNIC / GenSafeNav Safety-Aware CrowdNav Family

- Source URLs:
  - https://github.com/tasl-lab/SoNIC-Social-Nav
  - https://github.com/tasl-lab/GenSafeNav
- Current status: `prototype only`
- Evidence grade: `observed`
- Source-backed facts: SoNIC and GenSafeNav expose model/checkpoint assets that can be referenced by
  the Robot SF metadata layer. The source-harness probe found the upstream source environment
  blocked in the current Robot SF environment. Issue #1393 reran the GenSafeNav `Ours_GST`
  checkpoint path at commit `01baf926a5c77c1a4ab28635658eb014ef4f1767` and reproduced
  `ModuleNotFoundError: No module named 'gym'`; model-only reuse is possible through explicit
  compatibility shims.
- Robot SF synthesis: Keep these as model-only prototypes or source-side-first safety/OOD
  candidates. Do not claim SoNIC/GenSafeNav benchmark parity or conformal-safety support until
  calibration, future-trajectory, and source-harness boundaries are proven.
- Related Robot SF work: #601, #602, #626, #627, #1366, #1393,
  `docs/context/issue_626_sonic_source_harness_probe.md`,
  `docs/context/issue_627_sonic_wrapper_followup.md`,
  `docs/context/issue_602_guarded_ppo_profile.md`,
  `docs/context/policy_search/issue_1393_gensafenav_source_harness.md`.
- Reopen if: the source harness becomes reproducible in a pinned environment and any uncertainty or
  calibration fields are classified as train-only, deployment-observable, oracle-only, or forbidden.

### social-jym SARL / SARL-PPO

- Source URL: https://github.com/TommasoVandermeer/social-jym
- Current status: `source-side reproduction first`
- Evidence grade: `observed`
- Source-backed facts: The source harness can import the upstream package, reset a minimal source
  environment, and run a random SARL policy step. The wrapper spike proved one Robot SF step with
  source-shaped inputs, and the parity note proved one controlled SARL input match. The pinned
  checkout does not provide durable trained SARL/SARL-PPO artifacts for benchmark-quality policy
  claims.
- Robot SF synthesis: This is a source-side-only family until trained policy provenance and
  holonomic-to-unicycle action semantics are resolved. Random-policy or source-input parity is not
  benchmark evidence.
- Related Robot SF work: #729, #792, #905, #907, #909,
  `docs/context/issue_729_social_jym_assessment.md`,
  `docs/context/issue_909_social_jym_policy_provenance.md`,
  `docs/benchmark_planner_family_coverage.md`.
- Reopen if: durable trained policy artifacts are found, licensed, version-pinned, and loaded
  through a reproduced source command, with action projection losses explicitly bounded.

### Arena-Rosnav / Rosnav Learned Navigation Stack

- Source URL: https://github.com/Arena-Rosnav/arena-rosnav
- Current status: `source-side reproduction first`
- Evidence grade: `observed`
- Source-backed facts: Issue #1758 cloned the MIT-metadata `Arena-Rosnav/arena-rosnav` source at
  `5de9d38`. The checkout exposes ROS Noetic/Gazebo/Flatland launch, benchmark, training, and
  Rosnav action-node surfaces, plus `.repos` pins for many external Arena/planner repositories.
  Small source probes failed closed on missing `rl_utils` and `rospy`, and no `.zip`, `.pt`, `.pth`,
  `.onnx`, `best_model*`, or `last_model*` policy files were bundled in the shallow checkout.
- Robot SF synthesis: Arena-Rosnav is an ecosystem and benchmark-workflow reference, not a direct
  Robot SF learned local-policy candidate. Do not add it to the runnable candidate registry or
  open adapter work until the upstream workspace/container can run one named Rosnav agent with a
  durable checkpoint and an observation/action contract mapped to Issue #1618.
- Related Robot SF work: #1758, #1620, #1617, #1618,
  `docs/context/policy_search/issue_1758_arena_rosnav_source_assessment.md`.
- Reopen if: a source-side Arena command runs in a pinned workspace/container, a trained Rosnav
  checkpoint is available from a durable source, and the action/observation metadata can be
  expressed without ROS fallback or simulator-only hidden state.

### DreamerV3 / World-Model Navigation

- Source URLs:
  - https://github.com/danijar/dreamerv3
  - https://docs.ray.io/en/master/rllib/rllib-algorithms.html#dreamerv3
- Current status: `reject for now`
- Evidence grade: `observed`
- Source-backed facts: Robot SF implemented scenario-matrix training/evaluation parity surfaces for
  the RLlib DreamerV3 path, then closed the BR-08 program after probe/gate/full attempts produced no
  useful out-of-sample signal under the available resource budget.
- Robot SF synthesis: Do not spend more compute on flat-vector BR-08 DreamerV3. The current issue is
  not generic world models being bad; it is that this repository's DreamerV3 track failed the
  repeated signal and resource gates while PPO already gives stronger benchmark evidence.
- Related Robot SF work: #578, #608, #609, #782, #864, #1190,
  `docs/context/issue_578_608_609_dreamerv3_parity.md`,
  `docs/context/dreamerv3_program_close_out_2026_04_30.md`,
  `docs/context/issue_1190_dreamerv3_checkpoint_import_boundary.md`.
- Reopen if: a retained DreamerV3 checkpoint has a durable pointer, RLlib exposes a stable
  config-first import boundary, and a small gate comparison beats the current from-scratch stop
  condition without custom checkpoint surgery.

### DreamerNav-Style Multimodal World-Model Navigation

- Source URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12510832/
- Current status: `monitor only`
- Evidence grade: `inferred`
- Source-backed facts: DreamerNav-style work extends DreamerV3 with multimodal spatial perception,
  occupancy/depth inputs, hybrid planning, and curriculum training for indoor navigation.
- Robot SF synthesis: This is not a direct replacement for the retired BR-08 Robot SF DreamerV3
  track. It assumes richer perception and system-level navigation surfaces than the local AMV
  benchmark, so any future work should be a fresh reduction study, not a restart of the closed
  DreamerV3 program.
- Related Robot SF work: #578, #782, #864, #1190,
  `docs/context/dreamerv3_program_close_out_2026_04_30.md`,
  `docs/context/issue_1190_dreamerv3_checkpoint_import_boundary.md`.
- Reopen if: a public implementation and checkpoint become available and a small proof shows the
  method can be reduced to Robot SF's local planner contract without depth/visual assets or
  evaluation-seed leakage.

### SAC / TD3 Off-Policy Learned Navigation

- Source URL: https://github.com/DLR-RM/stable-baselines3
- Current status: `defer`
- Evidence grade: `observed`
- Source-backed facts: Robot SF has SAC train/eval contract work, SAC-native action handling, and a
  gate-sized checkpoint path. The benchmark planner-family matrix still treats SAC as experimental
  until a checkpoint passes quality gates and a benchmark config is added.
- Robot SF synthesis: SAC/TD3 are useful internal learned-policy research lines, but they should not
  be presented as benchmark-ready rows until the quality-gated checkpoint/config path exists. TD3
  remains lower priority than improving the SAC observation and curriculum path already exercised.
- Related Robot SF work: #790,
  `docs/context/issue_790_sac_benchmark_transfer_note.md`,
  `docs/benchmark_planner_family_coverage.md`.
- Reopen if: a SAC or TD3 checkpoint passes the same benchmark gate expected of PPO, with durable
  model provenance and a `configs/baselines/` entry.

### PPO, Guarded PPO, And ORCA-Prior PPO Variants

- Source URL: https://github.com/DLR-RM/stable-baselines3
- Current status: `monitor only`
- Evidence grade: `observed`
- Source-backed facts: `ppo_issue791_best_v1` is the strongest learning-only candidate tested in
  the 2026-05-05 pass, but it is not a safety promotion. ORCA-prior guarded PPO and related guarded
  variants ran through the policy-search funnel but did not pass promotion-quality safety/progress
  gates.
- Robot SF synthesis: Keep runnable PPO-derived candidates in `candidate_registry.yaml` only when
  they have config pointers and proof artifacts. Use this registry to remember that the current
  guarded/residual PPO variants are not winners and should not be reintroduced as new work without a
  new training-side hypothesis.
- Related Robot SF work: `docs/context/policy_search/reports/2026-05-05_best_learning_policy.md`,
  `docs/context/policy_search/reports/2026-05-05_learning_hybrid_policy_search.md`,
  `docs/context/policy_search/reports/2026-05-05_full_matrix_all_candidates_analysis.md`,
  `docs/context/policy_search/candidate_registry.yaml`.
- Reopen if: a new PPO-family candidate changes the training observation, residual target, or safety
  supervision rather than only retuning inference guards, and then passes the nominal gate before
  full-matrix escalation.

### DRL-VO

- Source URL: https://github.com/TempleRAIL/drl_vo_nav
- Current status: `prototype only`
- Evidence grade: `observed`
- Source-backed facts: Robot SF metadata records the upstream repo and commit
  `6d734b6e0df77fd4c4faa4649ca0fcb3e69cf835`; the current stack recognizes `drl_vo` as an
  experimental learning planner with explicit action projection into Robot SF `unicycle_vw`.
- Robot SF synthesis: DRL-VO is one of the closest external learned-hybrid candidates, but it needs
  a privileged-state audit before any main-table eligibility claim. Accurate pedestrian state,
  history, lidar, and subgoal fields must be classified before benchmark use.
- Related Robot SF work: #769, #775, #1364,
  `docs/context/issue_769_drl_vo_assessment.md`,
  `robot_sf/benchmark/algorithm_metadata.py`,
  `tests/benchmark/test_algorithm_metadata_contract.py`.
- Reopen if: issue #1364 proves the adapter uses only deployment-observable inputs, adds a smallest
  contract smoke, and records whether the result is main-table, oracle-only, or prototype-only.

### DR-MPC / Residual MPC Learned Social Navigation

- Source URL: https://github.com/James-R-Han/DR-MPC
- Current status: `source-side reproduction first`
- Evidence grade: `observed`
- Source-backed facts: Robot SF metadata tracks DR-MPC as an external residual-MPC policy surface
  with a local vendor reference and adapter boundary for structured robot/human state.
- Robot SF synthesis: DR-MPC is scientifically relevant, but it should stay source-side-first until
  the residual policy, MPC stack, dependency/runtime requirements, and observation privileges are
  proven in isolation.
- Related Robot SF work: `robot_sf/benchmark/algorithm_metadata.py`,
  `docs/benchmark_planner_family_coverage.md`.
- Reopen if: the source runtime can be reproduced from public assets, and the Robot SF adapter can
  run a contract smoke without private checkpoints, hidden solver assumptions, or oracle human
  trajectory fields.

### NavDP / NoMaD Diffusion And Visual Navigation

- Source URLs:
  - https://github.com/InternRobotics/NavDP
  - https://github.com/robodhruv/visualnav-transformer
- Current status: `monitor only`
- Evidence grade: `observed`
- Source-backed facts: The issue #1356 assessment records NavDP as an RGB-D conditioned navigation
  diffusion policy with privileged critic-value supervision during training, and NoMaD as a
  goal-masked diffusion visual-navigation stack with ROS bag/topomap and GPU-oriented tooling.
  Issue #1621 generalizes this to Diffusion Policy, Consistency Policy, Diffuser/LDP-style
  trajectory diffusion, and Robot SF-native state/lidar diffusion options.
- Robot SF synthesis: These methods are modern and worth tracking, but they are not fair Robot SF
  local-planner candidates until the visual/full-navigation assumptions can be reduced to a
  non-privileged 2D local action or trajectory contract without inventing a new method.
- Related Robot SF work: #1356, #1355,
  `docs/context/policy_search/2026-05-30_diffusion_policy_feasibility_issue_1621.md`,
  `docs/context/policy_search/README.md`.
- Reopen if: a source-side demo can run from public assets and the policy can be expressed as
  `observation_t -> action_t` or a short trajectory under Robot SF's planner contract without
  RGB-D, Habitat/Isaac assets, or privileged future information. For Robot SF-native diffusion,
  reopen only after a launch packet defines dataset splits, action trajectory schema, checkpoint
  provenance, latency target, and fail-closed missing-artifact behavior.

### NeuPAN Point-Obstacle Local Planning

- Source URL: https://github.com/hanruihua/NeuPAN
- Current status: `monitor only`
- Evidence grade: `proposal`
- Source-backed facts: The open assessment issue tracks NeuPAN as a modern model-based-learning /
  optimization-style point-obstacle planner with public code and a direct control-action shape.
- Robot SF synthesis: NeuPAN may be a useful reactive comparator, but it is not social-navigation
  evidence by itself. The point-obstacle abstraction, GPL licensing implications, and CPU control
  rate need a source-side assessment before any Robot SF adapter work.
- Related Robot SF work: #1368, #1355.
- Reopen if: pedestrians-as-point-obstacles is accepted only as a non-social comparator, licensing
  is resolved without vendoring risk, and a source-side smoke shows plausible control-rate runtime.

### SAGE / MPC-Transfer GNN Navigation

- Source URL: pending upstream repository verification in #1369; do not treat a SAGE source as
  usable until that issue records the exact paper/repo/checkpoint pointers.
- Current status: `source-side reproduction first`
- Evidence grade: `proposal`
- Source-backed facts: The open assessment issue frames SAGE as DRL plus heterogeneous graph
  observations and offline MPC experience transfer, with repository completeness still unknown.
- Robot SF synthesis: This is not an implementation target until public training/inference assets
  are verified. The main risk is opening a concept-only reimplementation that changes the scientific
  method instead of reproducing the released one.
- Related Robot SF work: #1369, #1365, #1355.
- Reopen if: #1369 identifies a public repository, license, checkpoints or inference script, and a
  bounded source-side smoke that can run without private MPC demonstrations.

### Tentabot-Style Motion-Primitive Value Policies

- Source URL: https://github.com/RIVeR-Lab/tentabot
- Current status: `monitor only`
- Evidence grade: `proposal`
- Source-backed facts: The open assessment issue records Tentabot as an occupancy-value
  motion-primitive family with an open-source ROS/Gazebo-oriented framework and a local-candidate
  scoring shape that is closer to Robot SF than full visual navigation.
- Robot SF synthesis: This is the most plausible learning component among the source-side-only
  candidates because it could become a learned scorer over existing rollout candidates. Do not port
  the ROS/Gazebo stack or train a policy until the source license, runtime, and 2D reduction are
  checked.
- Related Robot SF work: #1357, #1355,
  `docs/context/policy_search/candidate_registry.yaml`.
- Reopen if: the source-side path is reproducible without adding ROS/Gazebo as a Robot SF runtime
  dependency, and the method can be reduced to a learned scorer over existing Robot SF rollout
  candidates with a clear falsification experiment.

### DWA-RL / Learned Dynamic-Window Navigation

- Source URL: pending exact source verification in the 2026-05-20 candidate screen; do not treat
  DWA-RL as usable until a public repository, license, and inference asset path are recorded.
- Current status: `source-side reproduction first`
- Evidence grade: `proposal`
- Source-backed facts: The candidate screen identified DWA-RL-style work as a learned extension of
  dynamic-window local navigation rather than a ready Robot SF baseline.
- Robot SF synthesis: This family is only interesting if the learned component can be isolated from
  simulator-specific state and expressed as a local planner with deployment-observable inputs. A
  from-scratch reimplementation would not be comparable evidence for the original method.
- Related Robot SF work: #1359, #1355.
- Reopen if: a public source/checkpoint path is identified, the source-side demo runs from pinned
  assets, and the learned policy can be reduced to Robot SF's `observation_t -> action_t` or
  short-horizon trajectory contract without privileged future state.

### Generic Visual-Language / Object-Goal / Foundation-Model Navigation

- Source URL: family-level reject; representative systems vary by visual-language model,
  object-goal benchmark, or embodied foundation-model stack rather than one canonical source.
- Current status: `reject for now`
- Evidence grade: `observed`
- Source-backed facts: The 2026-05-20 screen explicitly called out generic visual-language
  navigation, object-goal navigation, foundation-model scoring policies, and full embodied visual
  navigation stacks as tempting but poorly bounded learned-policy follow-ups. Issue #1626 records a
  focused readiness pass over OpenVLA/Octo/RT-2-style manipulation VLAs, LM-Nav-style composed
  navigation, ViNT/NoMaD/NavDP visual navigation, and emerging navigation VLA project surfaces.
- Robot SF synthesis: Do not open local-planner implementation work for these families unless the
  method exposes a bounded local `observation_t -> action_t` policy under Robot SF scenario, seed,
  and metric contracts. Full embodied navigation stacks usually assume RGB/RGB-D observations,
  semantic goals, global mapping, language/object labels, or simulator assets that Robot SF's
  current AMV benchmark does not provide.
- Related Robot SF work: #1359, #1355,
  `docs/context/policy_search/2026-05-30_foundation_model_readiness_issue_1626.md`,
  `docs/context/policy_search/candidate_registry.yaml`.
- Reopen if: a specific public method has reproducible source assets and a narrow reduction proof
  showing that the policy can act only from Robot SF-available local observations without semantic
  oracle inputs, global-map leakage, or changing the benchmark task. If the policy needs language
  or visual inputs, reopen only after an observation-track contract defines image/depth/semantic
  payloads, task-language schema, topological-memory boundary, action adapter, and fail-closed
  missing-modality behavior.

## Maintenance Rules

- Before opening a new learned-policy issue, search this file and the linked prior context notes.
- Add a new entry when an assessment concludes `reject for now`, `monitor only`,
  `source-side reproduction first`, `prototype only`, or `defer`.
- Do not use this registry as evidence that a candidate is bad in general. The status is only about
  current Robot SF benchmark fit, source reproducibility, and fairness under this repository's
  contracts.
- When a candidate graduates, update this registry with the reason, then add the concrete runnable
  config pointer to `candidate_registry.yaml`.
