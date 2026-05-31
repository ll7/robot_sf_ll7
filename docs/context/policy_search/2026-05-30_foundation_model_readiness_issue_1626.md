# Foundation-Model Local-Navigation Readiness - Issue #1626 - 2026-05-30

Related issue:

- Issue #1626: <https://github.com/ll7/robot_sf_ll7/issues/1626>

Related Robot SF context:

- `docs/dev/observation_contract.md`
- `docs/context/issue_1246_observation_levels.md`
- `docs/context/issue_1612_observation_track_architecture.md`
- `docs/context/issue_1613_lidar_observation_track.md`
- `docs/context/issue_1618_learned_policy_adapter_interface.md`
- `docs/context/policy_search/contracts/learned_local_policy_eligibility.md`
- `docs/context/issue_691_benchmark_fallback_policy.md`
- `docs/context/policy_search/reject_monitor_registry.md`

## Goal

Assess whether Robot SF is ready to support foundation-model, VLA, or multimodal navigation
policies as fair local-navigation benchmark candidates. This is a readiness boundary only: it does
not integrate a model, download weights, add language tasks, import visual assets, or claim that
foundation-model branding is publication evidence.

## Representative Policy Families Checked

| Family | Typical observation requirements | Typical action/task output | Source and checkpoint state | Robot SF fit today | Readiness verdict |
| --- | --- | --- | --- | --- | --- |
| OpenVLA-style manipulation VLA | RGB image plus language instruction; embodiment-specific action normalization | 7-DoF manipulation action or action tokens | `openvla/openvla` is MIT-licensed code; released pretrained models inherit Llama-family license constraints | Not a local-navigation policy and no Robot SF action space or task schema match | `reject for benchmark; design reference only` |
| Octo-style generalist robot policy | Image observations and task definitions over Open X-Embodiment manipulation trajectories | Transformer/diffusion action chunks for manipulation embodiments | `octo-models/octo` is MIT-licensed code with open weights/project resources | Useful open generalist-policy pattern, but source tasks/actions are not Robot SF AMV local navigation | `monitor only` |
| RT-2 / closed large VLA systems | RGB image, language instruction, web-scale VLM pretraining, robot action fine-tuning | Low-level robot actions represented through model tokens | Paper public; practical source/checkpoints are not a Robot SF-ready open adapter path | Closed/proprietary path and manipulation/task mismatch block benchmark use | `reject current adapter` |
| LM-Nav-style language navigation | Natural-language instruction, image-language association, topological/visual navigation memory, goal-conditioned visual navigation policy | High-level waypoint/subgoal selection through a navigation graph or visual policy | Paper/project page public; system composes pretrained language, image, and navigation components | Relevant as architecture, but Robot SF lacks language goals, visual/topological memory, and source-compatible image observations | `monitor/source-first` |
| ViNT / GNM / NoMaD visual navigation | Camera-image context, visual goal or topological image map, sometimes diffusion action sampling | Waypoints or action chunks for visual navigation/exploration | `robodhruv/visualnav-transformer` is MIT licensed and advertises checkpoints | Already assessed as monitor-only; topomap/image-goal assumptions do not reduce to current 2D local social navigation | `monitor only` |
| NavDP / InternVLA-N1 navigation stack | RGB-D observation, goal payloads, IsaacSim/IsaacLab scene assets, optional VLA-style dual-system framing | Planned trajectories executed by a follower/controller | `InternRobotics/NavDP` public repo; no GitHub license metadata; latest checkpoint access is form-gated | Modern navigation reference, but RGB-D/scene/follower/checkpoint boundaries block direct local-planner adapter | `monitor only` |
| Navigation VLA systems such as NaVILA/VAMOS | Visual stream, language/task command, semantic reasoning, embodiment grounding, often global or hierarchical navigation | High-level semantic plan plus embodied local commands or trajectories | Project/paper surfaces found; no Robot SF-compatible source/checkpoint contract verified in this pass | Scientifically on-topic, but requires task-language, visual, semantic, and embodiment contracts absent from current benchmark | `monitor only` |

Sources checked include:

- OpenVLA paper/source: <https://arxiv.org/abs/2406.09246>, <https://github.com/openvla/openvla>
- Octo paper/project/source: <https://arxiv.org/abs/2405.12213>,
  <https://octo-models.github.io/>, <https://github.com/octo-models/octo>
- RT-2 paper: <https://arxiv.org/abs/2307.15818>
- LM-Nav paper/project: <https://arxiv.org/abs/2207.04429>,
  <https://sites.google.com/view/lmnav/home>
- ViNT/NoMaD source: <https://github.com/robodhruv/visualnav-transformer>
- NavDP source: <https://github.com/InternRobotics/NavDP>
- NaVILA project: <https://navila-bot.github.io/>
- VAMOS project: <https://vamos-vla.github.io/>

## Current Robot SF Interfaces

Robot SF already has several pieces that make a future multimodal benchmark track possible:

- `docs/dev/observation_contract.md` defines default `drive_state`/`rays`, SocNav structured
  state, occupancy-grid augmentation, and benchmark observation levels.
- `robot_sf/benchmark/observation_levels.py` separates perception assumptions from raw
  observation modes and currently supports `oracle_full_state`, tracked-agent levels, `lidar_2d`,
  and `occluded_partial_state`.
- `docs/context/issue_1612_observation_track_architecture.md` defines track metadata so evidence
  from different observation contracts is not averaged together.
- `docs/context/issue_1613_lidar_observation_track.md` provides a concrete LiDAR track smoke
  boundary with `drive_state` and `rays` only.
- `robot_sf/sensor/image_sensor.py` and `robot_sf/sensor/image_sensor_fusion.py` can capture a
  pygame-rendered RGB frame and include an optional current-frame `image` key in sensor fusion.
- `tests/test_image_sensor.py`, `tests/test_image_sensor_fusion.py`, and
  `tests/baselines/test_ppo_planner.py::test_build_model_obs_image_requires_payload` verify the
  local image-observation plumbing.
- `configs/scenarios/README.md` supports `platform_semantics`, but those regions are currently
  scenario-side metadata only and are not consumed by planners or metrics.
- `docs/context/issue_1618_learned_policy_adapter_interface.md` and the learned-policy eligibility
  checklist require explicit observation/action schemas, artifact provenance, raw/adapted/guarded
  action diagnostics, and fail-closed behavior for missing checkpoints or unsupported modalities.

These are readiness ingredients, not a VLA benchmark track.

## Missing Abstractions

### Observation

- No first-class benchmark observation level exists for RGB, RGB-D, image-goal, semantic-map,
  topological-memory, or language-conditioned observations.
- The pygame image sensor captures rendered simulator frames. It is useful for local RL plumbing,
  but it is not a camera-calibrated perception benchmark and does not carry camera intrinsics,
  depth, segmentation, detector provenance, or sim-to-real claims.
- Image observations are not wired into benchmark track metadata, candidate registries, or learned
  policy artifact manifests as a supported evidence lane.
- No topological memory contract records image nodes, graph edges, localization confidence, or
  whether a policy may access global route history at evaluation time.
- No semantic-map or object-label contract defines which labels are scenario-authored,
  detector-derived, training-only, or forbidden at evaluation time.

### Task And Language

- Current benchmark tasks are route/goal navigation tasks, not natural-language instruction tasks.
- No schema exists for language goals, object-goal queries, instruction decomposition, allowed
  vocabulary, ambiguity handling, or prompt/version provenance.
- There is no benchmark rule for whether an LLM/VLM may use external world knowledge, web-scale
  priors, chain-of-thought-like latent reasoning, or scenario names during evaluation.
- Scenario `platform_semantics` are metadata-only. They do not yet define language task targets or
  planner-visible semantic objectives.

### Action And Adapter

- Robot SF benchmark actions are local planner commands or short trajectories with explicit
  kinematics and projection semantics.
- Most VLA/manipulation policies output embodiment-specific continuous actions or tokenized robot
  actions that are not compatible with Robot SF differential-drive or AMV command contracts.
- Navigation foundation stacks often output high-level subgoals, topological graph decisions, or
  trajectories executed by a follower/controller. A fair adapter must log the model output,
  adapted command, post-guard command, follower/controller role, and fallback reason separately.

### Benchmark And Artifacts

- No foundation-model candidate has a Robot SF model registry entry, durable checkpoint manifest,
  normalizer/processor provenance, accepted license boundary, or benchmark-track config.
- No local smoke proves a VLA-style model can run from Robot SF observations and fail closed when
  unsupported modalities, weights, processors, or action schemas are missing.
- Existing image and LiDAR tests cover plumbing and contract smoke, not foundation-model
  benchmark readiness.

## Recommendation

Do not integrate a foundation model or VLA policy now.

Recommended status:

- Near-term feasible work: define interfaces and benchmark contracts only when a concrete future
  candidate needs them. The safest first interface issue would be an `image_language_nav_v1`
  observation-track design, not model integration.
- Monitor-only work: OpenVLA/Octo/RT-2-style manipulation VLAs, LM-Nav-style composed navigation,
  ViNT/NoMaD/NavDP visual navigation, and emerging navigation VLA systems such as NaVILA/VAMOS.
- Reject for benchmark today: any candidate that cannot run from Robot SF-declared
  `observation_t`, requires RGB-D/topological/language assets absent from the scenario contract,
  uses closed or form-gated checkpoints without a durable artifact plan, or hides the local
  controller/follower behind a foundation-model label.

## Fail-Closed Policy

A future multimodal or VLA benchmark row must fail before episodes are written when any of these
conditions holds:

- requested observation level is unsupported, for example `rgb_language_v1` before such a track is
  defined;
- required image, depth, language, semantic-map, topological-memory, tokenizer, processor,
  checkpoint, normalizer, or action adapter is missing;
- the model asks for simulator state, future trajectories, scenario outcome labels, hidden route
  metadata, or unbounded global maps outside the declared track;
- model output cannot be adapted into the declared Robot SF action family with explicit
  raw/adapted/post-guard diagnostics;
- a fallback controller, follower, or classical planner produces the final command without the row
  being marked `fallback`, `degraded`, or `not_available`.

Fallback or degraded execution may be useful for diagnostics, but must not count as successful
benchmark evidence.

## Follow-Up Boundary

Any proposed follow-up should be an interface or benchmark-contract task by default. A good first
child issue would name:

- track: `image_language_nav_v1` or another concrete observation-track slug;
- observation schema: exact image shape, frame timing, optional depth/semantic fields, language
  task payload, and allowed history/topological memory;
- action schema: velocity command, waypoint, short trajectory, or high-level subgoal plus the
  follower/controller boundary;
- provenance: model/source/license/checkpoint or explicit `no checkpoint yet` status;
- falsification test: a smoke command that fails closed when the model requests an unsupported
  modality or produces an incompatible action.

Do not open a generic "add VLA baseline" implementation issue until that contract exists.

## Validation

Local interface checks:

```bash
rg -n "image|rgb|camera|sensor|lidar|language|semantic|topological|foundation|VLA|vision-language|multimodal|observation level|observation_mode|Observation" docs robot_sf configs scripts tests -g '*.md' -g '*.py' -g '*.yaml'
rg -n "observation_level|lidar_2d|camera|image|semantic|language|task_description|PlannerObservation|observation_spec" robot_sf docs/dev docs/context configs tests -g '*.py' -g '*.md' -g '*.yaml'
```

Upstream/source checks:

```bash
gh repo view openvla/openvla --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt,defaultBranchRef
gh repo view octo-models/octo --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt,defaultBranchRef
gh repo view robodhruv/visualnav-transformer --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt,defaultBranchRef
gh repo view InternRobotics/NavDP --json nameWithOwner,url,description,licenseInfo,stargazerCount,pushedAt,defaultBranchRef
```

Final branch validation:

```bash
rg -n "foundation|VLA|vision-language|image_language_nav|rgb_language|fail closed|fail-closed" docs/context docs/dev robot_sf tests
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
