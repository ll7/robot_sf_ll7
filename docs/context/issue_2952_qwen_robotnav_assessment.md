# Issue #2952 Qwen-RobotNav Feasibility Assessment

Issue: [#2952](https://github.com/ll7/robot_sf_ll7/issues/2952)

Date: 2026-06-23

Status: assessment-only, `evidence_tier: idea`. This note is a preflight feasibility/decision document.
It does **not** add a Robot SF wrapper, benchmark row, dependency, or downloaded weights. No model was
installed, imported, or executed; all upstream facts come from public web sources fetched on the date
above and are marked verified or unverified accordingly.

## Claim Boundary

- `evidence_tier: idea`; exploratory feasibility only.
- No availability claim beyond what the cited sources actually showed. Where a source rendered as
  marketing copy, returned JavaScript-only content, or did not state a fact, the fact is recorded as
  **unverified** rather than guessed.
- No compatibility, runnability, or benchmark-readiness claim. The interface comparison below is a
  paper-vs-code surface comparison, not a tested adapter.
- Uncertainty on the central availability question (are weights/code/license actually public) is high
  (well above the ~5% bar for a confident claim); see "Availability Audit" for why.

## Decision

Pick exactly one next step: **`blocked_asset_tracking`**.

Rationale (2-3 lines): The Qwen-RobotNav technical report is real and dated 2026-06-17
([arXiv 2606.18112](https://arxiv.org/abs/2606.18112)), but I could **not** verify public, downloadable
weights, a concrete code repository, or a license that permits local benchmark use. One secondary news
article's comparison table claims "code: Yes (GitHub)," while the HuggingFace paper page shows no
weights, no code link, and no license, and a `QwenLM` org search for "robot" returned zero repositories.
Until the asset/license question is resolved, the honest next step is to track the blocking assets, not
to spike an adapter or launch a benchmark preflight.

The other three options were rejected:

- `no_action`: rejected because the model targets exactly Robot SF's research surface (local/social
  navigation) and is recent and high-profile; it is worth re-checking once assets land.
- `smoke_adapter_spike`: rejected because a spike presupposes a runnable artifact and a license that
  permits local use; neither is verified. A spike now would be speculative scaffolding against an
  unconfirmed interface.
- `benchmark_preflight_launch_packet`: rejected for the same reason, more strongly — a benchmark
  preflight implies near-integration readiness, which is two unverified prerequisites away.

## Availability Audit

Sources fetched (web research; all on 2026-06-23):

- Qwen-RobotNav technical report (arXiv abstract):
  <https://arxiv.org/abs/2606.18112>
- Qwen-RobotNav report PDF (search-indexed title, full PDF body **unreachable** — see below):
  <https://qianwen-res.oss-accelerate.aliyuncs.com/qwenrobot/papers/Qwen_RobotNav.pdf>
- Qwen-RobotNav blog: <https://qwen.ai/blog?id=qwen-robotnav>
  (**unreachable as content**: WebFetch returned only the literal word "Qwen"; the page is
  JavaScript-rendered and did not yield body text.)
- Qwen-RobotSuite blog: <https://qwen.ai/blog?id=qwen-robotsuite>
  (**unreachable as content**: same JS-only rendering, returned only "Qwen".)
- HuggingFace paper page: <https://huggingface.co/papers/2606.18112>
- MarkTechPost coverage:
  <https://www.marktechpost.com/2026/06/16/meet-qwen-robotsuite-three-embodied-ai-models-for-vla-manipulation-video-world-modeling-and-navigation/>
- TechNode coverage:
  <https://technode.com/2026/06/17/alibaba-unveils-qwen-robot-series-with-three-foundation-models-for-embodied-ai/>
- The AI Insider coverage:
  <https://theaiinsider.tech/2026/06/18/alibaba-releases-qwen-for-robotics-with-suite-for-navigation-manipulation-and-world-modeling/>
- HuggingFace search for `Qwen-RobotNav` weights: no matching model checkpoint surfaced
  (general Qwen LLM/VL models only).
- `QwenLM` GitHub org search `q=robot`:
  <https://github.com/orgs/QwenLM/repositories?q=robot> returned **0 repositories**.
- Adjacent repo found: <https://github.com/QwenLM/Qwen-VLA> (a VLA generalist on Qwen3.5-4B unifying
  manipulation/navigation/trajectory). It is **not** confirmed to be the RobotNav model; the report
  states a Qwen3-VL base at 2B/4B/8B, while Qwen-VLA states a Qwen3.5-4B base. Treated as a separate,
  unverified-as-RobotNav artifact.

| Audit dimension | Finding | Verified? | Source |
| --- | --- | --- | --- |
| Model exists / report published | Technical report "Qwen-RobotNav: A Scalable Navigation Model Designed for an Agentic Navigation System," dated 2026-06-17. | Verified (exists) | arXiv 2606.18112 |
| Model sizes | 2B / 4B / 8B variants; favourable scaling 2B→8B. | Verified (claimed consistently across arXiv abstract + MarkTechPost) | arXiv 2606.18112; MarkTechPost |
| Base model | Built on Qwen3-VL (per coverage); report frames it as a multimodal foundation. | Partially verified (secondary source) | MarkTechPost |
| Weights publicly downloadable | **No** public checkpoint surfaced on HuggingFace; HF paper page shows no model/checkpoint links. | **Unverified / negative** | HF papers page; HF search |
| Code publicly released | **Conflicting.** MarkTechPost comparison table says "Yes (GitHub)"; HF paper page shows no code link; `QwenLM` org `q=robot` returns 0 repos. No concrete RobotNav repo URL confirmed. | **Unverified (conflict)** | MarkTechPost vs HF/GitHub |
| License | **Conflicting / unconfirmed.** One PDF-summary fetch reported "CC-BY 4.0" (likely the *paper* license, not weights); other sources did not state a model license. Qwen open-weight LLMs are generally Apache-2.0, but that cannot be assumed for this model. | **Unverified** | arXiv-derived summary; general Qwen licensing |
| Model size on disk | Not disclosed in any reached source. | Unverified | — |
| Inference requirements (GPU/RAM/disk) | Not disclosed in any reached source. | Unverified | — |
| API/demo availability | A "Chat2Robot" experimental interface / demos were shown per coverage; no public inference API confirmed for local use. | Unverified | The AI Insider |

Net availability verdict: the model is **announced and documented**, but **weights, a concrete code
repo, a usable license, model disk size, and hardware requirements are all unverified**. The single
positive code-release signal is a secondary-source table that conflicts with the primary
HuggingFace/GitHub evidence. Under the repo honesty rules, this is treated as *not confirmed available*.

## Interface Comparison

Qwen-RobotNav interface (from the report abstract + coverage; **paper-level, unverified by code**):

- **Observation**: camera images with encoded visual *history*; an externally reconfigurable
  observation strategy with a visual *token budget*, *temporal decay*, and *per-camera importance
  weights*. Camera identity and temporal order are passed as **natural-language tags** (stated to need
  "zero architectural modification to Qwen3-VL"). No LiDAR/range-scan or depth input is confirmed.
- **Action**: **waypoint trajectories** — reported as "8 waypoints, each a 2D position and heading"
  (per MarkTechPost). Not a per-step velocity command.
- **Task / tool interface**: a parameterised interface selects a **task mode** (VLN / PointNav /
  ObjNav / Tracking). It is the *reactive executor* in a **two-tier agentic system**: an upper-tier
  planner (Qwen3.6-Plus per coverage) decomposes long-horizon goals and the two tiers communicate
  **exclusively in natural language**.
- **Control rate / kinematics**: **not disclosed**. Holonomic vs differential-drive output convention
  is not stated; waypoint+heading output is kinematics-agnostic on its face.

Robot SF local-navigation surfaces (read from this worktree's code):

- **Action space** (`robot_sf/robot/differential_drive.py`): `spaces.Box` of shape (2,) =
  `(linear_speed, angular_speed)` (`PolarVec2D` per `robot_sf/common/types.py:45`), i.e. **unicycle
  v/ω** for a **differential-drive** robot bounded by `max_linear_speed` / `max_angular_speed`. A
  holonomic variant exists (`robot_sf/robot/holonomic_drive.py`) but the default benchmark robot is
  differential drive.
- **Observation surfaces**: `ObservationMode` is `default_gym` or `socnav_struct`
  (`robot_sf/gym_env/observation_mode.py`). The default sensor stack is a **2D LiDAR range scan**
  (`robot_sf/sensor/range_sensor.py`, `num_rays=272`, configurable FOV/`max_scan_dist`) fused with
  **drive state**. The closest existing learned-local-nav intake
  (`robot_sf/planner/learned_policy_adapter.py`) declares `observation_level = "lidar_2d"` with
  required inputs `("drive_state", "rays")` — explicitly **not** camera/RGB.
- **Step contract**: gym `step(action)` consumes one bounded `(v, ω)` action per tick; the
  `RouteNavigator` (`robot_sf/nav/navigation.py`) feeds 2D waypoints/goal, and the planner adapter
  emits a single velocity tuple per step (`plan(obs) -> (float, float)`).
- **Coordinate frame**: 2D world/ego planar metric frame; LiDAR in robot-ego polar layout.

Concrete mismatches (top items first):

| # | Dimension | Qwen-RobotNav (paper-level) | Robot SF local nav | Mismatch / adapter burden |
| --- | --- | --- | --- | --- |
| 1 | **Observation modality** | RGB camera image *history* + natural-language camera/temporal tags; no LiDAR/depth confirmed. | 2D LiDAR range scan (`num_rays=272`) + drive state; no RGB camera in the default `lidar_2d` learned-policy contract. | **Fundamental.** Robot SF has no camera image observation in its default local-nav surface; bridging would require synthesizing/rendering RGB views the simulator does not natively produce for this contract. Largest single blocker. |
| 2 | **Action space / control granularity** | 8-waypoint 2D+heading trajectory per inference. | One bounded `(linear, angular)` velocity tuple per `step()` tick. | **Structural.** A waypoint-trajectory → per-step `(v, ω)` tracking/controller layer is required; differential-drive feasibility of arbitrary waypoint+heading sets is not guaranteed. |
| 3 | **Interface paradigm** | Agentic two-tier, natural-language-mediated, task-mode-parameterised, RGB-token-budget reconfigurable. | Single-call numeric `step(obs)->(v,ω)` contract with a fixed `lidar_2d` observation level. | **Paradigm-level.** Robot SF's planner contract is a stateless numeric policy call; an NL-tag, token-budget, mode-switching agentic surface does not map onto `step()` without a substantial translation/agent shim. |
| 4 | **Kinematics / control rate** | Not disclosed; waypoint+heading output is kinematics-agnostic. | Differential-drive unicycle, fixed sim timestep `d_t`. | Unverified on the Qwen side; cannot confirm control-rate or holonomic/diff-drive compatibility from sources. |
| 5 | **Coordinate frame** | Camera/ego visual frame implied; waypoint frame not fully specified. | 2D metric world + robot-ego polar LiDAR. | Adapter must define and validate the waypoint coordinate frame; unverified from sources. |

## Exact Prerequisites Before Implementation-Ready

All of the following must hold before this could become an integration issue (none verified today):

1. **Weights public under a permissive, locally-usable license** (e.g. Apache-2.0 or equivalent) with a
   confirmed download location (HuggingFace/ModelScope/GitHub) — currently unverified/conflicting.
2. **Inference code + documented runtime** (a concrete repo with a model card and an inference entry
   point) — currently no confirmed RobotNav repo URL.
3. **Disclosed inference footprint**: model disk size and GPU/RAM/disk requirements, and confirmation a
   2B/4B variant fits this machine's local GPU budget — currently undisclosed.
4. **A camera-observation bridge**: Robot SF's default local-nav contract is `lidar_2d` + drive state.
   Either (a) a simulator RGB-render path feeding Qwen's camera/NL-tag protocol, or (b) confirmation
   that a non-camera (PointNav-style) Qwen mode exists with an obs surface Robot SF can supply.
5. **A waypoint-to-`(v, ω)` controller**: a trajectory-tracking layer mapping 8-waypoint+heading output
   onto bounded differential-drive `(linear, angular)` actions at the sim timestep.
6. **An agentic-to-`step()` shim** (or a confirmed single-call reactive mode) so the NL-mediated,
   mode-parameterised interface fits Robot SF's numeric `step(obs)->action` planner contract.

## Recommended Follow-Up (asset tracking, not integration)

Open/track one bounded follow-up only:

> Track Qwen-RobotNav asset availability. Re-check, when the Qwen team publishes: (a) a concrete code
> repository, (b) downloadable 2B/4B/8B checkpoints, and (c) an explicit model license. Record the repo
> URL, license, model disk size, and minimum GPU/RAM in this note. Only after weights + permissive
> license + a non-camera obs mode (or a feasible RGB bridge) are confirmed should a separate
> `smoke_adapter_spike` issue be considered.

Do **not** add any Qwen dependency, planner-registry entry, or benchmark row until the asset/license
prerequisites above are verified.

## Validation

This is a docs-only assessment. Validation performed:

```bash
cd /home/luttkule/git/robot_sf_ll7/.claude/worktrees/issue-2952-qwen-robotnav-feasibility
uv run python -c "import pathlib; print(pathlib.Path('docs/context/issue_2952_qwen_robotnav_assessment.md').read_text()[:200])"
uv run pre-commit run --files docs/context/issue_2952_qwen_robotnav_assessment.md 2>&1 | tail -20 || true
```

Robot SF surfaces were read directly from this worktree (not inferred):
`robot_sf/robot/differential_drive.py`, `robot_sf/robot/holonomic_drive.py`,
`robot_sf/common/types.py`, `robot_sf/gym_env/observation_mode.py`,
`robot_sf/sensor/range_sensor.py`, `robot_sf/planner/learned_policy_adapter.py`,
`robot_sf/nav/navigation.py`, `robot_sf/nav/nav_types.py`,
`robot_sf/nav/motion_planning_adapter.py`. Upstream Qwen facts are web-sourced only and graded above.
