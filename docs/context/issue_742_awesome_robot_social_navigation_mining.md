# Issue 742 Awesome Robot Social Navigation Mining Note

Date: 2026-04-02
Related issues:
- `robot_sf_ll7#742` Mine awesome-robot-social-navigation into repo-scoped follow-up issues
- `robot_sf_ll7#601` CrowdNav family feasibility spike
- `robot_sf_ll7#600` DSRNN stretch follow-up
- `robot_sf_ll7#627` Prototype fail-fast Robot SF wrapper for CrowdNav/SoNIC family
- `robot_sf_ll7#629` Planner Zoo: deep research for external local planner repositories and integration candidates
- `robot_sf_ll7#692` Scenario difficulty analysis and verified simple scenarios
- `robot_sf_ll7#750` Add interval-inclusive paper export handoff for Results statistical hardening
- `robot_sf_ll7#751` Run fixed-scenario multi-seed pilot and export paper-ready seed-variability evidence

## Goal

Mine the curated repository
[`Shuijing725/awesome-robot-social-navigation`](https://github.com/Shuijing725/awesome-robot-social-navigation)
into a small set of repo-scoped follow-ups for `robot_sf_ll7` without turning the upstream list into
a backlog dump.

## Upstream source used

This note uses the upstream README as an intake surface, not as a literal issue-generation source.
The categories and entries actually checked for this pass were:

- Surveys
  - Francis et al.:
    [Principles and guidelines for evaluating social robot navigation algorithms](https://arxiv.org/pdf/2306.16740)
- Datasets and Benchmarks
  - [SocNavBench](https://github.com/CMU-TBD/SocNavBench)
  - [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html)
  - [MuSoHu](https://cs.gmu.edu/~xiao/Research/MuSoHu/)
  - [THOR and THOR-Magni](http://thor.oru.se/)
  - [AI Habitat 3.0](https://aihabitat.org/habitat3/)
  - [Arena-Rosnav 3.0](https://github.com/Arena-Rosnav)
- Methods / learned and hybrid anchors
  - [CrowdNav](https://github.com/vita-epfl/CrowdNav)
  - [HEIGHT: Heterogeneous Interaction Graph Transformer for Robot Navigation in Crowded and Constrained Environments](https://sites.google.com/view/crowdnav-height/home)
  - [CrowdNav_HEIGHT](https://github.com/Shuijing725/CrowdNav_HEIGHT)
  - [Decentralized structural-RNN for robot crowd navigation with deep reinforcement learning](https://sites.google.com/illinois.edu/crowdnav-dsrnn/home)
  - [Occlusion-Aware Crowd Navigation Using People as Sensors](https://arxiv.org/abs/2210.00552)
  - [SCOPE: Stochastic Cartographic Occupancy Prediction Engine for Uncertainty-Aware Dynamic Navigation](https://arxiv.org/abs/2407.00144.pdf)
- Environment Models
  - occupancy prediction
  - occlusion inference
- Intake-only categories checked for exclusion pressure
  - Foundation Models for Social Navigation
  - Explainability and Trust
  - User Studies
  - Workshops

## Repo-native buckets

### Planner families

- Current repo backlog already tracks the main CrowdNav-lineage and predictive-family anchors:
  `CrowdNav`, `SoNIC`, `DSRNN`, `Go-MPC`, `Pred2Nav`, and `safe_control`.
- The only planner-family delta from this pass that is both new enough and bounded enough for a
  dedicated issue is `CrowdNav_HEIGHT`.

### Benchmark / dataset references

- `SocNavBench` is already part of the repo's benchmark vocabulary and adapter/documentation
  surfaces.
- `BARN`, `DynaBARN`, and `Arena-Rosnav` are useful benchmark-shape references, but they are
  already partially captured in `docs/dev/benchmark_plan_2026-01-14.md` and do not justify a new
  issue in this pass.
- `SCAND`, `MuSoHu`, `JRDB`, `THOR`, and `Habitat 3.0` remain useful reference datasets or
  simulator anchors, but they do not currently expose one narrow, repo-native execution path that
  beats a note-only verdict.

### Environment-model references

- The upstream list reinforces that occupancy prediction and occlusion inference are legitimate
  adjacent problem families, but the concrete repo-native paths here still run through existing
  predictive-planner, scenario-difficulty, and export/report issues rather than a new standalone
  environment-model issue.

### Evaluation-guidance references

- Francis et al. is the strongest non-planner intake item because it speaks directly to benchmark
  contract quality, reporting, uncertainty, and comparison hygiene.
- The repo already cites Francis in multiple places, but it does not yet have one issue dedicated to
  translating those principles into an explicit Robot SF benchmark-contract checklist.

## Dedupe table

| Mined item | Upstream category | Verdict | Repo anchor / rationale |
| --- | --- | --- | --- |
| CrowdNav | Methods / RL | `already tracked` | Covered by `#601`, `docs/context/issue_601_crowdnav_feasibility_note.md`, and `docs/benchmark_planner_family_coverage.md`. |
| SoNIC | Methods / safety-aware lineage | `already tracked` | Already positioned through `#601`, `#602`, and `#627`; still prototype-only, not missing. |
| DSRNN | Methods / RL | `already tracked` | Covered by `#600` and `docs/context/issue_600_dsrnn_stretch_follow_up.md`. |
| Go-MPC | Methods / MPC | `already tracked` | Covered by `#599` and `docs/context/issue_599_go_mpc_assessment.md`. |
| Pred2Nav | Methods / hybrid prediction | `already tracked` | Covered by `#604` and `docs/context/issue_604_pred2nav_assessment.md`. |
| `safe_control` | safety-controller family | `already tracked` | Covered by `#695` and `docs/context/issue_695_safe_control_feasibility_note.md`. |
| SocNavBench | Datasets and Benchmarks | `already tracked` | Already appears in `docs/benchmark_spec.md`, `docs/socnav_assets_setup.md`, and planner coverage docs. |
| `CrowdNav_HEIGHT` | Methods / learned-family references | `create follow-up` | Strongest learned-family delta: canonical repo now identifiable, MIT license visible, `test.py` exists, checkpoints are published, and `docs/context/issue_629_planner_zoo_research_prompt.md` already flags it as the best learned-policy breadth candidate. |
| Francis evaluation guidelines | Surveys | `create follow-up` | Already cited in tutorial/docs, but not yet converted into one explicit benchmark-contract/checklist issue tied to `docs/benchmark_spec.md`, `#692`, `#750`, and `#751`. |
| BARN | Datasets and Benchmarks | `note only` | Already used as benchmark-shape context in `docs/dev/benchmark_plan_2026-01-14.md`; no narrow immediate Robot SF implementation step discovered here. |
| DynaBARN | Datasets and Benchmarks | `note only` | Same as BARN: useful reference for protocol design, but not a bounded issue on its own in this pass. |
| Arena-Rosnav | Datasets and Benchmarks / simulators | `note only` | Already captured in `docs/dev/benchmark_plan_2026-01-14.md`; ROS-heavy comparison surface is still reference context, not a next issue. |
| SCAND | Real-world datasets | `note only` | Useful social-navigation demonstration dataset anchor, but no narrow import/eval contract is justified yet. |
| MuSoHu | Real-world datasets | `note only` | Same as SCAND; relevant for later dataset realism discussions, not current bounded backlog work. |
| JRDB | Real-world datasets | `note only` | Reference-only for now; no immediate Robot SF benchmark ingestion path. |
| THOR / THOR-Magni | Real-world datasets | `note only` | Useful environment anchor, but no bounded repo-native next step surfaced here. |
| Habitat 3.0 | Simulators | `note only` | Too broad and too 3D-stack heavy for this pass; remains a comparison anchor only. |
| SCOPE | Environment models / occupancy prediction | `note only` | Adjacent to predictive-planner and uncertainty work, but still better handled through existing predictive/evidence issues than a standalone intake issue. |
| Occlusion-aware crowd navigation | Environment models / occlusion inference | `note only` | Relevant to predictive-planner and scenario-difficulty framing, but not a separate bounded issue here. |
| Foundation-model section | Foundation Models for Social Navigation | `intentionally excluded` | No direct benchmark-contract dependency surfaced in this pass; opening issues here would create roadmap churn without a current execution path. |
| Explainability and Trust | Explainability and Trust | `intentionally excluded` | Valuable research context, but not a current repo-native benchmark blocker. |
| User studies | User Studies | `intentionally excluded` | Out of scope for the current benchmark/planner backlog pass. |
| Workshops | Workshops | `intentionally excluded` | Useful for awareness, not for repo-scoped follow-up issues. |

## Why the two new follow-ups are enough

### `CrowdNav_HEIGHT`

This is the only planner-family entry in the upstream intake that is both:

- not already given its own repo-native issue or context note,
- concrete enough to scope tightly,
- and strong enough to matter for future learned-family breadth decisions.

Current intake evidence:

- canonical repo: `Shuijing725/CrowdNav_HEIGHT`
- license: MIT
- runnable source paths exist:
  - `test.py`
  - `train.py`
  - `check_env.py`
- pretrained checkpoints are published from the upstream README
- current runtime expectation is legacy enough that the right first step is still a bounded
  assessment, not implementation

### Francis evaluation guidance

Francis is already part of the repo's benchmark vocabulary, but only diffusely:

- `docs/training/predictive_planner_complete_tutorial.md`
- `docs/dev/benchmark_plan_2026-01-14.md`
- scenario-pack and benchmark-spec surfaces that already reflect some of the same concerns

What is missing is one explicit issue that answers:

- which Francis-style principles are already satisfied by current Robot SF benchmark policy,
- which principles are only partially implemented,
- and which gaps deserve narrow follow-up actions rather than broad benchmark redesign.

## Explicit exclusions for this pass

Do not open standalone issues from this mining note for:

- `BARN`
- `DynaBARN`
- `Arena-Rosnav`
- `SCAND`
- `MuSoHu`
- `JRDB`
- `THOR`
- `Habitat 3.0`
- `SCOPE`
- foundation-model papers
- explainability / trust papers
- user-study entries
- workshop entries

These remain legitimate reference context, but the repo does not yet have a bounded next step for
them that is stronger than a conservative note.

## Recommended follow-up count

Keep the output of this issue to exactly two new follow-up issues:

1. `CrowdNav_HEIGHT` bounded external learned-family assessment
2. Francis evaluation-guidance to Robot SF benchmark-contract issue

Anything beyond those two should require a new discovery pass with a clearer execution path.
