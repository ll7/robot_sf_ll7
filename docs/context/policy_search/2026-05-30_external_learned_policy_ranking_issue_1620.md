# External Learned Local-Navigation Policy Ranking - Issue #1620 - 2026-05-30

Date: 2026-05-30

Related issue:

- Issue #1620: <https://github.com/ll7/robot_sf_ll7/issues/1620>

Related Robot SF anchors:

- Issue #1617 local-planner repository survey:
  `docs/context/issue_1617_local_planner_repo_survey.md`
- Issue #1618 learned-policy adapter interface:
  `docs/context/issue_1618_learned_policy_adapter_interface.md`
- Issue #1619 learned local-policy source/source-claim audit:
  <https://github.com/ll7/robot_sf_ll7/issues/1619>
- Issue #1355 learned local-navigation candidate screen:
  `docs/context/policy_search/2026-05-20_learned_local_navigation_screen.md`
- Learned-policy registry:
  `docs/context/policy_search/learned_policy_registry.md`
- Reject/monitor registry:
  `docs/context/policy_search/reject_monitor_registry.md`

## Goal

Rank external learned local-navigation policy candidates for Robot SF follow-up work. This note is
assessment-only: it does not import a planner, train a model, stage external assets, or claim
benchmark readiness.

The ranking uses current source checks plus the existing Robot SF adapter contract. A high rank
means "best next source-side or Robot SF-native learning work", not "ready benchmark row".

## Ranking Rubric

- `novelty`: likely paper or research value beyond already implemented Robot SF planners.
- `implementation_effort`: effort to produce a faithful Robot SF-compatible result.
- `expected_benchmark_value`: likely usefulness for Robot SF comparisons after proof.
- `publication_value`: likely value for a paper-facing discussion once provenance is clean.
- `adapter_fit`: how cleanly the upstream observation/action contract maps to Robot SF's learned
  local-policy interface from Issue #1618.

Verdicts:

- `implement now`: source and adapter are clean enough for immediate Robot SF implementation.
- `source-side reproduction first`: run or reproduce upstream first, then decide on adapter work.
- `monitor only`: keep as context; do not start adapter work yet.
- `reject for now`: current source, license, checkpoint, or contract gap blocks useful work.

## Ranked Shortlist

No external candidate currently deserves an `implement now` verdict. The strongest next work is
source-side reproduction or a Robot SF-native analogue that preserves the source idea without
pretending to be a faithful upstream benchmark row.

| Rank | Candidate/family | Source | License status | Checkpoint/source status | Observation/action fit | Reproducibility status | Novelty | Effort | Benchmark value | Publication value | Verdict |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Tentabot-style motion-primitive value policy | <https://github.com/RIVeR-Lab/tentabot> | no GitHub-detected license | ROS/Gazebo source visible; trained-model path is source-specific; Robot SF has a clean-room `tentabot_value_scorer_v0` spike staged in `candidate_registry.yaml` | good conceptual fit as a learned scorer over motion primitives, weak direct adapter fit | upstream source harness remains unproven; staged Robot SF scorer still needs its required smoke and nominal-sanity validation before evidence use | high | medium | high | high | `source-side reproduction first` |
| 2 | CrowdNav HEIGHT / IGAT graph policies | <https://github.com/Shuijing725/CrowdNav_HEIGHT> | MIT in GitHub metadata | active source; prior Robot SF note records checkpoint/source-harness blockers | medium if graph/history adapter is proven; weak without source-state parity | source harness blocked on legacy dependencies/assets | high | medium-large | high | high | `source-side reproduction first` |
| 3 | Arena-Rosnav learned policy stack | <https://github.com/Arena-Rosnav/arena-rosnav> | MIT in current org repo metadata | ROSNav source visible; full stack rather than a single local-policy checkpoint | medium for ROS benchmark comparison; poor as direct Robot SF planner | new source-side assessment needed before adapter claims | medium | large | medium | medium-high | `source-side reproduction first` |
| 4 | DRL-VO | <https://github.com/TempleRAIL/drl_vo_nav> | GPL-3.0 in GitHub metadata | ROS source visible; prior Robot SF audit keeps it prototype-only | medium-low due tracked-agent and VO contract assumptions | source/prototype only; privileged-state audit already exists | high | large | medium | high | `source-side reproduction first` |
| 5 | GenSafeNav / SoNIC safety-aware crowd navigation | <https://github.com/tasl-lab/GenSafeNav>, <https://github.com/tasl-lab/SoNIC-Social-Nav> | MIT in GitHub metadata | active source; source harness previously blocked by legacy gym/dependency gaps | medium if safety signal and state construction are mapped explicitly | source harness required | medium-high | medium-large | medium-high | high | `source-side reproduction first` |
| 6 | DS-RNN / RGL / classic CrowdNav graph baselines | <https://github.com/Shuijing725/CrowdNav_DSRNN>, <https://github.com/vita-epfl/CrowdNav>, <https://github.com/ChanganVR/RelationalGraphLearning> | DS-RNN/CrowdNav MIT in metadata; RGL no GitHub-detected license | source visible; old environment assumptions | medium-low due graph/history packing and source simulator semantics | already tracked; reproduce source before wrapper claims | medium | medium-large | medium | medium-high | `source-side reproduction first` |
| 7 | NeuPAN | <https://github.com/hanruihua/NeuPAN> | GPL-3.0 in GitHub metadata | source visible; no current Robot SF checkpoint import path | medium for point-obstacle navigation, weak for social crowd benchmark claims | source-side assessment exists; adapter blocked | medium | large | medium | medium | `monitor only` |
| 8 | SAGE / MPC-transfer GNN | <https://github.com/TIB-K330/drl_planner> | MIT in GitHub metadata | source visible; Issue #1369 source smoke reached missing legacy `gym`, with no checkpoint/inference proof | medium conceptually, weak until source buffer/checkpoint path is known | source-side blocked after partial smoke | medium | medium-large | medium | medium | `source-side reproduction first` |
| 9 | NavDP diffusion navigation | <https://github.com/InternRobotics/NavDP> | no GitHub-detected license | active source; no clean Robot SF 2D local-policy weight/contract | weak: RGB-D, privileged sim guidance, and trajectory follower assumptions | monitor-only in prior NavDP/NoMaD note | high | large | low-medium | high | `monitor only` |
| 10 | NoMaD / ViNT / GNM visual navigation | <https://github.com/robodhruv/visualnav-transformer> | MIT in GitHub metadata | repo description claims official code/checkpoint release; artifact use is not verified here | weak: visual goal/topomap policy, not Robot SF state local planner | monitor-only in prior NavDP/NoMaD note | high | large | low-medium | high | `monitor only` |
| 11 | Diffusion Policy / Consistency Policy / Diffuser families | <https://github.com/real-stanford/diffusion_policy>, <https://github.com/Aaditya-Prasad/consistency-policy>, <https://github.com/jannerm/diffuser> | MIT in GitHub metadata for checked repos | source visible, but not local social-navigation checkpoints | weak unless reframed as Robot SF-native training method | analysis issue exists; source is not adapter-ready | high | large | low-medium | medium-high | `monitor only` |
| 12 | Decision Transformer / trajectory transformer local-nav lane | Issue #1622, Issue #1752 | local analysis/data-preflight lane | no external local-navigation checkpoint selected | medium only after dataset preflight and offline action contract | source-side selection still open | medium | medium-large | medium | medium | `monitor only` |
| 13 | Foundation/VLA navigation policies | <https://github.com/openvla/openvla>, <https://github.com/octo-models/octo> | MIT in GitHub metadata | large robot-policy sources visible; manipulation or broad robot policy focus | poor for current 2D local-planner interface | readiness analysis exists; adapter is not current work | high | very large | low | medium-high | `reject for now` |
| 14 | Generic SAC/TD3/PPO mapless external baselines | prior Robot SF SAC/PPO notes | varies | no new external candidate with source/checkpoint advantage over Robot SF PPO lane | medium when trained locally, weak as external import | internal baseline lane only | low-medium | medium | medium | low-medium | `reject for now` |
| 15 | DWA-RL / learned dynamic-window variants | source family from issue #1355 screen | varies/unclear | no current source/checkpoint contract selected | medium conceptually, weak as external import | source-side route remains open only if public source/checkpoint path is identified | low-medium | medium | low-medium | low-medium | `source-side reproduction first` |

## Interpretation

The top external lane is not a direct import. Tentabot is ranked first because its motion-primitive
value idea matches Robot SF's planner shape better than visual or full-stack ROS navigation systems.
Robot SF already stages the clean-room `tentabot_value_scorer_v0` spike in
`docs/context/policy_search/candidate_registry.yaml`; the remaining gap is validation/provenance,
not inventing another duplicate spike. Upstream Tentabot source reproduction remains separate from
that clean-room scorer and must not be used as benchmark evidence until its own source harness,
license boundary, and observation/action metadata are proven.

CrowdNav HEIGHT/IGAT remains the highest-value graph/social-navigation family, but its value depends
on source-state parity and checkpoint/source-harness proof. The existing CrowdNav-family verdicts
correctly block a direct benchmark row until upstream execution and observation/action metadata are
proven.

Arena-Rosnav is worth a new source-side assessment because it is a current ROSNav benchmark stack
rather than a single policy. Issue #1617 only surveyed repository metadata, so Issue #1758 should
prove or fail closed on source-side execution, checkpoint/provenance, and observation/action
mapping before any stronger benchmark or publication claim.

Visual, diffusion, world-model, Decision Transformer, and VLA families remain important research
context. They do not yet expose a clean Robot SF 2D local-policy interface with source, checkpoint,
and fail-closed behavior, so their next work is analysis, dataset preflight, or monitor-only tracking.

## Next-Step Issue Routing

Do not open duplicate implementation issues for the ranked families without changing the source
evidence. Separate closed predecessor anchors from active next-step coverage:

Active next-step coverage and still-applicable gates:

- Tentabot-style learned scorer: staged `tentabot_value_scorer_v0` covers the Robot SF-native
  scorer spike; required smoke and nominal-sanity validation remain the relevant gate before using
  that staged candidate as evidence.
- CrowdNav/HEIGHT source harness: source-harness and checkpoint/source-state parity remain the
  active gate before any direct adapter or benchmark row.
- GenSafeNav/SoNIC source harness and conformal-contract checks remain the active gate before any
  guarded learned-policy benchmark claim.
- NeuPAN source-side boundary remains monitor-only until source-side proof and adapter metadata
  exist.
- General diffusion-policy/consistency-policy method feasibility: Issue #1621.
- Decision Transformer trajectory-data preflight: Issue #1622 and Issue #1752.
- Foundation/VLA readiness: Issue #1626.
- SAGE / MPC-transfer GNN remains blocked on source-side execution, checkpoint/inference path, and
  observation/action metadata.

Closed or historical anchors, not active next-step coverage:

- Tentabot source assessment and local spike predecessors: Issue #1357 and Issue #1387.
- CrowdNav/HEIGHT and classic CrowdNav-family predecessors: Issue #1394, Issue #1367, Issue #600,
  and the CrowdNav-family verdict.
- GenSafeNav/SoNIC source-harness and conformal-contract predecessors: Issue #1393 and Issue #1366.
- NeuPAN predecessor: Issue #1368.
- SAGE / MPC-transfer predecessor: Issue #1369.
- DRL-VO privileged-state/prototype predecessor: Issue #1364 and
  `docs/context/issue_769_drl_vo_assessment.md`.
- NavDP/NoMaD diffusion/visual navigation predecessor: Issue #1356; current broader diffusion
  routing is Issue #1621.
- DWA-RL learned dynamic-window route: closed registry-maintenance Issue #1359 records the current
  monitor boundary; open a new source-side issue only if public source/checkpoint evidence appears.

New recommended follow-up:

- Arena-Rosnav source-side assessment: Issue #1758. Acceptance should require current repo/license
  confirmation, a source-side run or explicit fail-closed block, trained-policy/checkpoint
  provenance, ROS dependency scope, observation/action mapping, and a verdict of
  `source-side reproduction first`, `monitor only`, or `reject for now`.

## Source Checks

Source and issue checks used for this note. These checks were run on 2026-05-30 and support
metadata/routing claims only; they are not source-side execution proof.

| Family | Checked result used in ranking |
| --- | --- |
| Tentabot | `RIVeR-Lab/tentabot` source was visible; GitHub license metadata was absent; Robot SF already has staged clean-room `tentabot_value_scorer_v0` config metadata, but no new validation result is introduced here. |
| CrowdNav HEIGHT / CrowdNav graph family | GitHub metadata showed visible source for the checked CrowdNav-family repositories; existing Robot SF notes still require source-harness/checkpoint parity before adapter claims. |
| Arena-Rosnav | `Arena-Rosnav/arena-rosnav` was checked directly and via search; current source assessment is delegated to Issue #1758. |
| DRL-VO | `TempleRAIL/drl_vo_nav` GitHub metadata was visible with GPL-3.0 license metadata; existing Robot SF assessment remains prototype-only. |
| GenSafeNav / SoNIC | `tasl-lab/GenSafeNav` and `tasl-lab/SoNIC-Social-Nav` were visible with MIT license metadata in the checked GitHub response; existing source-harness and conformal-contract gates remain. |
| NeuPAN | `hanruihua/NeuPAN` metadata was visible with GPL-3.0 license metadata; no Robot SF checkpoint/import path is claimed. |
| SAGE / MPC-transfer GNN | `TIB-K330/drl_planner` metadata was visible with MIT license metadata; prior source smoke remains blocked before checkpoint/inference claims. |
| NavDP / NoMaD | NavDP source was visible with no GitHub-detected license metadata; NoMaD source metadata remained monitor-only due visual/topomap assumptions. |
| Diffusion / Consistency / Diffuser | Checked repositories were visible; no local social-navigation checkpoint or Robot SF adapter contract is claimed. |
| Foundation/VLA | `openvla/openvla` and `octo-models/octo` metadata were visible; readiness remains reject-for-current-adapter. |
| Issue routing | Open-issue listing was used only for active coverage; closed predecessors are now marked as historical anchors. |

Command recipes:

```bash
gh repo view RIVeR-Lab/tentabot --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view Shuijing725/CrowdNav_HEIGHT --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view Shuijing725/CrowdNav_DSRNN --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view vita-epfl/CrowdNav --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view ChanganVR/RelationalGraphLearning --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view tasl-lab/SoNIC-Social-Nav --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view tasl-lab/GenSafeNav --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view TempleRAIL/drl_vo_nav --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view James-R-Han/DR-MPC --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view InternRobotics/NavDP --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view robodhruv/visualnav-transformer --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view hanruihua/NeuPAN --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view TIB-K330/drl_planner --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view Arena-Rosnav/arena-rosnav --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh search repos "Arena Rosnav" --limit 10 --json fullName,url,description,license,updatedAt,stargazersCount
gh repo view openvla/openvla --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view octo-models/octo --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view real-stanford/diffusion_policy --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view Aaditya-Prasad/consistency-policy --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh repo view jannerm/diffuser --json nameWithOwner,url,description,licenseInfo,isArchived,stargazerCount,updatedAt,defaultBranchRef
gh issue list --state open --limit 200 --json number,title,labels,url
```

Document validation:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
