# Issue #2953 Qwen-RobotNav Metric And Benchmark Audit (2026-06-19)

Issue: [#2953](https://github.com/ll7/robot_sf_ll7/issues/2953)

Status date: 2026-06-19

## Claim Boundary

This is an external-source audit only. Qwen-RobotNav reports strong results on vision-language
navigation, object navigation, tracking, embodied QA, and autonomous-driving suites, but those results
are not Robot SF benchmark evidence. They should be treated as diagnostic context for whether issue
[#2952](https://github.com/ll7/robot_sf_ll7/issues/2952) should proceed, narrow, or wait for public
assets.

Primary-source access was partial:

- The technical report PDF was accessible from the Qwen-hosted paper URL:
  <https://qianwen-res.oss-accelerate.aliyuncs.com/qwenrobot/papers/Qwen_RobotNav.pdf>
- The arXiv abstract page was accessible:
  <https://arxiv.org/abs/2606.18112>
- The `qwen.ai` blog URLs rendered as client-side shells in this environment, so the audit also
  used the Alibaba Cloud mirror of the official Qwen-RobotNav post for compact headline tables:
  <https://www.alibabacloud.com/blog/qwen-robotnav-a-scalable-navigation-model-designed-for-an-agentic-navigation-system_603266>
- The paper-cited repository URL `https://github.com/QwenLM/Qwen-RobotNav` returned `404` during
  this audit, so code, weights, dataset revisions, and evaluation scripts were not verified.

## Reported Benchmark Crosswalk

| Qwen-RobotNav surface | Reported metrics | Headline result from primary sources | Robot SF relevance | Caveat |
| --- | --- | --- | --- | --- |
| VLN-CE R2R/RxR Val-Unseen | `NE` lower, `OS/OSR`, `SR`, `SPL`, `nDTW` higher | Panoramic 8B reports `72.1%` R2R `SR`, `76.5%` RxR `SR`, and `72.5` RxR `nDTW`; the report states `+10.4` and `+12.1` `SR` over NavFoM. | Indirectly relevant metric-shape analogue for instruction-conditioned navigation. | Indoor VLN route following with different simulators, observations, and action interface; no social-force crowd dynamics. |
| VLNVerse fine/coarse splits | `TL`, `NE`, `OSR`, `SR`, `SPL` | 8B reports `63.75% SR / 57.93% SPL` fine-grained and `46.59% SR / 41.54% SPL` coarse-grained. | Indirectly relevant because it includes full-kinematics embodied locomotion. | Still a VLN benchmark, not Robot SF local/social navigation. |
| VLN-PE R2R Val-Unseen | `TL`, `NE`, `FR`, `OS`, `SR`, `SPL` | 8B reports `65.50% SR` and `61.19% SPL`. | Indirectly relevant for high-level spatial-reasoning comparison. | Flash-controller setting decouples low-level locomotion from planner behavior. |
| MP3D/HM3D closed-vocabulary ObjectNav | `SR`, `SPL`, distance-to-goal mentioned | 4B reports `52.2% SR` on MP3D and `75.6% SR` on HM3D-v2 with `1.72 m` distance-to-goal. | Indirectly relevant for goal-search behavior and path efficiency. | ObjectNav target search differs from Robot SF crowd-aware local planning. |
| HM3D-OVON open-vocabulary ObjectNav | `SR`, `SPL` on Seen/Synonyms/Unseen | 4B reports `57.7 / 60.1 / 53.1% SR` across the three splits. | Indirectly relevant for open-vocabulary target search only. | No direct collision/social-compliance mapping. |
| EVT-Bench single-target tracking | `TR` higher, `CR` lower, `SR` higher | 4B reports `90.0 TR`, `6.40 CR`, `77.4 SR`; 8B reports `89.7 TR`, `5.70 CR`, `78.6 SR`. | Diagnostic analogue for moving-agent tracking. | Qwen has strong tracking rate but lower success than some tracking-specialist baselines; target tracking is not social navigation. |
| HM-EQA, MT-HM3D, EXPRESS-Bench | QA accuracy/score higher, steps lower, `Epath` higher | Qwen3.6-Plus + Qwen-RobotNav reports `76.7` HM-EQA accuracy, `54.4` MT-HM3D accuracy, and `79.27` EXPRESS LLM score. | Diagnostic-only for agentic navigation systems. | QA score and navigation-step reductions are not local-navigation safety or social-compliance metrics. |
| NAVSIM navtest | `NC`, `DAC`, `TTC`, comfort, ego progress, `PDMS` higher | 4B reports `91.4 PDMS`, `99.8 NC`, and `98.5 TTC`; 8B reports `90.9 PDMS`, `99.8 NC`, and `98.2 TTC`. | Diagnostic-only safety/trajectory-planning analogue. | Autonomous-driving dynamics and ego-vehicle metrics do not map directly to Robot SF pedestrians/social force. |
| AlpaSim PhysicalAI-AV NuRec | Close-encounter and off-road rates lower, score higher | 4B reports `22.0%` close encounter, `34.0%` off-road, `0.15` score; 8B reports `22.0%`, `27.0%`, `0.17`. | Mostly not comparable. | Driving benchmark, and reported numbers are weak relative to the listed Alpamayo baselines. |

## Metric Mapping To Robot SF

| External metric | Robot SF analogue | Mapping status |
| --- | --- | --- |
| `SR` / success rate | Episode success or task completion | Indirect; success definitions differ by dataset and simulator. |
| `SPL` / path-efficiency weighted success | Robot SF path efficiency and completion tradeoff | Indirect; useful as a concept, not a comparable number. |
| `NE` / navigation error | Goal-distance or final-distance measures | Indirect; Robot SF scenarios are authored local-navigation tasks, not VLN route targets. |
| `CR`, `TTC`, `NC` | Collision and safety-related metrics | Diagnostic; closest to Robot SF safety concerns, but driving/tracking definitions differ. |
| `TR` / tracking rate | Dynamic-agent following or target-maintenance proxy | Diagnostic; not social compliance. |
| EQA accuracy, LLM score, steps, `Epath` | No direct Robot SF local-navigation metric | Not comparable except as agentic-system context. |

## Recommendation For #2952

Do not proceed directly to a Robot SF integration or benchmark based on the reported Qwen-RobotNav
headline results. The next #2952 step should be a narrow blocked-asset / adapter-feasibility issue that
first resolves the public artifact boundary:

- locate a valid public Qwen-RobotNav repository, weights, license, and evaluation entry point;
- classify whether inference can be run locally or only through an external service;
- map Qwen's waypoint/action interface to a Robot SF adapter contract before any benchmark run;
- predeclare any future Robot SF result as `adapter` or `degraded` unless the native contract is met.

If those assets remain unavailable, #2952 should stay blocked or monitoring-only. If they become
available, the first executable Robot SF proof should be a smoke adapter/preflight, not a benchmark
claim. Estimated confidence: `0.8`; the conclusion would change if Qwen publishes reproducible
weights, scripts, and a license-compatible local inference path.

## Validation

- Downloaded and text-extracted the Qwen-hosted technical report PDF into a private common Git-dir
  artifact for table inspection.
- Verified the source pages above on 2026-06-19; `qwen.ai` rendered client-side only in this
  environment, while the Alibaba Cloud mirror exposed the blog text and compact tables.
- Probed the paper-cited `QwenLM/Qwen-RobotNav` GitHub path and observed `404`.
- No code, benchmark config, runtime result, or Robot SF metric changed.
