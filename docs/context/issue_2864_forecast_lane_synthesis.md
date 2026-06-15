# Issue #2864 Forecast-lane synthesis after PRs #2849-#2862

Issue: [#2864](https://github.com/ll7/robot_sf_ll7/issues/2864)

Status: current synthesis.

## Claim boundary on 2026-06-15

This note is a compact routing summary for the fast forecast-lane implementation wave.
It is strictly diagnostic-to-infrastructure evidence.
It does not establish benchmark, dissertation, paper-facing, or safety claims.

## PR-to-issue mapping and evidence type

| PR | Title | Mapped issue(s) | Evidence type | Lane role |
| --- | --- | --- | --- | --- |
| [#2849](https://github.com/ll7/robot_sf_ll7/pull/2849) | feat: add ForecastBatch schema contract | #2836 (closed in PR body) | schema | schema |
| [#2850](https://github.com/ll7/robot_sf_ll7/pull/2850) | feat: add forecast batch probabilistic metrics | #2840 (referenced in PR body) | metric | metric |
| [#2851](https://github.com/ll7/robot_sf_ll7/pull/2851) | feat: add fast bicycle actor fixtures | #2727 (referenced in PR body) | baseline | baseline |
| [#2852](https://github.com/ll7/robot_sf_ll7/pull/2852) | feat: separate forecast metrics by actor class | #2846 (closed in PR body) | schema/metric | metric |
| [#2853](https://github.com/ll7/robot_sf_ll7/pull/2853) | Add motion-rich forecast trace evidence | #2774 (referenced in PR body) | diagnostic evidence | baseline |
| [#2854](https://github.com/ll7/robot_sf_ll7/pull/2854) | Add semantic forecast baselines | #2758 (closed in PR body) | baseline | baseline |
| [#2855](https://github.com/ll7/robot_sf_ll7/pull/2855) | Add interaction-aware forecast baseline | #2781 (closed in PR body) | baseline | baseline |
| [#2856](https://github.com/ll7/robot_sf_ll7/pull/2856) | Add closed-loop forecast coupling gate | #2843 (mapped from changed artifact paths) | closed-loop gate | closed-loop gate |
| [#2857](https://github.com/ll7/robot_sf_ll7/pull/2857) | Add opt-in forecast risk scoring diagnostic | inferred from branch `issue-2759-forecast-risk-scoring` | risk-scoring support | risk-scoring support |
| [#2858](https://github.com/ll7/robot_sf_ll7/pull/2858) | Fix shared-venv coverage isolation for worktrees | #2833 (closed in PR body) | workflow infra | other |
| [#2859](https://github.com/ll7/robot_sf_ll7/pull/2859) | Document prediction research lane routing | #2848 (closed in PR body) | documentation | documentation |
| [#2860](https://github.com/ll7/robot_sf_ll7/pull/2860) | feat: add forecast observation adapters | #2838 (referenced in PR body) | schema | schema |
| [#2861](https://github.com/ll7/robot_sf_ll7/pull/2861) | feat: add forecast dataset recorder | #2839 (closed in PR body) | dataset | dataset |
| [#2862](https://github.com/ll7/robot_sf_ll7/pull/2862) | feat: add forecast transferability stress matrix | inferred from branch `issue-2847-transferability-stress-matrix` | transferability support | transferability support |

## Allowed claims (explicitly supported)

- Forecast-lane schema, provenance, and adapter contracts are in place (`ForecastBatch.v1`,
  actor-class metadata, forecast metrics reporting, and observation adapters): Issue #2836, Issue #2838,
  Issue #2839, Issue #2840, Issue #2846.
- Open-loop baseline family coverage is available on bounded durable trace sets and can be run
  reproducibly with diagnostic intent: Issue #2727, Issue #2758, Issue #2774, Issue #2781.
- Closed-loop coupling decisions and transferability scaffolding are defined and now run diagnostically:
  Issue #2843, Issue #2847.
- Forecast-to-control risk channels are implemented as opt-in diagnostics with default-off behavior:
  Issue #2759.

## Blocked claims (must stay blocked)

- Forecast-based improvements in local-navigation safety remain unsupported as benchmark evidence.
- Forecast-based improvements in navigation progress remain unsupported as benchmark evidence.
- Forecast under transfer (noise/latency/dropout/occlusion/map-density/actor-type shifts) remains
  unsupported as benchmark evidence.
- Any predictor-accuracy or planner-ranking claim requires benchmark-eligible proof tied to same-seed
  closed-loop campaigns; current evidence is mixed or diagnostic-only.

## Recommendation for Issue #2835 by sublane

### Continue

Continue in **infrastructure/scaffold** mode:

- Keep Issue #2836, Issue #2838, Issue #2839, Issue #2840, Issue #2846 as done and required.
- Keep PR-driven evidence ingestion in `docs/context/evidence/README.md` and linked ledger/graph checks alive.
- Permit next-priority diagnostics in `Issue #2837`; treat the now-closed `Issue #2841` calibration
  report and `Issue #2842` conformal pilot as diagnostic inputs, not promotion evidence.

### Revise

Revise before any learned-predictor expansion:

- Issue #2843: gate recommendation is mixed/revise, not continue.
- Issue #2781 in PR #2855: interaction-aware baseline is mixed (improves likelihood proxy but worsens
  1s ADE),
  so next work should target the gate assumptions rather than promotion.
- Issue #2844 and Issue #2845: remain blocked until gate and transfer evidence are corrected.

### Stop

Stop any paper-facing or dissertation-ready safety/progress/transfer claim narratives until:

- Same-seed closed-loop evidence shows non-regressive route progress and safety under at least one
  transfer slice,
- false-positive and fallback/degraded behavior is explicitly bounded,
- and learned predictor work passes conservative transferability gates.

## Open lanes visible from current artifact graph

From `docs/context/prediction_lane_dependency_graph.json` and issue-chain signals:

- Issue #2837 (horizon/timestep ablation): open.
- Issue #2841 (calibration and reliability): closed diagnostic input as of 2026-06-15.
- Issue #2842 (conformal/reachable-set uncertainty pilot): closed diagnostic smoke input as of
  2026-06-15.
- Issue #2843 (closed-loop gate): closed with revise recommendation.
- Issue #2844 and Issue #2845: blocked by gate and transfer readiness.

## Validation notes (metadata and path checks run)

- Inspected: Issue #2864 issue body, parent Issue #2835 issue body/comments, PR bodies/metadata for
  PR #2849-#2862.
- Inspected local routing surfaces: `docs/ai/prediction_lane.md`,
  `docs/context/prediction_lane_dependency_graph.json`,
  `docs/context/evidence/README.md`,
  `docs/context/catalog.yaml`.
- Inspected recent dissertation ledger update issue: Issue #2870.
- Skipped expensive re-exec of experiments because this is docs-only synthesis work; only
  metadata and pointer consistency review was performed.
