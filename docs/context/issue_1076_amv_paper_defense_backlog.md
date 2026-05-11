# Issue #1076 AMV Paper-Defense Backlog Tracker

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1076>

Benchmark review guide: [docs/code_review.md](../code_review.md)

Camera-ready workflow: [docs/benchmark_camera_ready.md](../benchmark_camera_ready.md)

Publication artifact policy: [docs/benchmark_artifact_publication.md](../benchmark_artifact_publication.md)

## Purpose

Issue #1076 is a coordination tracker, not a direct feature implementation issue. Its job is to
hold the upstream AMV paper-defense backlog in one place, group the child work into execution
waves, and preserve which items are near-term submission-defense candidates versus longer-horizon
follow-up.

Maintainer decision on 2026-05-09:

- keep #1076 as the full parent tracker,
- execute Wave 1 first,
- keep Waves 2 and 3 as follow-up unless maintainers promote them.

## Tracker State

All approved child issues #1077 through #1092 are filed and link back to #1076. The filing waves
below intentionally separate paper-defense blockers from adoption, reproducibility, and long-horizon
benchmark extensions.

### Wave 1: Submission-Defense Priorities

| Issue | Scope | Current PR |
| --- | --- | --- |
| #1077 | Canonicalize collision encoding and enforce episode schema conformance. | #1095 |
| #1078 | Add paired bootstrap contrasts and effect sizes to aggregate outputs. | #1096 |
| #1079 | Ship `confirmation_v1` scenario matrix with semantically disjoint archetypes. | #1097 |
| #1080 | Declare planner observation specs and support controlled observation-mode overrides. | #1098 |
| #1081 | Add configurable observation-noise injection for benchmark runs. | #1099 |

Wave 1 is the current submission-defense execution subset. These PRs are draft implementation PRs
at the time this note was written; they are not merged paper evidence until reviewed, merged, and
rerun through the release evidence path.

### Wave 2: Named Limitation Closers

| Issue | Scope | Dependency Notes |
| --- | --- | --- |
| #1082 | Add `paper_cross_kinematics_v1` parity sweep across supported motion models. | Should build on #1080 so kinematics and observation contracts remain explicit. |
| #1083 | Add `sanity_v1` nominal-scenario matrix for easy deployment-like scenes. | Complements #1079; keep separate from paper-facing matrices until validated. |
| #1084 | Add automated inclusion-check gate for promoting planners from experimental to promoted. | Depends on the planner readiness language in `docs/benchmark_camera_ready.md`. |
| #1085 | Promote pedestrian-impact outputs to schema-validated aggregate metrics. | Should preserve schema compatibility and avoid changing existing aggregate semantics silently. |

Wave 2 closes named limitations but should not be treated as automatically required for the
submission-defense window.

### Wave 3: Adoption And Reproducibility Follow-Up

| Issue | Scope | Dependency Notes |
| --- | --- | --- |
| #1086 | Publish pinned Docker reproduction path for canonical bundle verification. | Should follow the publication artifact policy and avoid private-machine assumptions. |
| #1087 | Ship adapter starter template and reference example. | Should align with planner readiness and fallback-policy language. |
| #1088 | Add PR smoke workflow for promoted planners on a 1x1 benchmark slice. | Should reuse existing `scripts/dev/` entry points where possible. |
| #1089 | Generate a self-contained static dashboard from a benchmark bundle. | Should consume bundle artifacts without creating new benchmark claims. |

Wave 3 improves adoption and repeatability. These items can support external review, but they are
not substitutes for refreshed benchmark evidence.

### Wave 4: Long-Horizon Benchmark Extensions

| Issue | Scope | Dependency Notes |
| --- | --- | --- |
| #1090 | Add configurable FOV and occlusion filtering for planner observations. | Extends #1080/#1081 style observation contract work; avoid calibrated sensor claims unless proven. |
| #1091 | Import real-world pedestrian trajectory datasets as benchmark scenarios. | Requires provenance, licensing, and scenario-fit review before any paper-facing use. |
| #1092 | Add multi-AMV scenario support and inter-robot interaction metrics. | Broad benchmark extension; keep separate from single-AMV paper-defense evidence. |

Wave 4 is explicitly long-horizon. These issues may become valuable benchmark extensions, but they
should not be conflated with the current AMV paper-defense evidence path.

## Dependencies And Duplicates

- #1079 creates a confirmation matrix that later matrix issues (#1082 and #1083) can reference, but
  it does not replace paper-facing benchmark matrices by itself.
- #1080 establishes observation-mode metadata that later sensor/observation degradation issues
  (#1081 and #1090) should reuse instead of inventing parallel labels.
- #1086 should follow `docs/benchmark_artifact_publication.md`; local `output/` artifacts are not
  durable publication references.
- Duplicate/inconsistency note: #1077 currently has two open implementation PRs, #1094 and #1095.
  This tracker treats #1095 as the current Wave 1 PR because it is the later branch opened from the
  active issue-to-PR workflow, but maintainers should close or merge only one collision-contract PR.

## Validation

Checked on 2026-05-09:

```bash
rtk gh issue view 1076 --comments --json number,title,body,comments,labels,state,url
rtk bash -lc 'for i in $(seq 1077 1092); do gh issue view "$i" --json number,title,body,url | jq -r "[.number, .title, .url, ((.body // \"\") | if test(\"#1076\\\\b\") then \"links-1076\" else \"missing-1076-link\" end)] | @tsv"; done'
```

Result: every child issue #1077-#1092 reported `links-1076`.

This note does not claim any child issue is complete. Completion remains per-child and requires its
own implementation PR, validation proof, and review.
