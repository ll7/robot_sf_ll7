# Issue #1676 Proxemic Profile Comfort Slice

Date: 2026-05-30

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1676>  
Parent survey: [issue_1617_local_planner_repo_survey.md](issue_1617_local_planner_repo_survey.md)

## Goal

Issue #1676 asks for a Robot SF-native scenario/config slice inspired by proxemic-profile work in
the local-planner survey. This change keeps the slice inside existing policy-search machinery and
does not import Webots code, external assets, or subjective social-navigation labels.

## Metric Coverage

Existing Robot SF metrics can represent a limited comfort-vs-efficiency proxy:

- `min_clearance` and `mean_clearance`: robot-pedestrian surface clearance in meters.
- `near_misses`: thresholded intrusive-clearance events.
- `force_exceed_events` and `comfort_exposure`: force-based comfort exposure when force arrays are
  available.
- `time_to_goal_norm`, `path_efficiency`, and `success`: efficiency/progress side of the tradeoff.
- `snqi_mean`: composite score when a campaign declares the existing SNQI weights and baseline.

Missing metric contract: the repo does not currently have a subjective proxemic-comfort label,
human preference score, profile-specific comfort acceptance threshold, or social-norm success
metric. Profile tuning must therefore be interpreted as clearance/comfort-proxy diagnostics, not as
benchmark-strength social acceptability evidence.

## Profile Config Slice

The slice registers three exploratory policy-search candidates over the existing
`hybrid_rule_v3_teb_like_rollout` planner. The profiles change only social clearance and speed-cap
knobs so smoke runs remain comparable:

| Profile | Candidate | Intent |
| --- | --- | --- |
| Conservative | `proxemic_profile_conservative_issue_1676` | Larger desired dynamic clearance, stronger TTC/clearance weighting, wider slow zone, lower near-human speed. |
| Neutral | `proxemic_profile_neutral_issue_1676` | Explicit copy of the base v3 social-distance knobs for comparison. |
| Open | `proxemic_profile_open_issue_1676` | Smaller desired dynamic clearance, lower social weighting, narrower slow zone, higher near-human speed. |

Config files:

- `configs/policy_search/candidates/proxemic_profile_conservative_issue_1676.yaml`
- `configs/policy_search/candidates/proxemic_profile_neutral_issue_1676.yaml`
- `configs/policy_search/candidates/proxemic_profile_open_issue_1676.yaml`

The candidates are registered in `docs/context/policy_search/candidate_registry.yaml` with smoke
and nominal-sanity stages only. They are exploratory diagnostics, not promoted planners.

## Smoke Result

Observed locally on 2026-05-30:

| Candidate | Decision | Episodes | Success | Collision | Near miss | Mean min distance | Report |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `proxemic_profile_conservative_issue_1676` | pass | 1 | 1.0000 | 0.0000 | 0.0000 | n/a | [2026-05-30_proxemic_profile_conservative_issue_1676_smoke.md](policy_search/reports/2026-05-30_proxemic_profile_conservative_issue_1676_smoke.md) |
| `proxemic_profile_neutral_issue_1676` | pass | 1 | 1.0000 | 0.0000 | 0.0000 | n/a | [2026-05-30_proxemic_profile_neutral_issue_1676_smoke.md](policy_search/reports/2026-05-30_proxemic_profile_neutral_issue_1676_smoke.md) |
| `proxemic_profile_open_issue_1676` | pass | 1 | 1.0000 | 0.0000 | 0.0000 | n/a | [2026-05-30_proxemic_profile_open_issue_1676_smoke.md](policy_search/reports/2026-05-30_proxemic_profile_open_issue_1676_smoke.md) |

The smoke stage proves only that the three profile configs are executable through the policy-search
runner. It does not compare comfort because the smoke scenario does not produce robot-pedestrian
clearance samples (`mean_min_distance=n/a`).

## Next Preflight Path

The next useful local command is the `nominal_sanity` stage for all three profiles. Only after that
should a stress-slice or repeated-seed analysis be considered.

## Interpretation Limits

- A profile with higher clearance or lower comfort exposure is not automatically a better social
  navigation policy if success or time-to-goal regresses.
- A profile with higher success is not socially preferable if it increases near misses, force
  exposure, or fallback/degraded rows.
- Missing force arrays make `comfort_exposure` unavailable, not zero.
- These configs do not prove any external Webots or social-stress DRL result transfers to Robot SF.
- Any benchmark report using this slice must preserve fallback/degraded row status according to
  [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md).

## Validation

For this config/docs change:

```bash
git diff --check origin/main...HEAD
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
uv run pytest -q tests/validation/test_run_policy_search_candidate.py
```

Smoke commands and results should be recorded in the PR body. Generated `output/policy_search/...`
artifacts are local and ignored unless a compact evidence summary is deliberately promoted.
