# Issue #2409 Local Baseline Quarantine Decision

Date: 2026-06-06

Issue: [#2409](https://github.com/ll7/robot_sf_ll7/issues/2409)
Parent: [#1764](https://github.com/ll7/robot_sf_ll7/issues/1764)
Predecessor: [#2313](https://github.com/ll7/robot_sf_ll7/issues/2313)
Evidence:
[summary.json](evidence/issue_2409_local_baseline_quarantine_2026-06-06/summary.json)

## Scope

Issue #2409 asked whether the seven local-only baseline artifact rows should be retired or
quarantined instead of treated as recoverable benchmark dependencies. This note records the
follow-on decision after Issue #2313: the rows are already quarantined as absent/unavailable, and no
additional artifact promotion or benchmark evidence exists in this PR.

This is metadata-only. It does not recover checkpoints, upload artifacts, delete historical configs,
or treat any `output/` reference as durable.

## Issue #1764 Update (2026-06-29)

Issue #1764 converted the seven quarantined rows into explicit retired-local-only registry ids.
The baseline configs no longer contain executable `output/` `model_path` references, and the local
artifact blocklist is empty. This does not recover checkpoints, upload artifacts, run benchmarks,
or promote any paper-facing claim.

Current state: `scripts/validation/check_local_model_artifacts.py --fail-on-blocked configs/baselines
--json` returns `[]`; `scripts/tools/plan_model_artifact_promotion.py scan --json` reports seven
`retired_local_only` rows. Future work should recover a durable artifact with checksum or keep the
retirement in place.

## Historical Decision

At the time of issue #2409, all seven local-only baseline rows remained blocked/absent:

| Config | Status | Decision | #2409 classification |
| --- | --- | --- | --- |
| `configs/baselines/ppo_issue_791_horizon100_12178.yaml` | `blocked` / `unavailable` | `unavailable_recover_or_retire` | `quarantined_absent` |
| `configs/baselines/drl_vo_default.yaml` | `blocked` / `unavailable` | `unavailable_recover_or_retire` | `quarantined_absent` |
| `configs/baselines/ppo_issue_856_all_scenarios_12223.yaml` | `blocked` / `unavailable` | `unavailable_retire_or_rewrite` | `retire_or_rewrite_candidate` |
| `configs/baselines/sac_gate_socnav_struct.yaml` | `blocked` / `unavailable` | `unavailable_retire_or_rewrite` | `retire_or_rewrite_candidate` |
| `configs/baselines/sac_gate_socnav_struct_ego.yaml` | `blocked` / `unavailable` | `unavailable_retire_or_rewrite` | `retire_or_rewrite_candidate` |
| `configs/baselines/sac_gate_socnav_struct_ego_multi.yaml` | `blocked` / `unavailable` | `unavailable_retire_or_rewrite` | `retire_or_rewrite_candidate` |
| `configs/baselines/sac_gate_socnav_struct_ego_safe.yaml` | `blocked` / `unavailable` | `unavailable_retire_or_rewrite` | `retire_or_rewrite_candidate` |

Interpretation: Issue #2313 already satisfied the quarantine surface by adding explicit
`availability: unavailable`, `decision`, and `next_action` metadata to
`configs/baselines/local_model_artifact_blocklist.yaml`. The executable scanner still fails closed
for any unblocked promoted/local-only dependency and reports these rows as intentional `blocked`
dependencies with actionable next actions.

## Acceptance Mapping

- Existing Issue #2313 and parent Issue #1764 context checked:
  [issue_2313_local_baseline_quarantine.md](issue_2313_local_baseline_quarantine.md),
  [issue_2277_local_artifact_classification.md](issue_2277_local_artifact_classification.md), and
  [issue_1960_local_artifact_retirement.md](issue_1960_local_artifact_retirement.md).
- At the time of issue #2409, each affected baseline config row had a status: all seven were
  `blocked`/`unavailable`; two are recover-or-retire candidates, and five are retire-or-rewrite
  candidates.
- Remaining local-only dependencies fail closed through
  `scripts/validation/check_local_model_artifacts.py` and
  `robot_sf.benchmark.local_model_artifacts.validate_no_local_model_artifacts`.

## Validation

Executed from the Issue #2409 worktree:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/check_local_model_artifacts.py configs/baselines --json
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/plan_model_artifact_promotion.py scan --json
```

Result: seven rows; all artifact paths are absent in the clean worktree, all rows are
`availability=unavailable`, the scanner rows are `status=blocked`, and the planner rows are
`classification=unavailable`.

## Follow-Up Boundary

Do not split more analysis-only children for these same seven rows unless new evidence changes the
artifact state. Future #1764 children should be concrete actions: recover and publish a named
checkpoint with checksum/provenance, retire or rewrite a named config, or record a maintainer
decision to keep a named config blocked for local diagnostics only.
