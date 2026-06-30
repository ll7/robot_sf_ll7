# Issue #1960 Local Artifact Retirement

Issue: [#1960](https://github.com/ll7/robot_sf_ll7/issues/1960)
Parent: [#1764](https://github.com/ll7/robot_sf_ll7/issues/1764)
Evidence: [summary.json](evidence/issue_1960_local_artifact_retirement_2026-06-01/summary.json)

## Scope

This note records the safe local side of the #1764 artifact-retirement plan. It does not upload,
promote, delete, or benchmark any model artifact. The scanner output is metadata/provenance
guidance only; none of these rows is benchmark evidence until a durable artifact source exists.

The target configs now carry human-visible comments, and
`configs/baselines/local_model_artifact_blocklist.yaml` records the current classification and next
action next to each exact blocked local path. The executable preflight remains fail-closed: these
configs are still blocked until #1764 recovers durable artifacts or retires/rewrites them.

## Issue #1764 Update (2026-06-29)

Issue #1764 retired the seven remaining local-only baseline config references from executable
`model_path` fields. The configs now point at explicit `local_only: true` registry ids, and
`configs/baselines/local_model_artifact_blocklist.yaml` is empty. This remains metadata-only:
the retired ids are not benchmark evidence and must not be used for benchmark claims unless a
durable artifact pointer and checksum are recovered later.

Current expected checks:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/validation/check_local_model_artifacts.py --fail-on-blocked configs/baselines --json
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/tools/plan_model_artifact_promotion.py scan --json
```

Expected state after issue #1764: local-artifact preflight reports `[]`, and the promotion planner
reports seven `retired_local_only` rows for the historical target configs.

## Historical Decisions

| Config | Status | Decision | Next action |
| --- | --- | --- | --- |
| `configs/baselines/ppo_issue_791_horizon100_12178.yaml` | `missing` | `recover_before_promotion` | Recover a durable checkpoint source with checksum, or retire the config. |
| `configs/baselines/ppo_issue_856_all_scenarios_12223.yaml` | `retire_candidate` | `recover_or_retire` | Recover the local SLURM checkpoint and prove provenance only if it remains useful; otherwise retire or rewrite. |
| `configs/baselines/sac_gate_socnav_struct.yaml` | `retire_candidate` | `recover_or_retire` | Recover and prove provenance only if still useful; otherwise retire or rewrite. |
| `configs/baselines/sac_gate_socnav_struct_ego.yaml` | `retire_candidate` | `recover_or_retire` | Recover and prove provenance only if still useful; otherwise retire or rewrite. |
| `configs/baselines/sac_gate_socnav_struct_ego_multi.yaml` | `retire_candidate` | `recover_or_retire` | Recover and prove provenance only if still useful; otherwise retire or rewrite. |
| `configs/baselines/sac_gate_socnav_struct_ego_safe.yaml` | `retire_candidate` | `recover_or_retire` | Recover and prove provenance only if still useful; otherwise retire or rewrite. |
| `configs/baselines/drl_vo_default.yaml` | `missing` | `recover_before_promotion` | Recover a durable checkpoint source with checksum, or retire the config. |

## Validation

Run from the issue worktree:

```bash
uv run python scripts/tools/plan_model_artifact_promotion.py scan
uv run python scripts/validation/check_local_model_artifacts.py --json
```

Expected state on 2026-06-01: the scanner reports seven rows and the local-artifact preflight
reports the same seven rows as `blocked`. A nonzero strict audit with
`--fail-on-blocked` remains expected until #1764 resolves these local-only references.
