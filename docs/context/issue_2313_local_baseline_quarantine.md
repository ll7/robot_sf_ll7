# Issue #2313 Local Baseline Quarantine

Date: 2026-06-05

## Scope

Issue #2313 follows the Issue #2277 classification of seven absent local-only baseline model
artifacts. This note records the quarantine action only: it does not recover artifacts, invent
replacement checkpoints, delete historical configs, or treat any local `output/` reference as
benchmark evidence.

Related context:

- [Issue #1638 model-path preflight](issue_1638_model_path_preflight.md)
- [Issue #1960 local artifact retirement](issue_1960_local_artifact_retirement.md)
- [Issue #2277 local artifact classification](issue_2277_local_artifact_classification.md)
- `configs/baselines/local_model_artifact_blocklist.yaml`
- `scripts/validation/check_local_model_artifacts.py`
- `scripts/tools/plan_model_artifact_promotion.py`

## Quarantine Decision

The local-artifact blocklist now carries explicit `availability: unavailable` metadata for all
seven absent rows. The scanner still reports them as fail-closed `blocked` rows, but its JSON output
now includes the decision and next action so clean-worktree users see that the rows are unavailable
instead of promotion-ready.

| Config | Availability | Decision |
| --- | --- | --- |
| `configs/baselines/ppo_issue_791_horizon100_12178.yaml` | `unavailable` | `unavailable_recover_or_retire` |
| `configs/baselines/drl_vo_default.yaml` | `unavailable` | `unavailable_recover_or_retire` |
| `configs/baselines/ppo_issue_856_all_scenarios_12223.yaml` | `unavailable` | `unavailable_retire_or_rewrite` |
| `configs/baselines/sac_gate_socnav_struct.yaml` | `unavailable` | `unavailable_retire_or_rewrite` |
| `configs/baselines/sac_gate_socnav_struct_ego.yaml` | `unavailable` | `unavailable_retire_or_rewrite` |
| `configs/baselines/sac_gate_socnav_struct_ego_multi.yaml` | `unavailable` | `unavailable_retire_or_rewrite` |
| `configs/baselines/sac_gate_socnav_struct_ego_safe.yaml` | `unavailable` | `unavailable_retire_or_rewrite` |

The promotion/retirement planner now classifies known blocklisted rows as `classification:
unavailable` when the blocklist provides a decision. This makes the generated plan a quarantine
surface rather than another recover-by-default promotion queue.

## Validation

Evidence summary:
`docs/context/evidence/issue_2313_local_baseline_quarantine_2026-06-05/summary.json`

Validation commands executed:

```bash
scripts/dev/run_worktree_shared_venv.sh -- ruff check robot_sf/benchmark/local_model_artifacts.py scripts/tools/plan_model_artifact_promotion.py tests/validation/test_check_local_model_artifacts.py tests/tools/test_plan_model_artifact_promotion.py tests/benchmark/test_map_runner_utils.py
scripts/dev/run_worktree_shared_venv.sh -- ruff format --check robot_sf/benchmark/local_model_artifacts.py scripts/tools/plan_model_artifact_promotion.py tests/validation/test_check_local_model_artifacts.py tests/tools/test_plan_model_artifact_promotion.py tests/benchmark/test_map_runner_utils.py
scripts/dev/run_worktree_shared_venv.sh -- pytest tests/validation/test_check_local_model_artifacts.py tests/tools/test_plan_model_artifact_promotion.py tests/benchmark/test_map_runner_utils.py::test_parse_algo_config_reports_blocked_local_artifact_follow_up -q
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/check_local_model_artifacts.py configs/baselines --json
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/plan_model_artifact_promotion.py scan --json
python -m json.tool docs/context/evidence/issue_2313_local_baseline_quarantine_2026-06-05/summary.json
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```

Result: seven rows, all `availability=unavailable`; the scanner rows remain `status=blocked`, the
planner rows report `classification=unavailable`, and the focused pytest batch passed 16 tests.

## Boundary

These rows remain historical local/experimental configs. Any future benchmark, paper-facing, or
research-result use still requires a durable checkpoint source with checksum, a model registry entry
or artifact pointer, and fresh benchmark proof.
