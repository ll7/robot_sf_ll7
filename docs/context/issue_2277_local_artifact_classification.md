# Issue #2277 Local Artifact Classification

Date: 2026-06-05

## Scope

This note classifies the remaining local-only baseline model artifacts for Issue #2277 and parent
Issue #1764. It does not promote artifacts, upload checkpoints, delete configs, rewrite model
references, or treat any missing local `output/` file as benchmark evidence.

Related context:

- [Issue #1638 model-path preflight](issue_1638_model_path_preflight.md)
- [Issue #1960 local artifact retirement](issue_1960_local_artifact_retirement.md)
- `configs/baselines/local_model_artifact_blocklist.yaml`
- `scripts/validation/check_local_model_artifacts.py`
- `scripts/tools/plan_model_artifact_promotion.py`

## Current Classification

The current clean-worktree scan found seven direct `model_path: output/...` baseline references.
All seven are explicitly blocklisted and absent locally in this worktree. No unblocked local-only
baseline model dependency was found.

| Config | Local artifact class | Status | Recommended next action |
| --- | --- | --- | --- |
| `configs/baselines/ppo_issue_791_horizon100_12178.yaml` | `durable-required` | `recoverable_or_retire` | Recover a durable checkpoint source with checksum only if this horizon-100 PPO candidate still matters; otherwise retire or rewrite the config. |
| `configs/baselines/drl_vo_default.yaml` | `durable-required` | `recoverable_or_retire` | Recover a durable DRL-VO checkpoint source with checksum before any benchmark use; otherwise retire or rewrite the config. |
| `configs/baselines/ppo_issue_856_all_scenarios_12223.yaml` | `non-evidence-local-only` | `retire_candidate` | Recover and prove the SLURM checkpoint provenance only if this Issue #856 comparison remains useful; otherwise retire or rewrite. |
| `configs/baselines/sac_gate_socnav_struct.yaml` | `non-evidence-local-only` | `retire_candidate` | Recover and prove provenance only if the experimental SAC baseline remains useful; otherwise retire or rewrite. |
| `configs/baselines/sac_gate_socnav_struct_ego.yaml` | `non-evidence-local-only` | `retire_candidate` | Recover and prove provenance only if the ego-frame SAC baseline remains useful; otherwise retire or rewrite. |
| `configs/baselines/sac_gate_socnav_struct_ego_multi.yaml` | `non-evidence-local-only` | `retire_candidate` | Recover and prove provenance only if the multi-scenario SAC baseline remains useful; otherwise retire or rewrite. |
| `configs/baselines/sac_gate_socnav_struct_ego_safe.yaml` | `non-evidence-local-only` | `retire_candidate` | Recover and prove provenance only if the safety-shaped SAC baseline remains useful; otherwise retire or rewrite. |

Interpretation: the two `durable-required` rows are plausible recovery candidates only if their
source checkpoint can be located and checksummed. The five `retire_candidate` rows look more likely
to become retire/rewrite work unless a maintainer still needs their specific experimental
comparisons.

## Validation

Executed from commit `3a7475f0919040826f832ecf0f938e0b9a22f088`:

```bash
uv run python scripts/validation/check_local_model_artifacts.py configs/baselines --json
```

Result: seven rows, all `status=blocked`, `surface=local_experimental`.

```bash
uv run python scripts/tools/plan_model_artifact_promotion.py scan --json
```

Result: seven rows; all local artifacts report `exists=false`. The planner output is metadata and
provenance guidance only, not benchmark evidence.

```bash
rg -n "^(model_path|resume_from):\\s*(output|results)/" configs/baselines configs/training
```

Result: seven direct `model_path` rows under `configs/baselines`; no training `resume_from` rows
matched the current `output/` or `results/` pattern.

Machine-readable summary:
`docs/context/evidence/issue_2277_local_artifact_classification_2026-06-05/summary.json`

## Follow-Up Boundary

Future #1764 work should split only if it is about a concrete action:

- recover and promote a named checkpoint with source, checksum, license/access, and registry entry,
- retire or rewrite a named stale config,
- or document a maintainer decision to keep a config blocked for local diagnostics only.

Until then, these rows must remain blocked and must not be used as benchmark, paper-facing, or
research-result evidence.
