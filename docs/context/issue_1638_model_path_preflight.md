# Issue #1638 Local Model Path Preflight - 2026-05-30

Date: 2026-05-30

Related issue:

- Issue #1638: <https://github.com/ll7/robot_sf_ll7/issues/1638>

Follow-up:

- Issue #1764: <https://github.com/ll7/robot_sf_ll7/issues/1764>
- Issue #1775: <https://github.com/ll7/robot_sf_ll7/issues/1775>

Related anchors:

- Durable artifact references:
  `docs/context/issue_1053_durable_artifact_references.md`
- Artifact evidence vocabulary:
  `docs/context/artifact_evidence_vocabulary.md`
- Model registry:
  `model/registry.yaml`

## Outcome

This slice removes the remaining silent BR-06 PPO `output/model_cache` pins that already had durable
registry targets:

- `configs/baselines/ppo_issue_576_br06_v2_15m.yaml`
- `configs/baselines/ppo_15m_grid_socnav_holonomic.yaml`

Both now resolve through:

- `model_id: ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200`

The model registry already carries the GitHub release artifact, checksum, W&B provenance, and local
cache path for that model. The configs no longer need to duplicate the local cache path.

## Preflight

The new checker is:

```bash
uv run python scripts/validation/check_local_model_artifacts.py configs/baselines
```

It scans YAML for `model_path` and `resume_from` values under local `output/model_cache/`,
`output/models/`, or `output/slurm/` paths. Unlisted references fail as `UNBLOCKED`, which means the
config silently depends on one developer worktree. Known unresolved references must be listed in:

- `configs/baselines/local_model_artifact_blocklist.yaml`

The blocklist points at Issue #1764 and records why each remaining local artifact still needs
promotion or retirement.

The same fail-closed check is also called when benchmark algorithm configs are loaded through the
map runner and lightweight runner. A benchmark launch that points at a blocked `output/` model path
now fails before planner construction, with the blocklist reason and follow-up issue in the error
trail.

The promoted-config surface is explicit in:

- `configs/benchmarks/promoted_config_surfaces.yaml`

Configs listed there are treated as benchmark-promoted inputs because camera-ready, paper-facing,
or benchmark-baseline workflows cite them as reusable benchmark surfaces. They fail the preflight
whenever `model_path` or `resume_from` points into `output/`, even if that exact reference is
listed in the local-artifact blocklist. Non-promoted local or experimental configs may remain
blocklisted while Issue #1764 decides whether to promote, replace, or retire those artifacts.

Strict audits can run:

```bash
uv run python scripts/validation/check_local_model_artifacts.py \
  --fail-on-blocked configs/baselines
```

That exits nonzero until every blocked artifact is promoted, replaced with `model_id`, or the config
is retired.

## Remaining Blockers

The remaining local-only configs are not treated as successful benchmark dependencies. They are
explicitly blocked pending Issue #1764:

- `configs/baselines/ppo_issue_791_horizon100_12178.yaml`
- `configs/baselines/ppo_issue_856_all_scenarios_12223.yaml`
- `configs/baselines/sac_gate_socnav_struct.yaml`
- `configs/baselines/sac_gate_socnav_struct_ego.yaml`
- `configs/baselines/sac_gate_socnav_struct_ego_multi.yaml`
- `configs/baselines/sac_gate_socnav_struct_ego_safe.yaml`
- `configs/baselines/drl_vo_default.yaml`

The audit also found three training-only `resume_from: output/slurm/...` warm-start ablation
configs under `configs/training/ppo/ablations/`. They are not benchmark-facing baseline configs and
remain out of this first preflight gate so developer-only resume experiments are not broken. They
should still move to `resume_model_id` if those checkpoints become durable dependencies.

## Validation

Validation commands:

```bash
uv run pytest tests/validation/test_check_local_model_artifacts.py \
  tests/benchmark/test_ppo_baseline_contract.py \
  tests/benchmark/test_map_runner_utils.py::test_parse_algo_config_validates_yaml \
  tests/benchmark/test_map_runner_utils.py::test_parse_algo_config_rejects_local_output_model_path \
  tests/benchmark/test_map_runner_utils.py::test_parse_algo_config_reports_blocked_local_artifact_follow_up \
  tests/baselines/test_ppo_planner.py::test_issue_791_portable_baseline_uses_registry_and_auto_device \
  tests/baselines/test_ppo_planner.py::test_load_model_resolves_registry_model_id -q
uv run python scripts/validation/check_local_model_artifacts.py configs/baselines
rg -n "^(model_path|resume_from):\\s*output/" configs/baselines configs/training
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
