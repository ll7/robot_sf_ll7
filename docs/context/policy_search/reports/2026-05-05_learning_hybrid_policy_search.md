# 2026-05-05 Learning-Hybrid Policy Search

## Goal

Implement and evaluate a deadline-friendly learning-based local navigation candidate for the
2026-05-05 prompt in `output/tmp/prompt/2026-05-05_policy_search.md`.

The bounded interpretation was: use the existing trained PPO checkpoint as the learning component,
add the fastest plausible ORCA-prior safety/residual mechanism, and benchmark against same-slice
ORCA before considering expensive stress or full-matrix stages.

## Literature Shortlist

| Rank | Candidate | Fit | Decision |
|---:|---|---|---|
| 1 | ORCA-prior / residual guarded PPO | Uses existing PPO checkpoint and ORCA safety baseline; directly matches the prompt bias toward hybrid RL plus ORCA. | Implemented. |
| 2 | Dynamic safety shield / RL-tuned controller | 2025 safety-shield work supports combining optimization controllers with RL prediction/control, but needs new training surface. | Deferred. |
| 3 | Classical/RL planner switcher | 2024 hybrid social-navigation and local-planner work supports switching between geometric and learned heads. | Partly represented by ORCA-primary variant; richer selector deferred. |
| 4 | Learned-risk MPC / waypoint risk model | 2025 LR-MPC style risk estimator fits the repo's existing learned-risk handoff, but requires dataset/training. | Deferred to learned-risk-model handoff. |
| 5 | New sequence/attention DRL policy | CAMRL/Mamba and recent DRL surveys suggest promise, but implementation/training exceeds the local deadline. | Not attempted. |

Source anchors:

- DRL navigation survey, 2024: https://link.springer.com/article/10.1007/s10846-024-02198-w
- Dynamic safety shield, 2025: https://www.hrl.uni-bonn.de/publications/2025/dawood25l4dc
- Rethinking Social Robot Navigation, ICRA 2024: https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-karnansocial2024.html
- Hybrid Classical/RL Local Planner, 2024: https://arxiv.org/abs/2410.03066
- LR-MPC risk adaptation, 2025: https://arxiv.org/abs/2506.14305
- CAMRL, 2024: https://arxiv.org/abs/2408.02661

## Implementation

Changed:

- `robot_sf/planner/guarded_ppo.py`
  - added optional planner-prior support for `GuardedPPOAdapter`,
  - added ORCA prior construction via `prior_policy: orca`,
  - added residual command blending with safety/progress checks,
  - added optional ORCA fallback construction while retaining risk-DWA default behavior.
- `robot_sf/benchmark/map_runner.py`
  - wires the optional guarded-PPO prior into benchmark policy construction,
  - records `prior_blend_safe` and `prior_safe` guard decisions.
- `configs/policy_search/candidates/orca_prior_guarded_ppo_v1.yaml`
- `configs/policy_search/candidates/orca_prior_guarded_ppo_v2_static_global.yaml`
- `configs/policy_search/candidates/orca_primary_guarded_ppo_v1.yaml`
- `configs/algos/orca_prior_guarded_ppo_v1.yaml`
- `configs/benchmarks/paper_experiment_matrix_v1_orca_prior_guarded_ppo_compare.yaml`
- `docs/context/policy_search/candidate_registry.yaml`
- `docs/context/policy_search/experiment_ledger.md`
- `tests/planner/test_guarded_ppo.py`

Training lineage:

- No fresh long PPO training was launched. The candidate uses existing model id
  `ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200`.
- Registry source: `model/registry.yaml`.
- W&B source run: `ll7/robot_sf/b60iopxt`, state `finished`.
- Local lineage capture:
  `output/policy_search/20260505T134203+0200/training_lineage/ppo_br06_v3_training_lineage.json`.

## Validation Commands

```bash
uv run pytest tests/planner/test_guarded_ppo.py -q
uv run ruff check robot_sf/planner/guarded_ppo.py robot_sf/benchmark/map_runner.py tests/planner/test_guarded_ppo.py
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 uv run python scripts/validation/run_policy_search_candidate.py --candidate orca_prior_guarded_ppo_v1 --stage smoke --output-dir output/policy_search/20260505T134203+0200/orca_prior_guarded_ppo_v1/smoke --workers 1
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 uv run python scripts/validation/run_policy_search_candidate.py --candidate orca_prior_guarded_ppo_v1 --stage nominal_sanity --output-dir output/policy_search/20260505T134203+0200/orca_prior_guarded_ppo_v1/nominal_sanity --workers 1
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 uv run python scripts/validation/run_policy_search_candidate.py --candidate orca_prior_guarded_ppo_v2_static_global --stage smoke --output-dir output/policy_search/20260505T134203+0200/orca_prior_guarded_ppo_v2_static_global/smoke --workers 1
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 uv run python scripts/validation/run_policy_search_candidate.py --candidate orca_prior_guarded_ppo_v2_static_global --stage nominal_sanity --output-dir output/policy_search/20260505T134203+0200/orca_prior_guarded_ppo_v2_static_global/nominal_sanity --workers 1
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 uv run python scripts/validation/run_policy_search_candidate.py --candidate orca_primary_guarded_ppo_v1 --stage smoke --output-dir output/policy_search/20260505T134203+0200/orca_primary_guarded_ppo_v1/smoke --workers 1
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 uv run python scripts/validation/run_policy_search_candidate.py --candidate orca_primary_guarded_ppo_v1 --stage nominal_sanity --output-dir output/policy_search/20260505T134203+0200/orca_primary_guarded_ppo_v1/nominal_sanity --workers 1
```

Raw ORCA same-slice comparison was run programmatically through
`scripts.validation.run_policy_search_candidate._run_stage_eval(...)` with `algo='orca'` and
`allow_fallback: false`.

The named paper-matrix comparison was run with:

```bash
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/paper_experiment_matrix_v1_orca_prior_guarded_ppo_compare.yaml --output-root output/policy_search/20260505T134203+0200/camera_ready --campaign-id paper_orca_prior_guarded_ppo_20260505 --skip-publication-bundle --log-level WARNING
```

## Results

| Candidate | Stage | Episodes | Success | Collision | Near Miss | Decision |
|---|---|---:|---:|---:|---:|---|
| orca | nominal_sanity | 18 | 0.2222 | 0.0556 | 0.2222 | reference |
| orca_prior_guarded_ppo_v1 | smoke | 1 | 1.0000 | 0.0000 | 0.0000 | pass |
| orca_prior_guarded_ppo_v1 | nominal_sanity | 18 | 0.2778 | 0.1111 | 0.1667 | revise |
| orca_prior_guarded_ppo_v2_static_global | smoke | 1 | 1.0000 | 0.0000 | 0.0000 | pass |
| orca_prior_guarded_ppo_v2_static_global | nominal_sanity | 18 | 0.2778 | 0.1111 | 0.1667 | revise |
| orca_primary_guarded_ppo_v1 | smoke | 1 | 1.0000 | 0.0000 | 0.0000 | pass |
| orca_primary_guarded_ppo_v1 | nominal_sanity | 18 | 0.1667 | 0.0000 | 0.1667 | revise |

Paper-matrix S3 (`configs/benchmarks/paper_experiment_matrix_v1_orca_prior_guarded_ppo_compare.yaml`,
seed set `eval` = 111/112/113):

| Planner | Episodes | Success | Collision | Near Miss | TTG Norm | SNQI |
|---|---:|---:|---:|---:|---:|---:|
| orca | 144 | 0.1806 | 0.0347 | 4.9097 | 0.9650 | -0.2589 |
| orca_prior_guarded_ppo_v1 | 144 | 0.0556 | 0.0764 | 3.3056 | 0.9840 | -0.4902 |

Campaign artifacts:

- `output/policy_search/20260505T134203+0200/camera_ready/paper_orca_prior_guarded_ppo_20260505/reports/campaign_table.md`
- `output/policy_search/20260505T134203+0200/camera_ready/paper_orca_prior_guarded_ppo_20260505/reports/snqi_diagnostics.md`

## Interpretation

Observed:

- The PPO-plus-ORCA residual variants (`v1`, `v2`) show a useful success trend over same-slice
  ORCA: `0.2778` versus `0.2222`.
- That gain is not promotion-quality because static collisions increase from `0.0556` to `0.1111`.
- The ORCA-primary variant removes nominal collisions entirely, but success falls to `0.1667`,
  below same-slice ORCA, due to low-progress timeouts.
- No learning-hybrid candidate passed the nominal gate (`success >= 0.80`, `collision <= 0.02`),
  so stress-slice or full-matrix escalation is not justified.
- A scoped paper-matrix run was still executed to satisfy the 2026-05-05 prompt's named benchmark
  requirement. On that surface, `orca_prior_guarded_ppo_v1` is clearly worse than ORCA on success,
  collision rate, and SNQI, despite fewer near misses.

Conclusion:

The implemented ORCA-prior guarded PPO branch is useful as a benchmarkable experimental learning
hybrid, but it should be rejected as a policy-search winner. The current checkpoint plus
action-level guard cannot dominate ORCA on both progress and static safety, and the paper-matrix
comparison shows the nominal success trend does not transfer. The best next iteration is not
another guard threshold tweak; it is a trained residual policy or learned risk scorer that sees
ORCA command/risk features during training.

## Follow-up Boundary

Recommended next work:

- Add ORCA suggested `(v, omega)` and short-horizon guard diagnostics to the PPO observation or
  auxiliary training target.
- Train a small residual policy that predicts a bounded delta over ORCA rather than blending PPO
  and ORCA only at inference time.
- Reuse the same nominal gate before any stress or full-matrix run.

Do not promote any 2026-05-05 learning-hybrid variant from this note.
