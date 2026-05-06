# Best Learning Policy Trial (2026-05-05)

## Goal

Try the strongest learning-based policy already available in the repository and evaluate whether it
is a better candidate than the guarded-PPO variants from the same day. This was an evaluation and
integration pass, not a new training run.

## Candidate Surfaces

- Candidate config: `configs/policy_search/candidates/ppo_issue791_best_v1.yaml`
- Camera-ready comparison config: `configs/benchmarks/paper_experiment_matrix_v1_best_ppo_compare.yaml`
- Base PPO config: `configs/baselines/ppo_15m_grid_socnav.yaml`
- Model registry entry: `model/registry.yaml`
- Prior verdict: `docs/context/issue_791_best_policy_verdict_2026_04_21.md`

The candidate uses model id
`ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`, sourced from W&B run
`ll7/robot_sf/ibo3aqus` and artifact
`ll7/robot_sf/ppo_expert_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity-best-success:v9`.
The base config keeps predictive foresight enabled with
`predictive_foresight_model_id: predictive_proxy_selected_v2_full`,
`predictive_foresight_device: cuda`, and `fallback_to_goal: false`.

## W&B Availability

Availability was verified from this checkout on 2026-05-05:

```bash
curl -L -s -o /dev/null -w 'http_status=%{http_code}\n' \
  'https://wandb.ai/ll7/robot_sf?nw=nwuserll7'

curl --netrc -sS --max-time 15 -H 'Content-Type: application/json' \
  -X POST https://api.wandb.ai/graphql \
  -d '{"query":"query Project($entity:String!, $project:String!){project(name:$project, entityName:$entity){name entity{name}}}","variables":{"entity":"ll7","project":"robot_sf"}}'

curl --netrc -sS --max-time 15 -H 'Content-Type: application/json' \
  -X POST https://api.wandb.ai/graphql \
  -d '{"query":"query Run($entity:String!, $project:String!, $run:String!){project(name:$project, entityName:$entity){run(name:$run){name displayName state}}}","variables":{"entity":"ll7","project":"robot_sf","run":"ibo3aqus"}}'
```

Observed result: the project URL returned HTTP 200, the GraphQL project query returned
`robot_sf` under entity `ll7`, and run `ibo3aqus` resolved as `finished` with display name
`ppo_expert_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_20260417T161217`.
The smoke run also hydrated the model artifact through the registry path.

## Validation

```bash
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 \
  uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate ppo_issue791_best_v1 \
  --stage smoke \
  --output-dir output/policy_search/20260505_best_learning/ppo_issue791_best_v1/smoke \
  --workers 1

LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 \
  uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate ppo_issue791_best_v1 \
  --stage nominal_sanity \
  --output-dir output/policy_search/20260505_best_learning/ppo_issue791_best_v1/nominal_sanity \
  --workers 1

LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 \
  uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_best_ppo_compare.yaml \
  --output-root output/policy_search/20260505_best_learning/camera_ready \
  --label preflight \
  --skip-publication-bundle \
  --mode preflight \
  --log-level WARNING

LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 \
  uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_best_ppo_compare.yaml \
  --output-root output/policy_search/20260505_best_learning/camera_ready \
  --campaign-id paper_best_ppo_20260505 \
  --skip-publication-bundle \
  --log-level WARNING
```

## Results

| Scope | Planner | Episodes | Success | Collision | Near misses | TTG norm | SNQI surface | Decision |
|---|---|---:|---:|---:|---:|---:|---|---|
| smoke | ppo_issue791_best_v1 | 1 | 1.0000 | 0.0000 | 0.0000 | n/a | n/a | pass |
| nominal_sanity | ppo_issue791_best_v1 | 18 | 0.2778 | 0.0000 | 0.2222 | n/a | n/a | revise |
| paper_matrix_s3 | ORCA | 144 | 0.1806 | 0.0347 | 4.9097 | 0.9650 | table -0.2589, diagnostics -0.145440 | reference |
| paper_matrix_s3 | PPO | 144 | 0.2569 | 0.0903 | 3.3403 | 0.9305 | table -0.2953, diagnostics -0.129115 | tracked |

Paper-matrix artifacts:

- Campaign root: `output/policy_search/20260505_best_learning/camera_ready/paper_best_ppo_20260505`
- Campaign table: `output/policy_search/20260505_best_learning/camera_ready/paper_best_ppo_20260505/reports/campaign_table.md`
- Campaign report: `output/policy_search/20260505_best_learning/camera_ready/paper_best_ppo_20260505/reports/campaign_report.md`
- SNQI diagnostics: `output/policy_search/20260505_best_learning/camera_ready/paper_best_ppo_20260505/reports/snqi_diagnostics.md`

## Conclusion

`ppo_issue791_best_v1` is the strongest learning-only candidate tested in this pass. On the scoped
camera-ready comparison it improves success over ORCA (0.2569 vs 0.1806), reduces near misses
(3.3403 vs 4.9097), improves normalized time-to-goal (0.9305 vs 0.9650), and ranks first in the
SNQI diagnostics recomputation (-0.129115 vs -0.145440).

It is not a safety promotion. The collision rate is worse than ORCA (0.0903 vs 0.0347), and the
campaign table's aggregate `snqi_mean` is lower for PPO (-0.2953 vs -0.2589). Treat this as the best
available learned-policy baseline for success-oriented comparison, while keeping ORCA or other
collision-safer policies as the safety anchor.
