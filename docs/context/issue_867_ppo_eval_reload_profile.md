# Issue 867 PPO Evaluation Reload Profile

## Goal

Measure whether issue-791-style PPO evaluation time is materially dominated by fresh environment
construction, SVG/map reload work, or predictive-model reloads after the issue-863 log-deduping
change.

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/867>

## Measurement Scope

The probe used the issue-791 reward-curriculum promotion config as the evaluation contract:

- `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22.yaml`
- evaluation surface: `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`
- three deterministic evaluation episodes, covering the first three maintained-eval scenarios:
  `classic_bottleneck_low`, `classic_bottleneck_medium`, and `classic_bottleneck_high`
- no PPO training; a dummy zero-action model called the same `_evaluate_policy(...)` path used by
  `scripts/training/train_ppo.py`
- predictive foresight forced to CPU for local repeatability

The local output files are ignored and reproducible:

- `output/benchmarks/perf/ppo_eval_reload_issue867.json`
- `output/benchmarks/perf/ppo_eval_reload_issue867.md`
- `output/benchmarks/perf/ppo_eval_reload_issue867_disabled_fresh.json`
- `output/benchmarks/perf/ppo_eval_reload_issue867_disabled_fresh.md`

## Probe Method

The probe wrapped `scripts.training.train_ppo.make_robot_env` and
`robot_sf.planner.socnav.PredictionPlannerAdapter._build_model` during a direct call to
`_evaluate_policy(...)`.

Recorded timing buckets:

- environment construction,
- reset,
- first step,
- total step time for the episode,
- predictive checkpoint load/build time.

Two comparisons were used:

- predictive enabled, fresh process,
- predictive disabled, fresh process.

A third disabled pass in the same process after the predictive-enabled pass showed process-local map
and backend caches, but is not the primary apples-to-apples comparison because it benefits from the
first pass warming the process.

## Evidence

Fresh-process comparison, three 500-step episodes:

| Case | total_sec | model_load_count | env_create_sec | reset_sec | first_step_sec | episode_step_total_sec | model_load_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| predictive enabled | 24.854 | 3 | 0.994 | 0.044 | 1.467 | 5.607 | 0.035 |
| predictive disabled | 23.192 | 0 | 0.736 | 0.007 | 1.484 | 5.038 | 0.000 |

The predictive-enabled pass loaded the predictive checkpoint once per fresh evaluation environment.
Those cached local loads were small:

- episode 1: `0.041s`
- episode 2: `0.035s`
- episode 3: `0.034s`

The larger common costs appeared in both enabled and disabled fresh-process probes:

- first evaluated scenario env construction: about `5.44s` to `5.48s`,
- first evaluated scenario first step: about `4.62s` to `4.67s`,
- later high-density bottleneck first step: about `1.47s` to `1.48s`.

In-process cache effect, for context only:

| Case | total_sec | env_create_sec | reset_sec | first_step_sec | episode_step_total_sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| disabled after enabled pass | 9.948 | 0.010 | 0.007 | 0.006 | 3.272 |

That warm disabled pass confirms there is substantial process-local startup/cache behavior, but it
should not be used as the main predictive-vs-disabled delta.

## Conclusion

PPO evaluation still constructs a fresh environment per evaluation episode, and predictive foresight
therefore builds a fresh `PredictionPlannerAdapter` per episode. However, on this local run the
cached predictive checkpoint reload itself is not the dominant overhead: it is about `0.035s` per
episode.

The dominant observed cost is shared cold startup and lazy first-step work that exists even when
predictive foresight is disabled. Predictive foresight adds some per-step overhead on this slice
(`5.607s` versus `5.038s` median episode step time for 500 steps), but the evidence does not justify
a model-resolver memoization change by itself.

## Decision

No optimization is retained for issue #867.

The current per-episode environment construction remains conservative for reset isolation and
benchmark semantics. A future optimization should not target cached model path resolution first.
If PPO evaluation wall time becomes a blocker, the next measured target should split:

- process-cold map/scenario loading,
- first-step simulator/backend initialization,
- predictive inference cost per step,
- and only then environment reuse or predictive adapter reuse safety.

The issue-863 phase markers remain useful as observability for long evaluation phases.
