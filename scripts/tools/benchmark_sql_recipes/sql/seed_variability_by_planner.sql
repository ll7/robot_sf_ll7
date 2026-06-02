WITH per_seed AS (
  SELECT
    e.algo,
    e.scenario_family,
    e.seed,
    AVG(CAST(m.value_bool AS DOUBLE)) AS seed_success_rate
  FROM {episodes} AS e
  JOIN {metrics} AS m USING (episode_id)
  WHERE m.metric_path = 'success'
  GROUP BY e.algo, e.scenario_family, e.seed
)
SELECT
  algo,
  scenario_family,
  COUNT(*) AS seed_count,
  AVG(seed_success_rate) AS mean_seed_success_rate,
  STDDEV_SAMP(seed_success_rate) AS success_rate_stddev
FROM per_seed
GROUP BY algo, scenario_family
ORDER BY algo, scenario_family
