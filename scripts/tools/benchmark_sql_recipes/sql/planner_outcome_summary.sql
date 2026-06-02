WITH success_metric AS (
  SELECT episode_id, CAST(value_bool AS DOUBLE) AS success
  FROM {metrics}
  WHERE metric_path = 'success'
),
collision_metric AS (
  SELECT episode_id, value_number AS collisions
  FROM {metrics}
  WHERE metric_path = 'collisions'
),
min_ttc_metric AS (
  SELECT episode_id, value_number AS min_ttc
  FROM {metrics}
  WHERE metric_path = 'min_ttc'
),
clearance_metric AS (
  SELECT episode_id, value_number AS clearance
  FROM {metrics}
  WHERE metric_path = 'clearance'
)
SELECT
  e.algo,
  e.scenario_family,
  COUNT(*) AS episode_count,
  AVG(s.success) AS success_rate,
  AVG(CASE WHEN COALESCE(c.collisions, 0.0) > 0.0 THEN 1.0 ELSE 0.0 END) AS collision_rate,
  AVG(t.min_ttc) AS mean_min_ttc,
  AVG(cl.clearance) AS mean_clearance
FROM {episodes} AS e
LEFT JOIN success_metric AS s USING (episode_id)
LEFT JOIN collision_metric AS c USING (episode_id)
LEFT JOIN min_ttc_metric AS t USING (episode_id)
LEFT JOIN clearance_metric AS cl USING (episode_id)
GROUP BY e.algo, e.scenario_family
ORDER BY e.algo, e.scenario_family
