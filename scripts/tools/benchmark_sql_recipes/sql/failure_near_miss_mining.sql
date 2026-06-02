WITH collision_metric AS (
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
  e.episode_id,
  e.algo,
  e.scenario_family,
  e.seed,
  e.termination_reason,
  COALESCE(c.collisions, 0.0) AS collisions,
  t.min_ttc,
  cl.clearance
FROM {episodes} AS e
LEFT JOIN collision_metric AS c USING (episode_id)
LEFT JOIN min_ttc_metric AS t USING (episode_id)
LEFT JOIN clearance_metric AS cl USING (episode_id)
WHERE
  e.termination_reason != 'goal_reached'
  OR COALESCE(c.collisions, 0.0) > 0.0
  OR t.min_ttc < 0.5
ORDER BY COALESCE(c.collisions, 0.0) DESC, t.min_ttc ASC NULLS LAST, e.episode_id
