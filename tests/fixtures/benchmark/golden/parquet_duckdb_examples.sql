-- Robot SF benchmark Parquet analytics examples.
-- Run from the export directory, or replace the file paths with absolute paths.

-- Grouped safety metrics by planner and scenario family.
WITH metric_values AS (
    SELECT
        e.algo,
        e.scenario_family,
        m.metric_path,
        m.value_number
    FROM read_parquet('episodes.parquet') AS e
    JOIN read_parquet('metrics.parquet') AS m USING (episode_id)
)
SELECT
    algo,
    scenario_family,
    AVG(CASE WHEN metric_path = 'min_ttc' THEN value_number END) AS avg_min_ttc,
    AVG(CASE WHEN metric_path = 'clearance' THEN value_number END) AS avg_clearance,
    SUM(CASE WHEN metric_path = 'collisions' THEN value_number ELSE 0 END) AS collisions
FROM metric_values
GROUP BY algo, scenario_family
ORDER BY algo, scenario_family;

-- Failure and near-miss mining.
SELECT
    e.episode_id,
    e.algo,
    e.scenario_id,
    e.scenario_family,
    e.termination_reason,
    m.value_number AS min_ttc
FROM read_parquet('episodes.parquet') AS e
LEFT JOIN read_parquet('metrics.parquet') AS m
    ON e.episode_id = m.episode_id AND m.metric_path = 'min_ttc'
WHERE e.termination_reason IN ('collision', 'deadlock', 'timeout')
   OR m.value_number < 0.5
ORDER BY e.algo, min_ttc NULLS LAST;
