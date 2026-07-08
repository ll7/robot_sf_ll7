# Issue #4848 Group-Crossing Exemplar Selection Report

Generated: 2026-07-08T10:21:00.152419+00:00
Total exemplars selected: 9

## Selection Criteria

- Scenario class: group_crossing (low/medium/high density)
- Planners: goal, orca, social_force
- Selection metric: path_efficiency (higher is better)
- Selection modes: median, best, worst

## Selected Episodes

- **goal** / classic_group_crossing_high / seed 24 / best: path_efficiency=1.0000
- **goal** / classic_group_crossing_low / seed 20 / worst: path_efficiency=1.0000
- **goal** / classic_group_crossing_medium / seed 22 / median: path_efficiency=1.0000
- **orca** / classic_group_crossing_high / seed 24 / best: path_efficiency=1.0000
- **orca** / classic_group_crossing_low / seed 20 / worst: path_efficiency=1.0000
- **orca** / classic_group_crossing_medium / seed 22 / median: path_efficiency=1.0000
- **social_force** / classic_group_crossing_low / seed 21 / median: path_efficiency=0.3817
- **social_force** / classic_group_crossing_medium / seed 22 / best: path_efficiency=0.7793
- **social_force** / classic_group_crossing_medium / seed 23 / worst: path_efficiency=0.2447

## Scenario Class Rationale

Group-crossing scenarios were chosen as the second exemplar class to complement the doorway scenario from issue_4253/4268. Group-crossing introduces bidirectional pedestrian flow with social group dynamics (50% of pedestrians in groups), providing richer interaction diversity than single-agent avoidance scenarios. The three density levels (low/medium/high) capture different interaction regimes: sparse crossing, moderate crowd navigation, and high-density stress conditions.

## Planner Rationale

- **goal**: Classical baseline that ignores pedestrians (provides lower bound)
- **orca**: Classical collision-avoidance with time-based navigation (contrast to learned)
- **social_force**: Physics-based social navigation (explicit social force model)

These three planners span the classical-to-social spectrum and provide diverse interaction behaviors for visualization and worked examples.
