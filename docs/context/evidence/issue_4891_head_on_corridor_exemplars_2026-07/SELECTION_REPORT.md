<!-- AI-GENERATED (robot_sf#4891, 2026-07-09) - NEEDS-REVIEW -->
# Issue #4891 Head-On Corridor Exemplar Selection Report

Generated: 2026-07-09T12:00:00+00:00
Total exemplars selected: 9

## Selection Criteria

- Scenario class: head-on corridor (low/medium density)
- Planners: goal, orca, social_force
- Selection metric: path_efficiency (higher is better)
- Selection modes: median, best, worst

## Selected Episodes

- **goal** / classic_head_on_corridor_low / seed 24 / best: path_efficiency=1.0000
- **goal** / classic_head_on_corridor_medium / seed 21 / median: path_efficiency=1.0000
- **goal** / classic_head_on_corridor_medium / seed 23 / worst: path_efficiency=1.0000
- **orca** / classic_head_on_corridor_low / seed 22 / median: path_efficiency=1.0000
- **orca** / classic_head_on_corridor_medium / seed 23 / worst: path_efficiency=1.0000
- **orca** / classic_head_on_corridor_medium / seed 24 / best: path_efficiency=1.0000
- **social_force** / classic_head_on_corridor_medium / seed 21 / median: path_efficiency=1.0000
- **social_force** / classic_head_on_corridor_medium / seed 22 / best: path_efficiency=1.0000
- **social_force** / classic_head_on_corridor_medium / seed 24 / worst: path_efficiency=0.6836

## Scenario Class Rationale

Head-on corridor scenarios were chosen as the third exemplar class to complement the doorway scenario (issue_4253/4268) and group-crossing scenario (issue #4848). Head-on corridor introduces opposing pedestrian flows in a constrained space, creating rich navigation challenges where the robot must negotiate right-of-way with oncoming pedestrians. The two density levels (low/medium) capture different interaction regimes: sparse head-on encounters and moderate corridor crowding.

## Interaction Diversity

Head-on corridor provides richer interaction diversity than bottleneck scenarios:
- Consistent pedestrian presence (2-4 pedestrians per episode)
- Opposing flow patterns requiring negotiation
- Constrained space amplifying social navigation challenges
- Mix of success and collision outcomes showing planner sensitivity

## Planner Rationale

- **goal**: Classical baseline that ignores pedestrians (provides lower bound)
- **orca**: Classical collision-avoidance with time-based navigation (contrast to learned)
- **social_force**: Physics-based social navigation (explicit social force model)

These three planners span the classical-to-social spectrum and provide diverse interaction behaviors for visualization and worked examples.

<!-- /AI-GENERATED -->
