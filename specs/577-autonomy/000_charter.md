# 577 Autonomous Portfolio Charter

## Mission
Build a portfolio of planners (predictive and non-predictive) and maximize route-complete success under v2 semantics.

## Success targets
- Primary: >=80% global success on classic_interactions.
- Stretch: >=90% global success.

## Hard constraints
- No contradiction records.
- Reproducible ranking with fixed seed.
- Every experiment logged in `030_experiment_registry.md`.

## Portfolio families
- prediction_planner (existing + upgrades)
- risk_dwa (new non-learning)
- mppi_social (new sampled non-learning)
- hybrid_portfolio (new switching controller)
