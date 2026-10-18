"""Dependency-free checkpoint roster shared by runtime smoke and context observation."""

_RUNTIME_SMOKE_CHECKPOINT_PLANNER_KEYS = frozenset(
    {"prediction_planner", "ppo", "sacadrl", "guarded_ppo", "predictive_mppi"}
)
