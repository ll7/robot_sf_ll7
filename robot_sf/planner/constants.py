"""Shared semantic default constants for the planner subpackage.

These constants replace recurring magic-number literals (see issue #6457) whose
*physical or tuning meaning is identical across multiple planner modules*. Only
values that recur with one clear, shared semantic across modules live here;
within-module repeats stay as module-level constants in their owning module,
and structurally ambiguous literals (loop bounds, array/tensor dimensions, the
quadratic-formula coefficient ``4.0``, docstring coordinate examples, etc.) are
left in place.

Swapping a literal for one of these names must never change a computed value,
control-flow branch, default, unit, or array shape: each name is a direct,
value-preserving reference to the original literal.
"""

from __future__ import annotations

#: Default reward weight on goal-progress for scoring/cost terms across the
#: rule-based and sampling local planners (policy-stack, hybrid rule, MPPI,
#: risk-DWA). Pure tuning weight; behaviorally identical to the literal ``4.0``.
DEFAULT_GOAL_PROGRESS_WEIGHT: float = 4.0

#: Default robot maximum angular (yaw) speed in rad/s shared by the classical
#: and learning-adapter local-planner configs. Behaviorally identical to ``1.2``.
DEFAULT_MAX_ANGULAR_SPEED: float = 1.2

#: Default robot maximum linear speed in m/s shared by the sampling and gap
#: local-planner configs. Behaviorally identical to ``1.2``. Note this is the
#: same numeric value as :data:`DEFAULT_MAX_ANGULAR_SPEED` but a different
#: physical quantity (m/s vs rad/s), so it is kept as a separate constant.
DEFAULT_MAX_LINEAR_SPEED: float = 1.2

#: Default number of Gaussian mixture modes per pedestrian for the learned GMM
#: predictor family and the chance-constrained MPC that embeds that predictor.
#: Behaviorally identical to the literal ``3``.
DEFAULT_GMM_MODE_COUNT: int = 3

#: Default forward lookahead distance (m) for the stream-gap corridor probe: how far ahead of
#: the robot the gap corridor is sampled. Shared by the stream-gap planner config and the
#: gap-prediction adapter that builds that config. Behaviorally identical to ``4.0``.
DEFAULT_FORWARD_LOOKAHEAD_M: float = 4.0

#: Default stream-gap corridor sampling horizon (s) paired with ``sample_dt``. Shared by the
#: stream-gap planner config and the gap-prediction adapter that builds that config.
#: Behaviorally identical to ``4.0``; a different quantity from :data:`DEFAULT_FORWARD_LOOKAHEAD_M`
#: (seconds vs meters) despite the same numeric value, so kept as a separate constant.
DEFAULT_SAMPLE_HORIZON_S: float = 4.0

#: Default wall-clock budget in seconds for the OMPL-based geometric planners.
#: Behaviorally identical to the literal ``5.0``.
DEFAULT_PLANNING_TIME_BUDGET_S: float = 5.0
