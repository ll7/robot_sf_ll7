"""Shared semantic default constants for the planner subpackage.

These constants replace recurring magic-number literals (see issue #6457) whose
meaning is identical across multiple planner modules because those modules
implement or mirror the same configuration contract. Algorithm-specific tuning
defaults stay private to their owning module even when their names and values
coincide; changing one planner's default must not silently retune another.
Structurally ambiguous literals (loop bounds, array/tensor dimensions, the
quadratic-formula coefficient ``4.0``, docstring coordinate examples, etc.) are
left in place.

Swapping a literal for one of these names must never change a computed value,
control-flow branch, default, unit, or array shape: each name is a direct,
value-preserving reference to the original literal.
"""

from __future__ import annotations

#: Default number of Gaussian mixture modes per pedestrian for the learned GMM
#: predictor family and the chance-constrained MPC that embeds that predictor.
#: Behaviorally identical to the literal ``3``.
DEFAULT_GMM_MODE_COUNT: int = 3

#: Default forward lookahead distance (m) for the stream-gap corridor probe: how far ahead of
#: the robot the gap corridor is sampled. Shared by the stream-gap planner config and the
#: gap-prediction adapter that builds that config. Behaviorally identical to ``4.0``.
DEFAULT_STREAM_GAP_FORWARD_LOOKAHEAD_M: float = 4.0

#: Default stream-gap corridor sampling horizon (s) paired with ``sample_dt``. Shared by the
#: stream-gap planner config and the gap-prediction adapter that builds that config.
#: Behaviorally identical to ``4.0``; a different quantity from
#: :data:`DEFAULT_STREAM_GAP_FORWARD_LOOKAHEAD_M` (seconds vs meters) despite the same numeric
#: value, so kept as a separate constant.
DEFAULT_STREAM_GAP_SAMPLE_HORIZON_S: float = 4.0

#: Stream-gap's default robot command limits. These are shared only with the
#: gap-prediction adapter because that adapter mirrors ``StreamGapPlannerConfig``.
DEFAULT_STREAM_GAP_MAX_LINEAR_SPEED: float = 1.2
DEFAULT_STREAM_GAP_MAX_ANGULAR_SPEED: float = 1.2

#: Default stream-gap commit-hold step count: how many consecutive steps the planner holds its
#: commit command once a gap is committed. Shared by ``StreamGapPlannerConfig`` and the
#: GapPrediction adapter that builds that exact config. Behaviorally identical to ``6``.
DEFAULT_STREAM_GAP_COMMIT_HOLD_STEPS: int = 6

__all__ = [
    "DEFAULT_GMM_MODE_COUNT",
    "DEFAULT_STREAM_GAP_COMMIT_HOLD_STEPS",
    "DEFAULT_STREAM_GAP_FORWARD_LOOKAHEAD_M",
    "DEFAULT_STREAM_GAP_MAX_ANGULAR_SPEED",
    "DEFAULT_STREAM_GAP_MAX_LINEAR_SPEED",
    "DEFAULT_STREAM_GAP_SAMPLE_HORIZON_S",
]
