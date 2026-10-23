# Per-step reward smoothness

For route-completion and social-quality profiles, `smoothness` now reads `jerk_step`,
the current magnitude of the third finite difference of robot position, in m/s³.
It is clipped to [0, 5] and divided by 5 before weighting. During finite-difference
warmup, invalid timestep or missing current-step data, the penalty is zero.

`jerk_mean` remains the episode running mean for metrics and SNQI; it never substitutes
for the current-step reward. A smooth step after a jerk spike therefore pays no historical
jerk cost. Proxy state is reset and primed at every environment reset as before.

This changes future training rewards only, not benchmark evaluation of existing checkpoints.
No observation, action, termination, canonical benchmark metric or frozen artifact is changed.
No training or evaluation campaign was run.
