# Short-window goal progress reward

`RobotEnv` supplies net distance-to-final-goal progress over the latest ten simulation steps
(`goal_progress_window`, metres), including the reset position. At the default dt=0.1 s this
is a one-second window. `goal_progress_window_ready` is false for the first nine steps;
warmup pays no stagnation penalty. Reset creates fresh history, including scenario rebuilds.

The shaped stagnation cost is `clip(1 - progress / 0.05, 0, 1)` once ready: stationary or
retreating robots pay full cost, 2.5 cm of progress pays half, and at least 5 cm pays zero.
A brief pause can retain credit for recent progress; oscillation with no net progress cannot.
Standalone reward callers without window metadata use their consecutive-distance progress;
they have no episode history. The production environment always supplies the window.

This changes future training reward behavior only, not benchmark evaluation of existing
checkpoints. Observation, action, termination and canonical benchmark metric semantics are
unchanged. No training or benchmark campaign was run, and no frozen artifact is changed.
