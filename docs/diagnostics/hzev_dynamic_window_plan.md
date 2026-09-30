# HZEV dynamic-window diagnostic

Diagnostic only, derived from the real 0.0.8 template at main cf67fa7.
Hypothesis: authored short horizons prevent waiting until pedestrians leave.
No release evidence admission, main mutation, Slurm submission, or held-out seeds.
The user-selected fresh clone owns branch `diag/scenario-dynamic-window`.

1. Freeze the canonical loader's 48 scenarios and authored budgets; bind a sidecar
   with 800 steps each and dev seeds 1001–1005.
2. Reuse map-runner episodes and native goal/ORCA builders in a diagnostic entry
   point. A zero-command stationary arm bypasses policy inference only. Record
   reset geometry, route respawns, all steps, and first normal terminal event.
   Continue without reset after contact; park movers after first goal completion.
3. Measure corridor clearance, last mover interaction, and optimistic wait-then-go
   arithmetic; distinguish recurring flow and right censoring from clearance.
4. Validate synthetic traces and one local short stationary episode, serially.
5. Push only the requested diagnostic branch, verify remote SHA, preserve the
   local evidence outside output/, and hand off a source-pinned sbatch command.

Owned paths: configs/{benchmarks,scenarios}/hzev*, scripts/validation/*hzev*,
tests/benchmark/test_hzev_dynamic_window.py, docs/diagnostics/hzev*.
Acceptance: exact 48×5×3 roster, 800 trace steps per cluster row, no silent
fallback, dev seed guards, reproducible analysis, executable CPU-only Slurm packet.
Local smoke is infrastructure proof only; cluster acquisition and interpretation
remain the orchestrator's responsibility. Partial traces cannot settle a scenario.
Artifacts are atomically written with checksums; incomplete manifests or missing
rows block full analysis. Resume requires the same source/config identity.

## Repair after Slurm job 15805

Freeze the ordinary loop result at its first terminal event, before computing
production metrics. Store all 800 frames separately as diagnostic trace data;
never weaken the production integrity validator. Preserve all analysis definitions.
Owned repair paths: the two diagnostic Python scripts, sbatch, this plan, the
diagnostic guide, and tests/benchmark/test_hzev_dynamic_window.py.

Prove collision-then-goal and goal-then-collision isolation with controlled
800-step events, plus real full-length classic_bottleneck_low/1002/all-arm
acquisition and real mover witnesses. Stationary zero motion may make a real
goal witness impossible; report that boundary rather than changing its policy.
Compare two short seeded episodes at one versus three workers. Run six real
scenarios × one dev seed × three arms at H800, measuring episode and batch wall
time. Local acquisition uses at most three workers and seeds 1001–1005 only.
Allow 1..allocated CPUs in the CLI (default three); sbatch reserves one CPU for
the parent. Set its time from the sample with 50% margin for 720 episodes.
Keep raw rows, manifests, logs and measurement summaries outside the checkout.
Commit explicit paths and push only diag/scenario-dynamic-window without force;
verify remote SHA and write ~/hzevfix_report.md. No Slurm submission in this lane.
