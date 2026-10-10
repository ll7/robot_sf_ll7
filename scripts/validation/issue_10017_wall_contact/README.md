# Pedestrian Wall Contact Diagnostics

Development-only issue #10017 diagnostics, not release or calibration evidence.
The committed slots preserve the earlier 72-task harness: 40 lone apertures,
30 three-width doorway slots, one crowded bottleneck, and one doorway scenario.
Five laws make the requested 360-slot matrix; the two amplified controls add 144 slots.
All stochastic episode seeds are in 1001-1010. Every slot lasts 70 simulated seconds.

Set `WALL_CONTACT_ARTIFACT_ROOT` to an owned directory outside the source checkout.
The runner checks committed source bytes and HEAD, uses nice 15, numeric threads one,
and at most eight child processes. Run one batch at a time and check host load first.

```bash
export WALL_CONTACT_ARTIFACT_ROOT=/absolute/owned/artifact/directory
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1
nice -n 15 scripts/dev/run_worktree_shared_venv.sh --profile all-extras -- uv run python scripts/validation/issue_10017_wall_contact/prepare.py
nice -n 15 scripts/dev/run_worktree_shared_venv.sh --profile all-extras -- uv run python scripts/validation/issue_10017_wall_contact/run_variant_measurements.py legacy range_only physical_margin contact_stiff multi_segment wall10x physical_margin10x
TASK_INDICES=70 nice -n 15 scripts/dev/run_worktree_shared_venv.sh --profile all-extras -- uv run python scripts/validation/issue_10017_wall_contact/run_variant_measurements.py small_dt no_social physical_margin10x_small_dt
nice -n 15 scripts/dev/run_worktree_shared_venv.sh --profile all-extras -- uv run python scripts/validation/issue_10017_wall_contact/analyze_variants.py
nice -n 15 scripts/dev/run_worktree_shared_venv.sh --profile all-extras -- uv run python scripts/validation/issue_10017_wall_contact/analyze_traces.py
```

`wall10x` amplifies the range-only wall force, without changing any other force.
`physical_margin10x` combines amplification with the existing 0.05 m contact-radius
margin, matching the 0.40 m physical body when PySF's force radius is 0.35 m.
These are diagnostic controls, not new defaults. `small_dt` uses 0.01 s integration;
`no_social` zeros pedestrian social and group forces while keeping the same initial
positions, goals, robot, geometry, and wall force.

`trajectory.npz` retains the independent Shapely body-clearance metric, including
swept segments. The metric always uses `SimulationSettings.ped_radius` and the same
map polygons and backend segments; it never selects geometry based on the wall law.
`force-trace.npz` records the post-behavior state at force evaluation, all component
vectors, each segment's hypothetical force, closest points, and the exact integrator
velocities. Trace analysis distinguishes hypothetical from actually aggregated
segment forces and initial overlaps from newly overlapping pedestrians. Geometry
digests and radius equality are checked across every bottleneck control.

The overlap count is an any-time body-overlap count, not a count of pedestrian
centers passing through solid polygons. Initial overlaps count. The proximity-stall
counter has a final-wall-distance condition, so legacy stand-off can hide goal
incompletion from that counter; always compare assigned-goal outcomes as well.
