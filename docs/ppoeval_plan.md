# PPOEVAL diagnostic plan
Build V1 on pinned main and V2 on PR #9995 c9102a9c; V3 derives from V2.
Use identical diagnostic subset, release checkpoints, H600/dt0.1 and template budgets.
V3 alone enables the historical instantaneous +/-3 m/s plant, reverse, +/-1 rad/s.
Passive traces capture requested PPO target before wrapper clipping and selected command after guard.
Run one native episode of each PPO arm per variant, seed1001, medium head-on; no population claims.
Validate plant transitions, compare script against hand-calculated metrics, lint and checksum artifacts.
Deliver three diagnostic branches, Slurm scripts, exact commands and local report. No submission,
release, main push, merge, holdout seed or subagent. Outputs remain outside checkout and include
source/config/environment/checkpoint identity plus file checksums. New output ID after any failure.

Guard best-effort outcomes are measured diagnostics, never benchmark-success evidence.
Keep canonical campaign exit/status; return acquisition completion only after all requested
rows and complete native proposal traces exist. Failures remain preserved under fresh run IDs.
