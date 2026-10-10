# SCENFIX2 development diagnostics

PR #10026's full-rectangle crowd regression is repaired without editing maps or robot zones.
The reaction buffer is one second at the population walking-speed cap plus 0.1 m,
which is 0.75 m on the two contact-probe maps. Zoned crowds reserve both robot start
and goal zones, as synthesized crowds already do; reset checks use the actual robot
pose and route respawns use its live pose.

`summary.json` records a native ORCA run on 48 scenarios × dev seeds 1001-1030,
H600/dt 0.1: **1227/1440 successes**, including **17/30 narrow_hallway**, **13/30
robot_crowding**, and **30/30 circular_crossing**. All 1440 rows are unique, valid,
and use ORCAPlannerAdapter with status ok; the runner reported no failures.
The review's historical comparator was 1242/1440 on base and 1218/1440 before
this repair. These are development diagnostics, not held-out evaluation or release
admission. The main merge and source-record caveat are explicit in `manifest.json`.

The three stationary JSON files retain all 60 records per tree: original PR head,
repaired PR, and local combination with #10024 at 77cf81eb. Both repaired trees
have zero contacts in ten zero-action steps and zero resets inside the buffer on
both maps. The combined tree is never pushed.

All 63 new regression cases pass. They protect zoned robot reservations,
actual-start reaction gaps, and fallback rectangle corner order; 20 cases fail on
current origin/main, and the original pre-repair head also fails. Existing importer
validation has 3341 passes and one pre-existing frozen-provenance failure. Combined
validation has 141 passes and three coordination failures: the two #9727 dev
fixtures and shared-world ordering. Whichever PR lands second must re-pin seeds
1003/1011 and correct the shared-world RNG comment/test.

Lane status is **blocked**: importer validation briefly exceeded the two-simulation-
process limit when a test spawned two workers beside the ORCA process. No held-out
episode was intentionally selected; explicit unsafe test exclusions are retained in
the private lane report. No skip/xfail or frozen-contract edits were added. All 60 committed-code replays of the two named scenarios exactly match the full
run in outcome, step count, metrics and spawn validity (`repeat-summary.json`). Raw
ORCA rows, runner, logs, exclusion lists, hashes, and the report are retained outside
the worktree in the lane evidence directory. #10037 tracks the remaining 0.0.9 goals.
