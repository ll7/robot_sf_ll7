# Stage 1: review of the existing bicycle diagnostics

Reviewed on 2026-10-28 against the 0.0.8 release commit `373dbfde`. The sources are
`docs/validation/bicycle_planner.md`, `bicycle_probe_provenance.json`,
`bicycle_wrapper_probe_provenance.json`, `scripts/validation/run_bicycle_probe.py` at each
producer commit, PR #10100 and issue #10093.

| Question | 6,650-cell matrix (job16478) | 180-episode wrapper arm (job17387) | Precursor KINPROBE (job16192) |
|---|---|---|---|
| Source commit | `b81d5823` (historical round 1 of PR #10100) | `990dc6f0` (round 2) | `f746827c` (before #10100) |
| Same production bytes as 0.0.8? | No. Since then, `kinematics_model.py`, `bicycle_drive.py`, `classic_planner_adapter.py`, `map_runner_episode.py`, `differential_drive.py`, `scenario_loader.py` and both T60 YAMLs have changed | No. `map_runner_episode.py`, `kinematics_model.py` and `bicycle_drive.py` have changed since | No. It contains the bicycle adapter defect that #10100 fixed |
| Configuration | 14 release planners; six KINPROBE scenarios (cross-trap high, station platform, T-intersection, frontal approach, narrow hallway, robot crowding) on seeds 1001–1010; empty-world bearings on seeds 1001–1005; arms DD, T60-30, T60-45, and creep-on variants | `goal` only, with the safety wrapper and deadlock recovery on (not the release configuration); same six scenarios, seeds 1001–1010 | 14 planners; same axes |
| Plant matched to the thesis DD? | **No.** The DD arm is a proxy: 0.64 m radius, 1.34 m/s, 1 m/s², no reverse. It matches the T60 arms but not the release robot (1.0 m, 2.0 m/s). Policy configs were rewritten to radius 0.64 and a ≤1.34 m/s cap | **No.** Same 0.64 m / 1.34 m/s proxy | Radius 1.0 m, 2.0 m/s, 1 m/s², no reverse (matched), but wheelbase 0.85 m and steering 0.6 rad. It ran through the defective adapter, which turned world-velocity commands into zero commands |
| Raw records | Only hashes are in the repository (fixed corpus 19,951 files, original 13,861, replay 382). The document says they "remain in the task lane". Not found on the reviewing host and not in the W&B project | Only hashes (541 files). The task-lane path is not found and is not in W&B | **Found** on the reviewing host (2,730 results, traces and records). Now archived with a sha256 manifest (host archive only) |
| Reproducible now? | The creep-on arms explicitly are not (the current `map_runner` never applies ordinary creep) | Only from the historical commit | Only from the historical commit plus an uncommitted patch |
| Statistics | Wilson intervals pooled over 14 planners; no paired seed-block intervals; 10 seeds | Wilson intervals; 10 seeds | Counts only |
| Review status | Marked `AI-GENERATED / NEEDS-REVIEW`. PR #10100 was merged on 2026-10-03 with the `review-bot-auto` label; only bot reviews are recorded; issue #10093 is still open | Same | Uncommitted lane diagnostic; never reviewed |

## Verdict

**The existing records do not support a bounded statement about the thesis embodiment.**

- None of the three corpora ran on the 0.0.8 production bytes.
- The two corpora documented in the repository compare a 0.64 m / 1.34 m/s proxy differential drive with a T60 proxy, not the release robot.
- Their raw records cannot be located.
- Their review status is "needs review".

The precursor corpus has a matched radius and speed cap, but it measures a since-fixed adapter defect, not the bicycle embodiment.

The historical material still has diagnostic value. It already showed the zero-speed-yaw deadlock mechanism (the stuck windows under creep-off), and stage 2 reproduces that mechanism on the release code. Stage 2 (the compact design) was therefore run.
