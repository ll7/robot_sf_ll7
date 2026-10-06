# Room and elevator route-width candidate (#9856)

## Observed defect and candidate boundary

The ordered-waypoint preflight in #9819 found all 120 cells in the four room/elevator families blocked on the historical matrix. Of these, 114 were newly exposed after the checker began visiting every required waypoint rather than connecting the reset directly to the final goal. The raw path touched openings about 1.9 m wide at required waypoints, below the 2.2 m width required for a radius-1.0 m robot plus 0.10 m margin on each side. This is setup evidence, not a planner outcome.

The `classic_interactions_francis2023_goal_zone_entry_route_width_v4.yaml` matrix includes the reviewed v3 goal-zone successor and changes only the four map-file identities below. It preserves all 48 scenario names, order, seeds, planner-independent row conditions, and the historical 0.0.7 inputs. Its SHA-256 is `8a8aa9adb2b098f039a3e92a4849e16eb8d3665e928211f08e0c689584c9f389`.

| Scenario suffix | Source SHA-256 | Successor SHA-256 | Versioned correction |
| --- | --- | --- | --- |
| `entering_room` | `3d18c06ee133949b14e4274c548b19b61f10ed518c105b5a94f660b53376c945` | `1506b9ddbee141c7256c92802cb6513d3add4cf5a270a9bad3dd197ec7a69166` | Route through `(14.5, 9)` in the unchanged room opening. |
| `exiting_room` | `3e84da29278394078098cf393e7e6fe54914b7365aa99621468b3ade4ffd2fbf` | `cfd25fa09fd2a315f84f39408e57c0ec59bff14ea86d6020a07b93d75a3e434c` | Route through `(14.5, 9)` in the unchanged room opening; source is #9815's goal-zone successor. |
| `entering_elevator` | `fa52e7b55bae9ae8a1fdfc0c13cd3e18e43d2c005cb325fe9a978fa1c39f6c10` | `1b486a22cbd1ee1677732bd9d3086892e83d7c6e86f6ad380db921c621597563` | Route through `(12.5, 9.5)`; move the elevator shell 0.4 m away from the unchanged goal zone to clear randomly sampled final targets. |
| `exiting_elevator` | `10ad90b9ee7dac2d84ec0ec8017575b9ab5b2549d249e215ec88b64d88db9019` | `4d176eca64ea8528f5eabd9df4e0314a4a3b0b7ba615736933cd3a5697d3eb1d` | Route through `(12.5, 9.5)` and use the same shell geometry; source is #9815's goal-zone successor. |

The elevator shell retains its five labelled wall rectangles and outer frame. The top/bottom wall bands move from `y=7..8`/`12..13` to `6.6..7.6`/`12.4..13.4`; the back wall moves from `x=17..18` to `17.4..18.4`; the entrance pieces follow the wall bands. The unchanged 2 × 2 m entering-elevator goal zone is then at least 1.4 m from obstacles throughout, including the sampled endpoint. These edits may change pedestrian and robot trajectories and require the #9668 paired-release comparison before result admission.

## Diagnostic evidence and next gates

The parsed successor routes have minimum straight-segment obstacle clearances of 1.900 m (entering room), 1.786 m (exiting room), 1.900 m (entering elevator), and 1.900 m (exiting elevator), all above the 1.1 m setup margin. The scenario loader retains 48 ordered identities and changes only the four map-file references.

An initial direct run of #9819's exact `eff993794baa93adffabc9f4d17425c7a375cf11` preflight cell checker against this v4 matrix returned **120/120 valid** affected cells for seeds 111–140. The prototype JSON is held in common Git-dir custody at `.git/codex-agent-runs/issue-9856-route-width-20260928/affected120-eff993-prototype.json`, SHA-256 `62c0b972f7cc42b0d0772ebd57079b1ca60a027efc8310dfced99458b027d13b`. That direct check is diagnostic because it did not resolve a physical checksummed release manifest. #9819 has since merged; rerun its manifest-bound preflight on a checksummed corrected candidate after #9863 supplies the DOI-free manifest path, inspect all four families and the remaining 48-matrix blocks, and preserve both reports and digests. No nominal campaign starts on this prototype result.

An earlier direct checker produced complete 1,440-row diagnostics for the historical matrix, inherited v3 matrix, and this v4 matrix. All three reports pair on identical scenario/seed identities. Historical setup blocked 281 cells. Inherited v3 blocked 279, clearing only two `classic_station_platform_medium` respawn-window failures. This v4 blocked 159, clearing exactly the 120 room/elevator cells relative to v3, with **zero newly blocked identities**. The v3 report SHA-256 is `30638e0a00d4da4ee1c957645a5bd214426819daeb73be605f9f51e64dd2c697`; the earlier v4 report SHA-256 is `f1f52e1224f19ae9c27834fb952f990138be0f3f41590ba8550d036d13b8ab5b`. The historical reference is #9819's `spawn_matrix_preflight_headeff.v1.json`, SHA-256 `b78d5422294a4f5c2fc1bafd98ef6205fdd9faee53ef9d273f364b15d701857c`.

The 159 remaining v4 blocks are **107 reachability-plus-width failures** and **52 reachability-only failures**; no reset-clearance or respawn-window check failed in this direct run. The 30 historical 2 m narrow-doorway cells remain an explicitly labelled infeasibility probe under their separate manifest, not nominal release inputs.

A fresh rerun used the exact reviewed #9819 implementation at `eff993794baa93adffabc9f4d17425c7a375cf11` against this candidate. It loaded the existing release manifest, substituted the candidate matrix path and digest in memory, and evaluated all 48 identities × 30 seeds in one process. The report emitted 1,440 rows with no input or worker error, found 159 remaining blocks, and found **0/120 invalid** rows for the four corrected room/elevator families. The JSON is held at `.git/codex-agent-runs/issue-9856-route-width-20260928/issue9856-9819-eff993-v4-preflight-workers1.json` with SHA-256 `2b7e0a3b12ad639f217723a3ccf16bde083fd847507bfe02cec730020a26616d`; its Markdown companion is `.git/codex-agent-runs/issue-9856-route-width-20260928/issue9856-9819-eff993-v4-preflight-workers1.md` with SHA-256 `96cd0cb6d4f0d9f7ad83c2bdc2a124744b3469a4821413f5c75b08cb36b4a32c`. An initial eight-worker attempt failed before geometry because the in-memory module was not pickleable; that failure is retained at `.git/codex-agent-runs/issue-9856-route-width-20260928/issue9856-9819-eff993-v4-preflight-workers8-pickling-failure.json` (SHA-256 `f1a4f34ad906517afd52a8b4df49441c06a1ea2ea1bd3bdf106b6a531e8ee22a`) with its Markdown companion (SHA-256 `f20b206ffabe59aa98cdc0cef2a1c4bd4c729670c383f8051738ae09e0fd81eb`). This archived premerge run remains diagnostic: the candidate matrix was not written into a physical checksummed release manifest, so it does not establish canonical release admission.

An independent continuous-geometry diagnostic classified the 159 blocked cells by resetting each exact scenario/seed, buffering parsed wall obstacles by the 1.0 m robot radius plus 0.10 m margin, and checking exact clearance at the reset and every required navigator waypoint. When those points cleared, it checked whether they belonged to one connected free-space polygon inside the map bounds. The resulting JSON is held at `.git/codex-agent-runs/issue-9856-route-width-20260928/v4_block_classification.json`, SHA-256 `309753092ba20d36c65d446affb2ffe6a2650a4b6f54c2e8c326d9c601259d17`. This geometric oracle does not establish planner success, pedestrian safety, or release admission.

| Geometric diagnostic category | All blocked cells | Nominal cells | Disposition |
| --- | ---: | ---: | --- |
| Sampled final goal below exact 0.10 m clearance margin | 73 | 71 | Version or correct source geometry/goal sampling under #9859. |
| Exact required points clear and in one continuous free component, but raster grid blocks | 58 | 58 | Correct conservative checker precision under #9860. This includes 55 grid-blocked goals and 3 grid-blocked starts. |
| Required points in disconnected continuous free components | 28 | 0 | Keep the historical narrow doorway under its separate infeasibility-probe manifest. |

The 71 genuine non-probe goal defects and 58 grid false blocks remain nominal release blockers. Both [#9859](https://github.com/ll7/robot_sf_ll7/issues/9859) and [#9860](https://github.com/ll7/robot_sf_ll7/issues/9860) are children of #9730 and block #9668. The other two goal-clearance failures are in the historical doorway probe. A DOI-free diagnostic manifest can bind the preflight API to these candidate inputs, but the current canonical release-manifest loader requires publication coordinates; [#9863](https://github.com/ll7/robot_sf_ll7/issues/9863) tracks a proper prepublication mode. Neither this classification nor an API-bound diagnostic replaces the canonical release gate, and no nominal evaluation starts on these inputs.

| Remaining scenario | Blocked | Reachability + width | Reachability only |
| --- | ---: | ---: | ---: |
| `classic_t_intersection_low` | 11 | 11 | 0 |
| `classic_t_intersection_medium` | 11 | 11 | 0 |
| `classic_urban_crossing_medium` | 4 | 3 | 1 |
| `francis2023_accompanying_peer` | 7 | 3 | 4 |
| `francis2023_blind_corner` | 6 | 3 | 3 |
| `francis2023_circular_crossing` | 2 | 2 | 0 |
| `francis2023_crowd_navigation` | 1 | 1 | 0 |
| `francis2023_down_path` | 7 | 3 | 4 |
| `francis2023_following_human` | 7 | 3 | 4 |
| `francis2023_frontal_approach` | 6 | 2 | 4 |
| `francis2023_intersection_no_gesture` | 6 | 3 | 3 |
| `francis2023_intersection_proceed` | 6 | 3 | 3 |
| `francis2023_intersection_wait` | 6 | 3 | 3 |
| `francis2023_join_group` | 2 | 2 | 0 |
| `francis2023_leading_human` | 7 | 3 | 4 |
| `francis2023_leave_group` | 2 | 2 | 0 |
| `francis2023_narrow_doorway` | 30 | 30 | 0 |
| `francis2023_narrow_hallway` | 3 | 3 | 0 |
| `francis2023_parallel_traffic` | 6 | 2 | 4 |
| `francis2023_pedestrian_obstruction` | 6 | 2 | 4 |
| `francis2023_pedestrian_overtaking` | 6 | 2 | 4 |
| `francis2023_perpendicular_traffic` | 6 | 3 | 3 |
| `francis2023_robot_crowding` | 5 | 5 | 0 |
| `francis2023_robot_overtaking` | 6 | 2 | 4 |

The 0.0.7 matrix remains SHA-256 `d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5`; the preregistered historical 2 m doorway SVG remains `7538ed173d462a5107afc1a1e43b5b2e6d2bc5c9604035cdec9a551e20a8b15e`. Neither the historical release bundle nor its tag or Zenodo record changes here.
