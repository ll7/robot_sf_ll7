# Guarded-PPO obstacle contacts in the frozen 0.0.6 campaign

Claim boundary: descriptive analysis of the frozen release bundle only. No episode was
stepped or rerun, no checkpoint retrained, no runtime changed.
Evidence tier: diagnostic reading of retained episode records.

## Input identity

- Bundle: `benchmark_0_0_6_s30_h600_20260911_publication_bundle` (checksums verified)
- Source commit: `31cdfe0361abe2c520117a17f99c1b7a0aba4359`
- Arms: `guarded_ppo__differential_drive` (BR-06 v3 checkpoint behind the runtime guard) and
  `ppo__differential_drive` (different checkpoint, no guard) — descriptive comparison only,
  not a clean guard ablation because the checkpoints differ.

## Availability matrix (what the bundle can and cannot answer)

| Issue packet item | Verdict | Reason |
| --- | --- | --- |
| Contact table: cell, step, map location | partial | cell + step (via `collision_time`) retained; map location NA — no per-step positions retained |
| Pedestrian within clearance in preceding N steps | NA | no step traces |
| Guard counts per episode + contact-step decision | available | `guard_stats`, `shield_stats.last_decision` |
| Guard active in last-k-steps window | NA | per-step decision series not retained |
| Base-PPO same views | available | episode-level |
| Contact timing vs cap | available | `collision_time` vs 600-step (60 s) cap |
| Per-map overlay figure | substituted | family-rate bars + timing histogram instead (no positions) |

## Table 7.1 verification

| arm | episodes | ped mean | obstacle mean | success | timeouts |
| --- | --- | --- | --- | --- | --- |
| guarded_ppo | 1440 | 0.0174 | 0.334 | 329/1440 | 605 |
| base ppo | 1440 | 0.0986 | 0.3243 | 796/1440 | 35 |

Guarded obstacle mean (0.334) matches Table 7.1 (0.33); base PPO is nearly identical
(0.3243). Guarded sheds pedestrian contacts (0.0174 vs 0.0986) while success collapses (329 vs 796) and timeouts rise (605 vs 35). The obstacle rate is not a trade the guard
introduced — descriptively, both checkpoints hit walls at the same rate.

## Obstacle-contact rate by scenario family

| family | guarded contacts / episodes (rate) | base contacts / episodes (rate) |
| --- | --- | --- |
| classic_cross_trap | 69/90 (0.767) | 14/90 (0.156) |
| classic_merging | 58/60 (0.967) | 59/60 (0.983) |
| classic_doorway | 52/90 (0.578) | 45/90 (0.500) |
| classic_overtaking | 51/60 (0.850) | 13/60 (0.217) |
| classic_t_intersection | 50/60 (0.833) | 44/60 (0.733) |
| classic_bottleneck | 42/90 (0.467) | 70/90 (0.778) |
| francis2023_narrow_doorway | 30/30 (1.000) | 30/30 (1.000) |
| francis2023_following_human | 28/30 (0.933) | 1/30 (0.033) |
| francis2023_blind_corner | 20/30 (0.667) | 17/30 (0.567) |
| francis2023_entering_elevator | 18/30 (0.600) | 11/30 (0.367) |
| classic_group_crossing | 17/90 (0.189) | 10/90 (0.111) |
| francis2023_narrow_hallway | 14/30 (0.467) | 26/30 (0.867) |
| francis2023_entering_room | 12/30 (0.400) | 4/30 (0.133) |
| classic_station_platform | 9/30 (0.300) | 14/30 (0.467) |
| classic_head_on_corridor | 6/60 (0.100) | 8/60 (0.133) |
| classic_urban_crossing | 2/30 (0.067) | 7/30 (0.233) |
| francis2023_crowd_navigation | 1/30 (0.033) | 1/30 (0.033) |
| francis2023_exiting_elevator | 1/30 (0.033) | 17/30 (0.567) |
| francis2023_exiting_room | 1/30 (0.033) | 3/30 (0.100) |
| classic_realworld_double_bottleneck | 0/30 (0.000) | 28/30 (0.933) |
| francis2023_accompanying_peer | 0/30 (0.000) | 0/30 (0.000) |
| francis2023_circular_crossing | 0/30 (0.000) | 0/30 (0.000) |
| francis2023_down_path | 0/30 (0.000) | 0/30 (0.000) |
| francis2023_frontal_approach | 0/30 (0.000) | 0/30 (0.000) |
| francis2023_intersection_no_gesture | 0/30 (0.000) | 5/30 (0.167) |
| francis2023_intersection_proceed | 0/30 (0.000) | 6/30 (0.200) |
| francis2023_intersection_wait | 0/30 (0.000) | 6/30 (0.200) |
| francis2023_join_group | 0/30 (0.000) | 6/30 (0.200) |
| francis2023_leading_human | 0/30 (0.000) | 0/30 (0.000) |
| francis2023_leave_group | 0/30 (0.000) | 1/30 (0.033) |
| francis2023_parallel_traffic | 0/30 (0.000) | 9/30 (0.300) |
| francis2023_pedestrian_obstruction | 0/30 (0.000) | 0/30 (0.000) |
| francis2023_pedestrian_overtaking | 0/30 (0.000) | 0/30 (0.000) |
| francis2023_perpendicular_traffic | 0/30 (0.000) | 10/30 (0.333) |
| francis2023_robot_crowding | 0/30 (0.000) | 0/30 (0.000) |
| francis2023_robot_overtaking | 0/30 (0.000) | 2/30 (0.067) |

![family rates](../figures/issue_9480_guarded_obstacle_trade/family_rates.png)

Shared wall-heavy cells (merging ~0.97, narrow_doorway 1.00 both arms, doorway,
t_intersection, bottleneck) hit both checkpoints. Divergences (cross_trap 0.767 vs 0.156,
overtaking 0.850 vs 0.217, narrow_hallway 0.467 vs 0.867) are descriptive only: the
checkpoints differ, so no family delta isolates the guard.

## Contact timing vs the episode cap

Guarded contact times (s): n=481, median=13.0, p90=38.8, max=57.6.
Base contact times (s): n=467, median=9.6, p90=17.6, max=57.5.
Contacts after 50 s of the 60 s cap: guarded 18/481, base 5/467.

![contact timing](../figures/issue_9480_guarded_obstacle_trade/contact_timing_hist.png)

Contacts skew early/mid-episode for both arms; guarded contacts run later (median 13.0 s
vs 9.6 s), consistent with the guard prolonging episodes rather than with a
"late, under time pressure" cluster.

## Guard cross-tabulation (guarded arm, 481 contact episodes)

Per-episode guard decisions over 82473 contact-episode steps: ppo_clear=21607, ppo_safe=982, prior=0, fallback=29496, stop=30388, uncertainty=0.

Final (contact-step) decisions: fallback_safe=186, ppo_clear=159, ppo_safe=3, stop_safe=133.

![contact-step decision](../figures/issue_9480_guarded_obstacle_trade/contact_step_decision.png)

In 322/481 contact episodes the guard's final decision was a substitution
(`fallback_safe` 186, `stop_safe` 133, `ppo_safe` 3), i.e. the guard had engaged by
contact time in two thirds of cases yet contact still occurred. In 159/481 the guard
passed the PPO command through (`ppo_clear`). Correlation, not causation: without step
traces we cannot tell unavoidable-from-early-commitment apart from
fallback-steered-into-wall. The coverage reading is that the guard's obstacle clearance
(0.30 m) and short-horizon rollout do not prevent these contacts, and the fallback DWA
weights goal progress (4.5) far above obstacle clearance (1.2).

## Pedestrian proximity in contact vs clean episodes (guarded arm)

Mean pedestrian near-miss events per step: contact episodes 0.03118, clean episodes 0.06829. Episodes with any near-miss: 67/481 contact vs 361/934 clean.

Wall contacts concentrate in episodes with *less* pedestrian proximity — evidence against
'dodge-pedestrian-into-wall' as the dominant mechanism and consistent with
constrained-geometry contacts under weak obstacle coverage.

## Verified implementation facts (code, not prose)

- Training reward `route_completion_v3` (un-overridden): collision -10.0 covers
  pedestrian/robot/obstacle alike; `near_miss` -1.0 is pedestrian-only
  (`snqi_proxy`: robot-ped min distance); `ttc_risk` -0.8 falls back to `near_misses`
  because PPO env metadata never sets `time_to_collision` — hence effectively
  pedestrian-only in training. (Issue prose cites -1.5/-1.2; the frozen code and the
  March-2026 training-time weights are -1.0/-0.8.)
- Guard thresholds are NOT pedestrian-only: `guard_hard_ped_clearance` 0.58 m,
  `guard_hard_obstacle_clearance` 0.30 m, `guard_min_ttc` 0.70 s; fallback DWA weights
  pedestrian clearance 2.0 vs obstacle clearance 1.2 with goal progress 4.5.

## Dissertation paragraph (Section 7.4 candidate)

In the frozen 0.0.6 campaign guarded PPO nearly eliminates pedestrian contact (0.017 per
episode) while obstacle contact (0.334) matches the unguarded checkpoint (0.324), so the
configuration trades success for pedestrian safety rather than pedestrians for walls:
success falls 796 to 329 of 1440 with timeouts rising 35 to 605. Contacts concentrate in
constrained cells (merging, narrow doorway, doorway, t-intersection) at early-to-mid
episode times, in episodes with below-average pedestrian proximity, and in two thirds of
cases after the guard had already substituted a fallback or stop command — consistent
with pedestrian-asymmetric shaping (pedestrian-only near-miss/TTC terms) plus obstacle
coverage (0.30 m clearance, short-horizon rollout) too weak to save wall approaches the
policy commits to. Step-trace evidence for the final causal step is not retained in the
bundle.

Report schema: `issue_9480_guarded_obstacle_trade_report.v1`.
