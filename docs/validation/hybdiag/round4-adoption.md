<!-- AI-GENERATED/NEEDS-REVIEW -->
Recommend the physical wall fix on its demonstrated merits: wall exclusion alone
recovers 35 crowded successes (235 to 270/300), with zero contacts and a pooled
pedestrian minimum of 1.593 m at the displayed precision. Raw CSV minima differ
from off by about 0.000012 m; no separation noninferiority is claimed. However, do not deploy the wall-only configuration: it adds
one empty-world failure, narrow hallway seed 1002, and fails the adoption gate.

Recommend exactly these three opt-in settings together:

- Planner `physical_static_exclusion_enabled: true`.
- Planner `goal_next_validity_enabled: true`.
- Environment `include_goal_next_valid: true`.

All three defaults remain false. Wall exclusion exposes the latent goal defect:
off succeeds on hallway seed 1002, while wall-only fails. No arm isolates the
goal-validity pair alone. This combined arm is accepted:270/300 crowded
successes,100/102 empty successes, zero contacts, zero new empty failures and
collision intervals no higher than off. Sealed 0.0.8 confirmation is still pending.

The wall-only arm's sole new failure versus off is **livelock** in
`francis2023_narrow_hallway`, seed 1002: 60 s timeout, final goal gap 0.704654 m,
57 feasible moving candidates at the last step, zero no-moving time, and 0.5%
stopped time. The robot continues the known near-goal orbit because the absent
successor retains its legacy world-origin target. The independently flagged
validity pair succeeds on the same seed in 13.3 s. This is a planner goal-selection
defect, not demonstrated physical infeasibility; the existing goal fix handles
it, with no recovery behavior or default migration.

Withdrawing goal validity also loses five other empty successes relative to
Round 3's combined static/goal arm, although those five already fail with off.
All six are classified individually as livelock in
`round 4-failure-classifications.json`; every case has feasible moving candidates,
zero no-moving time and paired goal-validity success. Some do not trigger the
10 s/0.5 m freezing proxy because their cycles have larger displacement.

Crowded doorway seed 1026 also shows the independent goal benefit: wall-only
success in 49.7 s with freezing; the goal pair succeeds in 16.3 s without freezing.
The crowded success count is unchanged, while total robot exposure falls
8210.2 to 8112.9 s. Wall-only empty results are 94/102, versus 100/102 with goal
validity; the two remaining narrow-doorway timeouts are shared with off and
Round 3, not newly introduced failures.

The pooled pedestrian near rate rises versus off because route completion
changes exposure: doorway adds 30 events, station loses 15. Both retained wall
arms have 276 events versus 255 off, minimum separation 1.593 m, and zero contacts.
This is not a claim of statistical noninferiority or a separation guarantee.
Per-scenario events, robot-seconds and rates are published beside pooled rates.
