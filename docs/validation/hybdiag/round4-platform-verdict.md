<!-- AI-GENERATED/NEEDS-REVIEW -->
The robot crawls on the station platform. The blocker is the pessimistic braking
model (stop before the pedestrian's present position) and the current 1.0 m robot
radius. There is no positive evidence of physical infeasibility.

The platform candidate-injection flag and its production code were removed in
Round 4. With the braking cap restored, Round 3 measured 236 successes versus
235/300 off; both equalled static at 270/300. Platform-only minimum executed
pedestrian separation fell from 1.593 to 1.530 m, near events rose 29%, and near
miss time was approximately 2.2 times off. That trade-off does not justify this
experiment's adoption.

The committed `round2-braking-bound-audit.csv` and JSON show all 45 former station
successes needed requested commands above the current-position braking cap
(24 platform, 21 both). These are requested-command violations, not actual-speed
or contact claims. Historical Round 2/3 results remain reproducible from their
compact episode summaries; the removed flag is not supported in the current
driver. Historical counterproofs load the test source from reviewed head f16a53f5.

Follow-up [#10111](https://github.com/ll7/robot_sf_ll7/issues/10111), milestone
0.1.0, “Station platform: braking bound that credits pedestrian reactivity”,
links #10092 and this evidence. Investigate a bound using predicted positions
with a demonstrated separation guarantee, then re-evaluate after the 0.1.0
radius change. Do not loosen the braking cap without that evidence.

Near misses are a separate safety measure from collision counts. The pooled
static/off event rate also changes with exposure: doorway completion adds
encounters that the off robot never traverses, while station events decrease.
See the per-scenario events, robot-seconds and rates next to the pooled tables in
`round2-vs-round3.md` and in the generated `round*-near-misses.csv` files. This
exposure decomposition does not erase the platform-only safety cost above.
