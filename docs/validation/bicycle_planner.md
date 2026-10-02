# Bicycle planner adaptation

AI-GENERATED / NEEDS-REVIEW. Diagnostic development guidance; refs #10093.

The bicycle adapter preserves bounded requested speed, clips yaw to
`abs(v) * tan(max_steer) / wheelbase`, and converts yaw using the speed achievable
this step after acceleration/braking and action-space limits. A request with
`abs(v) < .001` and `abs(omega) > .000001` uses .1 m/s forward creep, bounded by the
platform speed cap. A zero/zero stop remains stopped. Creep enables an arc; it
cannot execute a turn in place. Reverse uses signed achievable speed.

Choose `configs/robots/t60_bicycle_30deg_v1.yaml` or
`configs/robots/t60_bicycle_45deg_v1.yaml`, and copy its `robot_config` mapping
into a scenario before `build_robot_config_from_scenario`. The existing loader
accepts this mapping; there is no global default switch. Both variants use an
estimated .90 m wheelbase, 1.34 m/s forward cap, 1 m/s² acceleration/braking,
no reverse, and a **.64 m covering disc**. The capsule is a separate step. The
steering variants .52/.79 radians are estimates, approximately 30°/45°; their
minimum centre-path turning radii are 1.57/.89 m. Wheelbase and steering are not
manufacturer measurements. Reverse #10068 was unmerged at the pinned base.

[Segway's T60 page](https://b2b.segway.com/kickscooter-t60/) identifies the
teleoperated, three-wheel platform but supplies no wheelbase, steering, or
autonomous speed specification. The task's 1.34 m/s cap is an autonomous proxy;
it is not the manual's riding-speed cap. The live #10093 discussion retrieved
on 2026-10-02 had two comments and no separate literature-notes comment.

Run the diagnostic helper from the repository root, with a project Python and
`PYTHONPATH=.:fast-pysf`, `BIKEFIX_OUTPUT` pointing to the evidence directory:
`python scripts/validation/run_bicycle_probe.py --prepare`, then
`python scripts/validation/run_bicycle_probe.py --worker 0 --workers 1`.
It consumes only the roster and six named scenarios from the release files,
replacing episode seeds explicitly with 1001–1005 (empty bearings) or 1001–1010
(scenarios). It never executes the release seed policy. DD/T60 use the same
.64 m disc, 1.34 m/s plant cap and acceleration/braking; diagnostic policy copies
retain smaller planner speed caps and bind explicit robot-radius fields to .64.
Learned policies are unchanged and may be outside their training distribution.

Empty goals lie 6 m away at 0/45/90/135/180°, initial heading zero, in a 400 m
empty map. Scenario completion and authored time caps remain the runner's
contracts under requested horizon600 and dt=.1. Stuck steps count requested
`abs(omega)>1e-5` with measured `abs(delta_heading/dt)<1e-5`; this includes braking
or transient limitations and is not by itself a failure classification.
Records and command/action/motion traces are preserved for each cell. Degraded
or fallback model execution must not be admitted as performance evidence.

## Test value

All witnesses use production objects or tracked config bytes; no production
test-only seam is needed. Each row answers behavior, credible regression and
why nearest existing coverage missed it. Expected numbers come from the bicycle
yaw law and hand-calculated step acceleration, never the function under test.

| Test in `test_bicycle_planner_physics.py` | Behavior protected / regression | Existing coverage gap |
|---|---|---|
| counter_rejects_low_speed_excess_yaw | Reject .5 rad/s at .1 m/s; independent box bounds return | Existing model test accepted a box; no physical yaw assertion |
| steering_uses_achievable_speed | Execute reachable .08 rad/s after acceleration to .1 m/s; target speed used in atan | Existing conversion test inspected action only |
| zero_speed_turn_creeps | Advance and turn for turn-only request; zero-speed steering discarded | Existing conversion used nonzero target speed |
| stop_stays_stopped | Preserve zero/zero stop; indiscriminate creep returns | No previous standstill control; this negative control passes on base |
| map_runner_world_velocity_not_silently_zeroed | Real XY command drives bicycle; diff-only cap lookup returns | Existing runner conversion coverage used differential settings |
| braking_uses_achievable_speed_and_deceleration | Execute v=.7, yaw=.2 after braking; symmetric clip or target denominator returns | Existing conversion never applied slowing turn/asymmetric brake |
| reverse_turn_yaw_has_requested_sign | Execute v=-.1, yaw=+.05; unsigned/wrong speed denominator returns | Existing reverse test checked bounds only |
| speed_priority_coupled_projection | Preserve bounded speed and constrain signed yaw including creep; box projection returns | Existing two model tests lacked curvature/boundary cases |
| creep_respects_low_speed_cap | Creep at .04 cap; unconditional .1 default returns | No previous cap below creep speed |
| runner_model_uses_actual_bicycle_caps | Use plant caps ahead of planner defaults; runtime limit wiring disappears | Existing standalone adapter test never resolved planner policy caps |
| episode_policy_receives_t60_limits | Bind real T60 plant into policy config; limits injected from planner defaults | Previous context tests used differential drive only |
| opt_in_t60_config_reaches_real_robot | Both tracked variants reach actual settings; loader/config integration drifts | No existing T60 configs or physical config-load assertion |

Three old assertions in `test_classic_planner_adapter.py` change deliberately:
`test_planner_action_adapter_bicycle_conversion` expects saturated .5 steering
at achievable .1 m/s; `test_bicycle_kinematics_model_projection_and_feasibility`
expects (0,0) for forbidden reverse; `test_bicycle_kinematics_model_allows_backwards_when_enabled`
rejects (-.5,.2) and accepts/projects (-.5,.1) at curvature .2.

Pre-fix: the seven physical checks fail six and pass the zero/zero control;
the added unit boundaries fail six and pass two unchanged-speed/stop controls.
The real runner-limit witness separately fails on base: expected (1.34,.4),
obtained (2,1). Config-load checks prove new integration, not an existing bug.
Post-fix targeted physical/config, release-identity and ecosystem checks passed
162 tests. Full-suite and campaign receipts are separate evidence gates.
