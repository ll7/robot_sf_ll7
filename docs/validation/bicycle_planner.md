# Bicycle planner adaptation

AI-GENERATED / NEEDS-REVIEW. Diagnostic development guidance; refs #10093 and draft PR #10100.

Bicycle commands preserve bounded requested speed and clip yaw to
`min(max_angular_speed, abs(v) * tan(max_steer) / wheelbase)`. Steering uses
speed achievable this step after acceleration and asymmetric braking. An
explicit physical curvature is required; independent speed/yaw caps cannot
supply wheelbase or steering geometry. Zero curvature is a valid straight plant.

## Creep and safety policy

**Creep is disabled by default**, including both opt-in T60 configs. Select
`robot_config.creep_speed: 0.1` to enable forward arcs for intentional yaw-only
requests. The gate is `0 <= v < .001 m/s` and `abs(omega) >= pi/180 rad/s`.
Negative requests never become forward creep; creep cannot reduce a positive
speed request. The adapter uses the model/config speed, without a second .1
constant. Hard-stop/yield, deadlock-monitor and CBF interventions carry a veto
through the runner to final conversion, suppressing creep even when opted in.
The safety wrapper's promise that recovery never adds forward speed remains true.
The dev-seed physical veto witness advances .389 m in 4 s on reviewed head
44c386d6 and 0 m after this fix; DD remains at 0 m. The monitor-only witness
changes .100 m to 0 m. These use the real filter/conversion stages and a
stationary-pedestrian fixture, not a full map collision rollout.
A stationary bicycle cannot perform the monitor's turn in place.

There is no extra duration gate: executed one-step tests at .05 and .1 m/s
show at most 1 cm displacement for an isolated .1 s request from rest, followed
by immediate braking on the next stop. Noise below one degree/second cannot
start creep. This bounds the isolated-pulse effect for the tested 1 m/s² plant;
it does not make a sustained intentional arc safe near obstacles. Safety vetoes
still take precedence on every step. Direct adapter callers must pass
`safety_intervention=True` when their upstream safety stage intervenes.

## Opt-in T60 proxy

Choose `configs/robots/t60_bicycle_30deg_v1.yaml` or
`configs/robots/t60_bicycle_45deg_v1.yaml` and copy its `robot_config` mapping
into a scenario before `build_robot_config_from_scenario`. Both use estimated
.90 m wheelbase, .52/.79 rad steering (approximately 30°/45°), 1.34 m/s forward
cap, 1 m/s² acceleration/braking, no reverse, and a **.64 m covering disc**.
The capsule and reverse integration are separate work. The corresponding
minimum centre-path turning radii are approximately 1.57/.89 m.
Wheelbase/steering/autonomous speed are task proxies, not manufacturer measurements.
[Segway's T60 page](https://b2b.segway.com/kickscooter-t60/) identifies the
three-wheel platform without those physical specifications. #10068 was open
at the pinned base. #10093 had two comments when initially retrieved; no
separate literature-notes comment was present, so explicit task parameters govern.

With no bicycle selection, differential and holonomic source paths and the
release identity inputs, golden oracles and ecosystem contracts keep their
bytes. The acceptance receipt compares 207 files to original base
`879f75b69eb93ca16006f2019ee5c85c7aa724dd`, repeats 240 differential/holonomic
command/action/motion samples against reviewed head `44c386d6`, and runs the
release identity/ecosystem/golden tests. No release episodes or sealed seeds run.

## Test value and red-to-green proof

The tests use production models, robot motion, the actual safety-filter/final
conversion stages, and tracked T60 YAML bytes. Hand-calculated yaw/step motion
provide independent expectations. No production test-only seam is introduced.
The table answers behavior, credible regression and nearest coverage gap;
that production integration also answers the fourth question about real bytes
and non-tautological assertions.

| Tests | Behavior and credible regression | Nearest prior coverage gap |
|---|---|---|
| default creep / noise gates | Stop remains stopped; tiny yaw cannot silently start .1 m/s motion | Old yaw-only physics test assumed implicit creep; no intent threshold |
| small reverse / tiny positive | Preserve signed bounded speed; near-zero reverse must not flip forward and disabled creep must not erase positive speed | Coupled projection boundaries lacked these near-zero signed requests |
| three safety-veto integrations | Forty actual hard-stop/deadlock-filter commands stay at zero speed/position even with creep opted in | Safety tests ended before bicycle conversion; adapter tests never composed safety stages |
| missing curvature / explicit zero | Raise for invented geometry, accept zero-steer plant | Old model constructors derived curvature from unrelated scalar caps |
| model/config creep speed, invalid settings/model guards and both T60 YAML variants | Execute .07 m/s when configured; both loaded default-off variants stay still on yaw-only command | Previous loader test only checked caps and never executed configured creep policy |
| one-step pulses / meaningful threshold | .05/.1 m/s isolated pulse stops immediately within 1 cm; one-degree request is admitted | New controls distinguish bounded intentional creep from noise and repeated drift |
| seven original physical witnesses | Coupled feasibility, achievable-speed steering, explicit creep, stop control, XY caps, asymmetric braking, signed reverse yaw | Previous tests inspected bounds/actions without executed physical yaw |

At reviewed head `44c386d6`, the new safety/intent/config selection gives
**21 failures / 4 passing controls**, with failures at physical speed/yaw,
missing-curvature and loaded-config assertions. The original seven physical
checks at base `879f75b6` give **6 failures / 1 stop-control pass**; fixed they
all pass. The focused three-file selection passes **49 tests**. Importing-area,
full slow-suite, seed-exclusion and script-gate receipts live in the lane report
and PR, bound to the accepted source. No skip/xfail or timeout was added to pass.

Three legacy assertions change deliberately in `test_classic_planner_adapter.py`:
`test_planner_action_adapter_bicycle_conversion` expects saturated .5 steering
at achievable .1 m/s; `test_bicycle_kinematics_model_projection_and_feasibility`
expects (0,0) for forbidden reverse; `test_bicycle_kinematics_model_allows_backwards_when_enabled`
rejects (-.5,.2) and accepts/projects (-.5,.1) at explicitly supplied curvature .2.

## Measurement and failure evidence

The complete **6,650-cell** fixed matrix ran on 32 Slurm CPUs (job16478),
producer `b81d58234439e58a5d86d8c4e7b9063a8c4dbacc`. The accepted
implementation matches all six measured runtime file hashes. All paired reset
states agree and learned checkpoint loads succeed without fallback. The extra
creep-on scenario arms supply counterfactuals alongside the requested default-off
scenario comparison. Historical controls and the exact reviewed-head replay remain
separate source-bound corpora.

Counts below pool all 14 planners. Parentheses are 95% Wilson rate intervals;
the linked CSVs contain every planner/scenario/bearing and the unrounded rates.

| Probe | Arm | N | Success | Collision | Timeout | Low-displacement windows |
|---|---|---:|---:|---:|---:|---:|
| Empty | DD | 350 | 305 (83.2–90.3%) | 0 (0.0–1.1%) | 45 (9.7–16.8%) | 260 |
| Empty | T60-30 | 350 | 210 (54.8–65.0%) | 0 (0.0–1.1%) | 140 (35.0–45.2%) | 52290 |
| Empty | T60-45 | 350 | 210 (54.8–65.0%) | 0 (0.0–1.1%) | 140 (35.0–45.2%) | 52290 |
| Empty | T60-30-on | 350 | 290 (78.6–86.4%) | 0 (0.0–1.1%) | 60 (13.6–21.4%) | 5190 |
| Empty | T60-45-on | 350 | 300 (81.7–89.0%) | 0 (0.0–1.1%) | 50 (11.0–18.3%) | 2340 |
| Six scenarios | DD | 840 | 479 (53.7–60.3%) | 145 (14.9–20.0%) | 216 (22.9–28.8%) | 3054 |
| Six scenarios | T60-30 | 840 | 425 (47.2–54.0%) | 141 (14.4–19.5%) | 274 (29.5–35.9%) | 38277 |
| Six scenarios | T60-45 | 840 | 419 (46.5–53.3%) | 159 (16.4–21.7%) | 262 (28.1–34.4%) | 34874 |
| Six scenarios | T60-30-on | 840 | 466 (52.1–58.8%) | 143 (14.6–19.7%) | 231 (24.6–30.6%) | 4772 |
| Six scenarios | T60-45-on | 840 | 451 (50.3–57.0%) | 161 (16.6–22.0%) | 228 (24.2–30.2%) | 4249 |

`T60-30`/`T60-45` are default creep-off; `-on` explicitly selects .1 m/s.

| Same legacy plant, empty world | Success / 350 | Low-displacement windows |
|---|---:|---:|
| Original base | 215 | 52290 |
| Fixed, creep off | 215 | 52290 |
| Fixed, creep on | 305 | 3510 |

Historical DD success is 305/350; the previous implicit-creep
implementation is 305/350. These historical controls are not the
new default-off T60 configs.

| T-intersection, all 14 planners × 10 seeds | Collision / 140 (95% Wilson rate CI) |
|---|---:|
| DD | 8 (2.9–10.9%) |
| T60-30 | 10 (3.9–12.6%) |
| T60-45 | 23 (11.2–23.4%) |
| T60-30-on | 10 (3.9–12.6%) |
| T60-45-on | 23 (11.2–23.4%) |

The reviewed-head T-intersection collision counts were DD8 / T60-30 10 / T60-45 23.

| Failure cohort | Adapter artefact | Planner/controller error | Genuine kinematic limit | Unclear |
|---|---:|---:|---:|---:|
| fixed-off | 0 | 365 | 0 | 0 |
| fixed-on | 0 | 127 | 0 | 0 |
| review-44 | 0 | 115 | 0 | 12 |

Unclear rows retain a specific missing-witness or motion-discrepancy reason.
The fixed matrix records 0 steps with creep after a recorded safety intervention.

The new stuck metric counts full, overlapping **2 s windows with net
translation below .05 m while at least one nonzero command is issued**.
Windows end every .1 s. Commands are raw v/yaw before projection, or native
command components for native policies; the nonzero threshold is 1e-6.
It counts safe stops and rotation without translation too, and is a progress
observation rather than a failure attribution. It is independent of creep or
achieved yaw. The old 54,360-to-zero yaw predicate claim is withdrawn.

Every failure row contains collision/timeout outcome, actual creep-firing count
`zero_turn`, terminal creep flag, minimum disc clearance over the final 20
available .1 s samples, terminal speed/yaw/safety state and displacement-window
count. The 127 reviewed-head failures were instrumented again: outcomes, steps,
resets and all 44,715 command/action/motion samples match the retained originals.
Case-specific numbers support each classification. A matched bicycle
counterfactual (including another planner) or physical empty-world trajectory
supports a controller error. Route witnesses require
completion, no contact flag and whole-path sampled disc clearance >= -1e-6 m;
their exact initial robot/pedestrian states and plant limits must match. This
certifies initial reachability, while later pedestrian reactions may diverge.
A recorded safety override or resolved implicit-creep failure supports an
adapter artefact. Twelve reviewed-head timeouts contain 107 sub-degree creep
steps in total; the corrected opt-in controller still fails, so their adapter
defect versus controller contribution remains unclear. The ledger records
noise/reverse creep counts rather than attributing those incidents from yaw-law
compliance alone. A genuine space/reverse limit needs a geometric certificate.
Remaining unclear rows state their measured state and the missing witness or
unresolved pose/velocity discrepancy;
no failure is declared a fundamental kinematic limit merely because DD succeeds.

## Reproduction and custody

The source-pinned matrix uses all 14 release planners, matched .64 m discs and
plant caps, explicit dev seeds 1001–1005 at empty bearings 0/45/90/135/180° and
1001–1010 at the six named KINPROBE scenarios. Empty goals lie 6 m away in a
400 m map, dt=.1 s and requested horizon600; authored scenario budgets and
runner outcome flags remain authoritative. Policy copies retain smaller speed
caps and bind explicit radius fields. Learned checkpoints load without fallback;
those policies are unchanged and may be outside their training distribution.
Wilson intervals are descriptive episode proportions; shared seeds/planners
are correlated, so these are not independent-seed ranking intervals.
No publication, release-admission or trained-bicycle-support claim follows.

Run `run_bicycle_probe.py --prepare`, then partition `--worker I --workers N`
with `--arms DD,T60-30,T60-45,T60-30-on,T60-45-on`; use `--arms BI-off,BI-on
--probe 1` for the same-geometry legacy controls. Set `BIKEFIX_OUTPUT` for each
corpus and `PYTHONPATH=.:fast-pysf` with the project Python. Physical reachability
is produced by `bicycle_empty_reachability.py` with the same output environment.
The committed summarizer emits the exact public filenames and schemas:

```bash
python scripts/validation/summarize_bicycle_probe.py \
  --input <fixed-raw> --legacy-input <original-raw> \
  --review-input <reviewed-head-replay> --reachability <physical-witness.json> \
  --output-root <repository-or-independent-replica>
```

[Comparison](bicycle_probe_comparison.csv),
[per-scenario/bearing results](bicycle_probe_per_scenario.csv),
[failure ledger](bicycle_probe_failures.csv), and
[summary](bicycle_probe_summary.json) carry all required counts and intervals.
CSV starts with its header, has no comment lines or dangling local path columns.
Review/custody information is in `.review.json` sidecars and the
[provenance JSON](bicycle_probe_provenance.json), including source/script hashes,
resolved axes and hashes over sorted raw results/traces/records. Regeneration
into an independent root must match all eight public CSV/JSON/sidecar files
byte-for-byte. Raw records and traces remain in the task lane, not release custody.
