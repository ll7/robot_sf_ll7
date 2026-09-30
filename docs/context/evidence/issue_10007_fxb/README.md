# FXB baseline controller diagnostics — issue #10007

Evidence status: **diagnostic-only**, development seeds 1001–1010, three
requested scenarios, H600 and dt 0.1. These results establish controller
defects and implementation changes; they do not admit release evidence or
establish general performance/safety improvements. Native RVO2 and fast-pysf
were used, without fallback or degraded rows. The before/after episode matrix
is complete (10 rows per scenario per arm/config).

**Execution deviation:** an initial broad pytest selection accidentally ran
four historical social-force episode tests, including held-out seeds 111/112
and a non-dev seed. That violated this lane's seed restriction. That entire
test run is excluded from FXB evidence; its private log remains separately
preserved. No held-out measurement enters the tables or committed raw bundle.
The later 207-test selection uses explicitly inspected nodes and has no skips,
xfails, or timeout changes. The accidental run exposed an existing historical
spin-bound assertion that fails after correcting dt; its old 0.5-s planner
clock did not match the 0.1-s simulation clock. It is **not** claimed resolved
or validated here. Do not tune against its held-out outcome. The orchestrator
should reassess that legacy tuning with dev seeds before broad readiness.

Base initially observed: `d41cceb7f9422337bd34b602ae49f3dddbddb2a5`.
Parent subsequently advanced by a merge of main; this lane rebased onto
`b3fa204dc4eda5004d67e4fcf518f3370fcd0966` and repeated social-force and
candidate-ORCA baseline episodes and regression failures there. The relevant
planner, map-runner, simulator, observation, algorithm config, and scenario
source bytes did not change between those parent heads. Fixed controller
source/probe revision: `6fd6d463c0422367802787c3c1ae2cb28347ad5d`.
Subsequent delivery changes contain diagnostics, documentation, and test
collection wiring; [source.json](source.json) pins the unchanged controller
bytes. The release-template ORCA comparator was measured at the original
parent with the same source/config bytes.

## F1 — confirmed on the real runner; B5 also confirmed

The first policy call in `classic_head_on_corridor_medium`, seed 1001,
receives a **flat** observation with `sim_timestep=0.10000000149011612` and
no `sim` block. Nothing normalizes it before the common adapter calls
`adapter.plan(obs)` (`map_runner.py:2140` at the reviewed parent).
Social force resolves **0.5 s**, its relaxation tau, on every recorded baseline
step. After the correction it resolves the observed **0.10000000149011612 s**.
The first `(v,w)` changes from `(0.9813681200523284, 0.19333665068859673)` to
`(0.3902016186362373, 0.19333665068859673)`. These are adapter commands;
the drive starts at zero and obeys its acceleration bounds.

Root cause: `_resolve_dt` only read nested `sim.timestep` and substituted
`social_force_tau`; this changed both force integration and filtering.
The shared `_simulation_timestep` resolver (`socnav_occupancy.py:17`) now
reads nested/flat observed time, validates a finite positive scalar, and
rejects missing/invalid clocks. SF uses it at `socnav_social_force.py:476`
and records the source in diagnostics. Native ORCA, heuristic ORCA, HRVO,
SA-CADRL, and bounded-v2 sampling also use that resolver. Flat observation
bytes are preserved, including learned-policy inputs.

B5: replaying the real reset bytes with dt=0.2 shows native RVO2's clock
changing from **0.10000000149 to 0.20000000298 s**. All **52** sampling
rollouts change from **0.1 to 0.20000000298 s**. A controlled SA-CADRL action
of 0.05 rad changes from **0.5 to 0.25 rad/s**, the independently calculated
0.05/0.2 result. Missing, zero, negative, NaN and infinite clocks previously
produced commands; SA-CADRL and bounded-v2 sampling now raise `ValueError`
at the timestep contract. The SA-CADRL test isolates decoding with a controlled
model, and makes no checkpoint/episode-performance claim.

## F2 — confirmed; differential-drive conversion corrected

The old heading slowdown capped **maximum speed**, rather than reducing the
requested speed. With the default slowdown 0.2 and a 2.0-m/s cap, 1.5-m/s
requests at 0/45/90/135/180 degrees all commanded **1.5 m/s**. The fixed
commands are **1.5 / 1.060660172 / 0 / 0 / 0 m/s**. The turn command remains
bounded by 1 rad/s. The same observed heading bytes are used in both probes.

Rule (`socnav_orca.py:972`): after occupancy steering, let `e` be the heading
error, `s` the ORCA speed, `p` the occupancy penalty, and `k` the configured
heading slowdown. Command
`v=min(s,v_max)*(1-p)*min(max(0,cos(e)), max(0,1-min(1,abs(e)/(pi/2))*k))`.
Thus sideways/backward targets rotate with zero **commanded** translation;
an optional stronger configured slowdown can reduce speed further. This
applies heading/occupancy reductions to the requested speed, not only the
maximum-speed cap. Physical braking takes time under the native drive limits;
this adaptation does not guarantee that the realized nonholonomic velocity
satisfies ORCA half-planes.

Release-template, heading error >90 degrees and adapter command >0.5 m/s:
doorway **4/1848 (0.2165%) → 0/1782** steps; corridor **0/1472 → 0/1477**;
frontal approach **0/1734 → 0/1740**. Defined-angle adapter-trace p95 (rad)
changes **0.661528 → 0.440808**, **0.369497 → 0.368697**, and
**0.232090 → 0.219789**, respectively. The existing trace reports no angle
when translation is zero, so these percentiles cover its defined-angle subset.
Independent physical-speed measurements at decision time, using the raw
planned vector's angle to the current heading, give bad-speed counts
**2 → 0**, **6 → 6**, **0 → 2**. Those remaining steps reflect current
drive motion and raw ORCA/occupancy steering disagreement; zero command is
not instantaneous braking. No physical-safety improvement is inferred.

## F3 — refuted for the release template, confirmed for the candidate

At the requested branch head, the release template already supplies
`socnav_release_v0_0_8.yaml` with **max_linear_speed=2.0**. The hunt's
"no config / 3.0" claim is therefore wrong for this template. Its native
solver-cap regression passes on the base and is a confirmation control.
The separate mutable `*_v0_0_8_candidate.yaml` has no ORCA config and really
builds native RVO2 with **3.0 m/s**. Its base regression fails at the native
`getAgentMaxSpeed` result **3.0 != 2.0**.

Both mutable manifests now bind the explicit
`configs/algos/orca_release_v0_0_8.yaml` with linear/angular limits 2.0/1.0.
Geometry remains observation-derived. On real corridor seed 1001, the candidate
solver's first requested magnitude is **3.00000008104 → 2.00000001519 m/s**;
the adapter speed is **2.85029384029 → 1.84834729040 m/s**. The old raw
adapter request is subsequently projected by the map runner to at most 2.0;
it is not the physical robot speed. Candidate bad-command counts >90 degrees
and >0.5 m/s change **11/2122 → 0/1782**, **5/1521 → 0/1477**, and
**1/1715 → 0/1740**. Candidate outcomes below combine F2 and F3; they do
not isolate either fix's causal episode effect.

No `*_frozen.yaml`, 0.0.2, or 0.0.7 artifacts were edited.

## Pedestrian-term lead — mechanism confirmed, current release claim refuted

The current template already selects **surface_v3** via
`social_force_release_v0_0_8.yaml`; the old candidate SF config retains
`legacy_kernel`. No selector was changed in this lane.
`probe_issue_10007_fxb_ped_term.py` checks release radii (1.0/0.4 m), dt=0.1,
and the native acceleration-limited drive against a stationary pedestrian
1.8 m ahead, over 80 steps. Legacy minimum clearance is **−0.915563 m**, with
first contact at **step 10**; surface-v3 minimum clearance is **+0.360053 m**,
without contact. This is a synthetic mechanism diagnostic, not a map episode.
At 1.4-m centre separation and robot speed 1 m/s, the wrapped legacy force is
`[-1.34434542146, +1.34434542146]`, or weighted
`[-1.07547633717, +1.07547633717]`; the code comment's "about 0.09" number
does **not** describe the current wrapped-kernel configuration. The contact
mechanism still reproduces. Recommendation: retain the existing release
surface-v3 selection; have the orchestrator reconcile/disclose the stale
candidate legacy selection and the comment's numeric context. This lane
does not switch either method.

## Ten-episode outcomes per scenario

S/C/T means success/collision/timeout counts out of ten. Clearances are the
mean of each episode's minimum pedestrian surface clearance; jerk is the
existing per-step proxy. The sample is small and has no held-out status.

### F1: social force, release template

| Scenario | S/C/T before → after | Mean min clearance (m) before → after | Mean jerk proxy before → after |
|---|---|---|---|
| classic_doorway_medium | 10/0/0 → 10/0/0 | 0.518717 → 0.517252 | 0.127716 → 0.053890 |
| classic_head_on_corridor_medium | 9/0/1 → 9/0/1 | 0.465501 → 0.382169 | 0.020191 → 0.017508 |
| francis2023_frontal_approach | 0/0/10 → 0/0/10 | 0.418698 → 0.297892 | 0.021857 → 0.022539 |

### F2: ORCA, release template (already 2.0 m/s)

| Scenario | S/C/T before → after | Mean min clearance (m) before → after | Mean jerk proxy before → after |
|---|---|---|---|
| classic_doorway_medium | 9/1/0 → 9/1/0 | 0.126259 → 0.139880 | 0.112865 → 0.112582 |
| classic_head_on_corridor_medium | 8/2/0 → 8/2/0 | 0.140184 → 0.139086 | 0.048710 → 0.050676 |
| francis2023_frontal_approach | 10/0/0 → 10/0/0 | 0.146331 → 0.148439 | 0.036649 → 0.034777 |

### F2+F3: ORCA, candidate manifest (3.0 to 2.0 m/s)

| Scenario | S/C/T before → after | Mean min clearance (m) before → after | Mean jerk proxy before → after |
|---|---|---|---|
| classic_doorway_medium | 8/1/1 → 9/1/0 | 0.134932 → 0.139880 | 0.121558 → 0.112582 |
| classic_head_on_corridor_medium | 9/1/0 → 8/2/0 | 0.159425 → 0.139086 | 0.043162 → 0.050676 |
| francis2023_frontal_approach | 10/0/0 → 10/0/0 | 0.124845 → 0.148439 | 0.036459 → 0.034777 |

## Test-value gate and validation

All controller regressions use the real seed-1001 reset archive in
`tests/benchmark/fixtures/issue_10007_fxb/head_on_1001_reset.npz`, including
the true flat timestep, actors, radii, pose and grid. Inputs are numerical
arrays and loaded with `allow_pickle=False`. No production test seam was added.

| Test family | Protected defect / credible regression | Existing coverage gap | Independent oracle / real bytes |
|---|---|---|---|
| SF flat dt | Reading only nested time restores tau=0.5 | Existing SF fixtures provide nested sim | Actual runner bytes, literal true dt=0.1 |
| SF representation parity | Flat/nested clocks change the same physical command | Nested fixture tests omit flattened input | Same physical bytes; representation-invariance check |
| SA-CADRL decoding | Default 0.1 silently changes turn rate | Existing decoding uses nested 0.1 | Real state, controlled action; hand division 0.05/0.2 |
| Native ORCA clock | Nested-only read changes RVO2 integration | Native cache tests use nested 0.1 | Native solver getTimeStep, literal 0.2 |
| Sampling rollout clock | Rollouts receive a default rather than sim time | Existing bounded-v2 fixtures use nested 0.1 | Wrap actual rollout calls, assert all 52 receive 0.2 |
| Missing/invalid dt | Defaults silently admit invalid observations | Existing valid-clock tests miss contract failures | Delete/alter only real time bytes; exact timestep ValueError |
| Release solver cap | Candidate loses config and solves at 3 instead of 2 | Template-only binding already passes | Actual manifest/YAML/policy builder/native solver getter |
| Heading projection | Default slowdown again becomes nonbinding | Old behind/sideways tests force slowdown=1.0 | Real heading; analytic 1.5*cos(45°), zero sideways/backwards |
| Existing adapter trace | Trace loses the reduced forward component | Previous trace assertion assumed no speed loss | [0.6,0.6] has forward 0.6 and magnitude sqrt(0.72) |

Base command (isolated latest-parent worktree, project environment,
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`):

```sh
python -m pytest -n0 tests/benchmark/test_issue_10007_fxb.py \
  tests/test_socnav_planner_adapter.py::test_orca_velocity_projection_records_optional_adapter_trace -q
```

Result: **21 failed, 2 passed**. The two passing cases are unchanged forward
motion and the already-correct template cap; neither is offered as fix proof.
Exact output is in [latest_base_regressions.txt](latest_base_regressions.txt).
Representative failure lines:

```text
E       assert 0.5 == 0.1 ± 1.0e-07
E       0.9813681200523284 | 0.3902016186362373 ± 3.9e-07
E       assert 0.5 == 0.25 ± 2.5e-07
E       assert 0.10000000149011612 == 0.2 ± 2.0e-07
E       [0]: 0.1 (ACTUAL), 0.2 (DESIRED)
E       assert 3.0 == 2.0 ± 2.0e-06
E       Failed: DID NOT RAISE <class 'ValueError'>
E       Obtained: 1.5
E       Expected: 0.0 ± 1.0e-08
E       assert 0.848528137423857 == 0.6 ± 6.0e-07
```

Fixed selection: **207 passed**, no skips/xfails. The exact `uv run pytest -n0`
argv is in [safe_test_command.json](safe_test_command.json), with the output
in [final_targeted_tests.txt](final_targeted_tests.txt). It covers these new
regressions and existing ORCA, SF, SA-CADRL, sampling, pedestrian-force and
map-runner integration tests, excluding the four historical episode nodes
whose seeds violate this lane's explicit restrictions. Ruff check/format and
`git diff --check` passed. Full repository readiness was not run: it would
invoke prohibited historical episode seeds. This is not a full-suite green
claim; the historical assertion above remains an explicit limitation.

## Preservation and reproduction

[measurements.json](measurements.json) includes each episode outcome,
clearance/jerk measurements, first ten seed-1001 actions per scenario, trace
statistics and raw source-file hashes. [pedestrian_term.json](pedestrian_term.json)
contains the standing-pedestrian probe. [manifest.json](manifest.json) pins
the compact files and the local raw bundle `~/fxb_evidence.tar.gz` (native
episodes/actions, valid test logs, package freeze and source pins).
Failed map-path attempts and the contaminated historical test run are kept
separately and excluded from the bundle/measurements.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python \
  scripts/validation/probe_issue_10007_fxb.py --algo social_force --out <fresh-sf-dir>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python \
  scripts/validation/probe_issue_10007_fxb.py --algo orca --out <fresh-orca-dir>
# Candidate comparison: add --manifest
# configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate.yaml
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python \
  scripts/validation/probe_issue_10007_fxb_ped_term.py
```

Use the pinned before/fixed source revisions in separate worktrees. There were
at most two simultaneous simulation processes. Do not compare these results
as if they were held-out release evidence. The episode metrics themselves
remain unchanged; `jerk_mean` is the existing per-step jerk proxy.

Next disposition: draft PR for orchestrator review; no merge, release or
evidence-admission action. Reassess legacy tuning and the seed-policy deviation
before any broader readiness or release conclusion.
