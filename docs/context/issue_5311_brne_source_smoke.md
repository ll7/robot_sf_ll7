# Issue #5311 — BRNE source-side smoke + contract mapping

**Status:** smoke PASS; control-budget verdict **BORDERLINE (conditional go)**. The
upstream pure-numpy/numba core runs end-to-end against the staged source on the
redistributable dependency path and is under the 100 ms control budget for crowds
below the upstream default agent cap; it reaches ~1.0–1.2× budget at the default
8-agent cap. This note records the contract mapping and go/no-go data only —
**no robot_sf planner registration, no benchmark arm, no campaign, no vendoring**
(external source stays external). BRNE is planner candidate #4.

## Provenance

- **Algorithm:** BRNE — *Bayesian Recursive Nash Equilibrium* for social
  navigation. Paper: Max Muchen Sun, Francesca Baldini, Katie Hughes, Peter
  Trautman, Todd Murphey, "Mixed strategy Nash equilibrium for crowd navigation,"
  *The International Journal of Robotics Research (IJRR)*, 2024.
  <https://doi.org/10.1177/02783649241302342>.
- **Upstream repo:** `MurpheyLab/brne`
  <https://github.com/MurpheyLab/brne>, pinned at commit
  `633a5cdcb39ab27f18b596cb8cb1968644f82391` (2024-12-01, `main` HEAD).
- **License:** **GPL-3.0** (copyleft). Decision recorded in
  `scripts/tools/manage_external_repos.py`: **local-only staging; NOT vendored,
  NOT redistributed** in this non-GPL repository. The smoke loads the core module
  by file path from the local clone; no robot_sf code imports it as a package.
  Reuse in any derivative that ships BRNE source would require that work to be
  GPL-3.0.
- **Staging:** `scripts/tools/manage_external_repos.py stage brne` →
  `third_party/external_repos/brne` (gitignored local-only clone); provenance
  manifest at `output/external_repos/manifests/brne.provenance.json`.
- **What is exercised:** only the pure-numpy/numba core algorithm
  `brne_nav/brne_py/brne_py/brne.py` — the same module the upstream ROS2 node
  imports. **Not exercised:** the ROS2 navigation node (`brne_nav.py`, needs
  `rclpy`/ROS2), the PyTorch path (`brne_torch`), the C++ library (`brnelib`),
  and the bundled `socnavbench` benchmark study.

## Reproducer (the canonical smoke + measurement)

```bash
uv run python scripts/tools/manage_external_repos.py stage brne
uv run python scripts/tools/probe_brne_source_harness.py
uv run pytest tests/baselines/test_brne_source_smoke.py -q
```

Artifacts (git-ignored, regenerated each run):

- `output/issue_5311/brne_smoke.json` — machine-readable results + contract.
- `output/issue_5311/brne_smoke.md` — human-readable table + verdict.

Focused test (skips cleanly without the staged clone, so CI does not regress):
`tests/baselines/test_brne_source_smoke.py`.

## Per-step runtime vs neighbor count (the control-budget question)

Driven through the **real** upstream `brne.brne_nav(...)` against synthetic
Robot SF-shaped observations, reproducing the upstream `brne_cb` numerical
pipeline and upstream defaults (`num_samples=196`, `plan_steps=25`, `dt=0.1`,
`maximum_agents=8`, 10 best-response iterations). The first step includes numba
JIT compilation of the parallel cost/weight kernels; steady steps measure the
warmstarted solve the upstream node repeats on its 10 Hz replan timer.

Benchmark step budget = `robot_sf.sim.sim_config.SimConfig.time_per_step_in_secs`
default = **100 ms** (the upstream ROS2 node uses the same 100 ms budget via
`replan_freq=10 Hz`).

| Pedestrians | Agents (incl. robot) | First step (ms) | Steady mean/step (ms) | Steady max (ms) | × budget |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | ~2400 | ~9   | ~11  | 0.09× |
| 2 | 3 | ~20   | ~19  | ~20  | 0.19× |
| 3 | 4 | ~32   | ~39  | ~40  | 0.39× |
| 5 | 6 | ~65   | ~67  | ~77  | 0.67× |
| 7 | 8 | ~116  | ~115 | ~125 | 1.15× |

(Fresh numbers from `output/issue_5311/brne_smoke.json` on the run that produced
this note; absolute ms vary with the machine, the ~linear scaling and the budget
crossing at the 8-agent default cap are the robust signal.)

**Runtime model.** Per-solve cost is dominated by the pairwise proximity cost
matrix `(num_agents·num_samples)² × plan_steps` plus 10 best-response weight
updates; both are numba-parallelized. At fixed `num_samples=196` the cost scales
roughly **linearly** with agent count. The JIT compile (~2.4 s) is a one-time
warmup paid once per process, amortized across the whole episode.

**Verdict: BORDERLINE / conditional go.** Under the 100 ms control budget for
every crowd below the upstream default agent cap (8 agents / 7 pedestrians);
~1.0–1.2× budget exactly at the cap. Crucially, the upstream activation gate
(`brne_activate_threshold=3.5 m`) limits BRNE to the *nearby interacting* agents,
not the whole visible crowd — so in sparse-to-moderate crowds the interacting
set is small and BRNE is comfortably real-time. The risk concentrates only in
dense crowds where many pedestrians fall inside the activation radius. Compared
to SICNav (#4870, 2–14× over budget on its redistributable path), BRNE is
materially closer to real-time feasibility. A knob exists to recover budget
headroom if needed: reducing `num_samples` (the 196 default is generous) trades
equilibrium resolution for speed roughly linearly.

## Contract mapping

### 1. Available observations vs BRNE's oracle inputs

BRNE's upstream `brne_cb` consumes, per replan tick:

- robot pose `[x, y, θ]` (from odometry),
- robot goal `[x, y]`,
- each pedestrian's `[x, y]` **and** `[vx, vy]`,
- a `people_timeout` freshness filter on the pedestrian buffer.

Robot SF's canonical `Observation` (`robot_sf/baselines/interface.py`) carries
`robot.position`, `robot.velocity`, `robot.goal`, `robot.radius`, and per-agent
`position`/`velocity`. **Mapping is near-trivial** for the privileged (ground
truth) observation level. The "oracle" caveat is that BRNE needs pedestrian
*velocities* and a robot *heading* `θ`; Robot SF velocity is available at the
privileged level but must be supplied (or finite-differenced) under a noisy/
realistic perception level. The upstream node itself estimates pedestrian
velocity by finite differencing at a fixed perception cadence (33 Hz, see
`ped_cb`), so a Robot SF adapter would either reuse the privileged velocity or
replicate that differencing — an explicit transfer decision, not a silent
assumption.

### 2. Trajectory-distribution output vs our action interface

BRNE's output is a **mixed-strategy weight distribution over the robot's sampled
trajectory set** (`weights[0]`, a length-`num_samples` vector normalized to mean
1.0). The upstream node collapses this to an expected control command by
weighting the control-space ensemble: `cmd = mean(ulist_essemble · weights[0])`,
yielding a `[v, ω]` unicycle command published as a ROS `Twist`. So the
benchmark-facing action is a single `(v, omega)` pair per step — directly
compatible with Robot SF's `unicycle_vw` action (`{"v", "omega"}`). The full
trajectory distribution is consumed internally for the Nash-equilibrium solve
and is not needed downstream. An adapter returns `{"v": float, "omega": float}`.

### 3. Static-obstacle handling

**This is the sharpest contract limitation.** BRNE handles only **corridor
bounds**: the `coll_beck` mask zeros out trajectory samples whose y-coordinate
leaves `[corridor_y_min, corridor_y_max]` (upstream defaults `±0.65 m`). There
is **no representation of arbitrary static obstacle geometry** (polygons, walls,
doorways, round obstacles). Robot SF scenarios include rich static obstacle maps
that BRNE has no native mechanism to avoid. Any future integration would require
either (a) restricting BRNE to corridor-like scenarios where bounds suffice, or
(b) an explicit, documented extension that injects static-obstacle cost terms
into the `costs_nb` proximity matrix — which departs from upstream-faithful
behavior and must be labeled as such. This is the dominant barrier beyond the
control budget.

### 4. Differential / Ackermann command mapping feasibility

**Native and direct.** BRNE integrates **unicycle dynamics** `[v·cos θ,
v·sin θ, ω]` via RK4 (`dyn_step`/`traj_sim_essemble`) and outputs `[v, ω]`.
Robot SF's unicycle robots use the same `(v, ω)` command space, so **no
projection is required** — unlike holonomic-output planners (CrowdNav family,
#4871) which need a holonomic→unicycle remap. This is the cleanest action
contract among the recent comparator smokes. (Caveat recorded: the single-
trajectory helper `traj_sim` is buggy upstream — it omits the required `dt`
argument to `dyn_step` — but the navigation loop uses the correct
`traj_sim_essemble`, which threads `dt`. An adapter must use the ensemble path.)

### 5. Runtime vs neighbor count

See the table and verdict above. Short form: ~linear in agent count, under the
100 ms budget up to ~6 agents, ~1.0–1.2× at the 8-agent default cap; JIT compile
(~2.4 s) amortizes once per episode. Real-time feasibility hinges on the
activation gate keeping the *interacting* agent set small.

### 6. Qualitative comparison axes vs ORCA, prediction MPC, SICNav

| Axis | BRNE | ORCA | Prediction MPC | SICNav |
| --- | --- | --- | --- | --- |
| Decision model | Mixed-strategy **Nash equilibrium** over trajectory distributions (game-theoretic, cooperative) | Velocity Obstacles (reactive, non-game-theoretic) | Model-predictive control with predicted human motion (optimization, open-loop optimal w.r.t. prediction) | MPC with ORCA-modeled humans + KKT inverse game (implicit Nash) |
| What it adds | Explicit reasoning about *cooperation/reciprocity* and freezing-robot failure modes; outputs a *distribution* over intent | Fast, well-understood baseline; no cooperation modeling | Predictive lookahead; depends on predictor quality | Game-theoretic + optimization; heavy solver |
| Output | `(v, ω)` unicycle | `(vx, vy)` holonomic (typically) | platform-dependent | `(v, ω)` unicycle (via wrapper) |
| Static obstacles | Corridor bounds only | Native (ORCA obstacle handling) | Native (constraint set) | Native (MPC constraints) |
| Control budget (this repo's path) | ~under 100 ms ≤6 agents; ~1× at 8-agent cap | trivial (<<1 ms) | varies | 2–14× over 100 ms (#4870) |
| Robot SF crowd model fit | Needs ORCA or SF pedestrian policy; BRNE samples humans as GP motion — model-agnostic on human side | matches ORCA-pedestrian Robot SF mode | predictor-dependent | needs ORCA-pedestrian mode |
| Dependency weight | numpy+scipy+numba (light) | already integrated | varies | casadi+IPOPT+rvo2+gym (heavy) |

**Scientific distinctness (the reason BRNE was selected):** BRNE is a
*game-theoretic interaction-planning* comparator — qualitatively more distinct
from the existing RL/ORCA/MPC entries than another velocity-obstacle variant,
and it directly targets the freezing-robot / cooperation axis that the current
roster under-covers. This distinctness is the motivation for keeping BRNE as a
candidate despite the static-obstacle limitation.

## Go / no-go recommendation

**Conditional GO for a bounded integration follow-up — NOT a blanket go.**

The control budget, the issue's stated gate, is met for sparse-to-moderate
crowds (≤6 interacting agents) and is only marginally (~1×) exceeded at the
upstream default 8-agent cap, with a clear `num_samples` knob to recover
headroom. The action contract (native unicycle) is the cleanest of the recent
smokes and needs no projection. The observation contract is near-trivial at the
privileged level.

Two conditions gate any integration work and must be resolved *before* a
benchmark arm is considered:

1. **Static-obstacle handling.** BRNE has no native mechanism for arbitrary
   static obstacle geometry (corridor bounds only). Robot SF scenarios rely on
   rich obstacle maps. Either restrict BRNE to corridor-class scenarios or
   design and clearly label a static-obstacle cost extension — this is the real
   scientific/engineering cost of integration, not the runtime.
2. **Crowd model.** BRNE samples humans as Gaussian-process motion around a
   constant-velocity mean; it does not assume a specific pedestrian simulator.
   A fair comparison would still need to declare the Robot SF pedestrian policy
   (social-force vs ORCA) used during evaluation, as for SICNav (#4870) and the
   CrowdNav family (#4871).

If those two are accepted as scoped follow-up design decisions, BRNE is a viable
**prototype-only / integrate-next** candidate for a corridor-class,
moderate-density slice. If a benchmark arm must cover dense crowds with complex
static geometry on the redistributable path, the answer is **no-go for that
arm** and BRNE stays assessment-only. **No-go is a valid, complete outcome of
this issue**; the recommendation here is deliberately conditional rather than
unconditional, per the issue's acceptance criteria.

## Out of scope (per issue)

- No robot_sf planner registration (`algorithm_metadata`, readiness roster, etc.).
- No benchmark arm, no campaign, no training run.
- No vendoring of BRNE source (GPL-3.0).
- No re-evaluation of the maintainer-approved research plan or smoke scope.

## Related

- `docs/context/external_planner_reuse_checklist.md` — the reusable intake
  checklist this smoke follows (provenance → source harness → contract → verdict).
- `docs/context/issue_4870_sicnav_smoke.md` — closest analog (MPC comparator,
  runtime-vs-neighbor-count, control-budget framing); same 100 ms budget gate.
- `docs/context/issue_4871_crowdnav_pred_attng_smoke.md` — learned-baseline
  comparator smoke; same contract-mapping structure.
- `docs/ai/planner_zoo_context.md` — planner-zoo readiness frame (BRNE remains
  "conceptually adjacent only" until a separate maintainer roster call).
