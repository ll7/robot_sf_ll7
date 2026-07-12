# Issue #5355 — Prediction × Constraint 2×2 Factorial Pre-Registration (option-1 base)

- **Claim boundary**: protocol evidence only. This note fixes the comparison surface,
  arms, endpoints, contrasts, multiplicity policy, seed budget, stop rules, and fidelity
  checks *before* any campaign submission. It promotes no benchmark, paper, or release
  claim, and it authorizes no GPU submission.
- **Evidence status**: `pre-registration` (design-grade). No rows executed. Cross-planner
  results referenced below are cited **qualitatively only**.
- **Base-planner decision**: **option 1** (maintainer, 2026-07-12) — the existing
  prediction-MPC framework hosts the factorial. Its native prediction consumption is
  Factor A's ON state; its explicit collision-constraint set is the toggleable Factor B.
- **Uncertainty on the design being executable as written**: moderate. Two config
  toggles named below **do not exist yet** and are recorded as binding implementation
  preconditions (§8). The intervention-fidelity requirement for Factor B (§1.3, §6) is the
  load-bearing risk the maintainer flagged and is addressed explicitly.

Issue: <https://github.com/ll7/robot_sf_ll7/issues/5355>

---

## 0. Base planner and what "option 1" concretely means in code

The base is `robot_sf/planner/prediction_mpc.py :: PredictionMPCPlannerAdapter`, a subclass
of `robot_sf/planner/nmpc_social.py :: NMPCSocialPlannerAdapter`. Reading the implementation
disentangles the two factors precisely:

- **The optimizer core** (shared, never toggled): SLSQP over a flattened unicycle
  `(v, w)` control sequence (`NMPCSocialPlannerAdapter.plan` -> `scipy.optimize.minimize`,
  `method="SLSQP"`). Cost terms: path/terminal goal distance, heading error, control
  effort, smoothness, progress, a static-obstacle soft-clearance term, an occupancy-grid
  term, a speed cap, and a symmetry-breaking preferred-avoidance-turn bias.
- **Factor A — prediction consumption.** Both the base soft-cost path
  (`NMPCSocialPlannerAdapter._predict_pedestrians`, which advances pedestrians by
  `ped_positions + ped_velocities * t`) and the hard-constraint path
  (`ConstantVelocityPedestrianPredictor.predict`, which advances by
  `ped_positions + ped_velocities_world * tau`) currently **always** propagate pedestrians
  forward at constant velocity. Prediction consumption = *using a forward-projected
  pedestrian future*; the OFF state = *freezing pedestrians at their current position*
  (zero-order hold / static-obstacle assumption).
- **Factor B — explicit constraint handling.** `PredictionMPCPlannerAdapter._optimizer_constraints`
  builds a `scipy.optimize.NonlinearConstraint` requiring squared robot-pedestrian
  distance >= `(robot_radius + ped_radius + margin)^2` at every horizon step
  (`_pedestrian_clearance_constraints`). The shipping prediction-MPC adapter also sets the
  base soft pedestrian term `pedestrian_clearance_weight = 0.0` in `_to_nmpc_config`
  (line ~299), i.e. its *only* pedestrian avoidance is the hard constraint set. Constraint
  handling = *hard time-varying pedestrian constraints*; the OFF state = *no hard
  constraints, pedestrian avoidance falls back to a soft clearance penalty*.

The shipping `configs/algos/prediction_mpc_cv_uncertainty_envelope.yaml` corresponds to the
**A-on / B-on** cell (with the issue-#4141 uncertainty envelope additionally enabled). The
uncertainty envelope is an **orthogonal knob and is held OFF (`enabled=false`, `alpha=0.0`)
for all four arms** so it cannot confound the factorial.

---

## 1. Arms (4)

All four arms are hosted by **one adapter with one frozen shared configuration**; arms
differ **only** in the two mechanism toggles. This is deliberate: matched-by-construction
capability parity is the fairness contract's (#5353) best-case demonstration.

### 1.1 Shared base configuration (identical in every arm)

Taken from the prediction-MPC constant-velocity config (envelope disabled). No value below
is tuned per arm.

| Field | Value | Field | Value |
| --- | --- | --- | --- |
| `max_linear_speed` | 0.9 | `terminal_goal_weight` | 5.0 |
| `max_angular_speed` | 1.1 | `heading_weight` | 0.5 |
| `horizon_steps` | 6 | `control_effort_weight` | 0.05 |
| `rollout_dt` | 0.25 | `smoothness_weight` | 0.2 |
| `goal_tolerance` | 0.25 | `pedestrian_safety_margin` / `pedestrian_margin` | 0.35 |
| `waypoint_switch_distance` | 0.75 | `static_obstacle_soft_weight` | 1.0 |
| `path_goal_weight` | 1.5 | `solver_ftol` | 1e-3 |
| `solver_max_iterations` | 40 | `warm_start` | true |
| `fallback_to_stop` | true | `pedestrian_uncertainty_envelope_enabled` | **false** |
| `pedestrian_uncertainty_alpha_mps` | **0.0** | `predictor_backend` | constant_velocity |

Local-minimum-escape heuristics present in the base core
(`avoidance_turn_bias_weight`, `symmetry_break_bias`, `_preferred_avoidance_turn`,
`hard_obstacle_guard_enabled`) are held **identical across all four arms** so they cannot
contaminate the Factor B contrast. Grading local-minimum handling as its own factor is
explicitly out of scope for this 2x2.

### 1.2 The two toggles

- **Factor A** — `prediction_enabled` (proposed field): `true` => pedestrians advanced at
  constant velocity across the horizon (native behavior); `false` => pedestrians frozen at
  their current position (velocity treated as zero) in **both** the soft-cost predictor and
  the hard-constraint predictor.
- **Factor B** — `hard_pedestrian_constraints_enabled` (proposed field): `true` =>
  `_optimizer_constraints` returns the hard time-varying pedestrian constraint and the soft
  pedestrian weight is `0.0`; `false` => `_optimizer_constraints` returns `()` and a
  **positive** soft pedestrian clearance weight `W_soft` is passed through to the cost.

`W_soft` is preregistered as a **single shared constant** = `4.5` (the base
`NMPCSocialConfig.pedestrian_clearance_weight` default), with `pedestrian_margin = 0.35`
matched to the B-on hard margin. It is **not** tuned per arm and **not** tuned to favor any
outcome (§2.4, §6.2).

### 1.3 Arm table (exact deltas)

| Arm ID | Factor A | Factor B | `prediction_enabled` | `hard_pedestrian_constraints_enabled` | soft `pedestrian_clearance_weight` | Pedestrian futures fed to planner |
| --- | --- | --- | --- | --- | --- | --- |
| **A00** | off | off | `false` | `false` | `4.5` (W_soft) | frozen @ current position, soft penalty |
| **A10** | on | off | `true` | `false` | `4.5` (W_soft) | constant-velocity forward, soft penalty |
| **A01** | off | on | `false` | `true` | `0.0` | frozen @ current position, hard constraint |
| **A11** | on | on | `true` | `true` | `0.0` | constant-velocity forward, hard constraint |

**A11 is the shipping prediction-MPC behavior** (constant-velocity futures as hard
constraints), with the uncertainty envelope disabled.

**What remains functional in the B-OFF arms (A00, A10) — the intervention-fidelity
requirement, documented exactly.** Removing the hard constraint set removes *only* the
`NonlinearConstraint`. Everything else stays live: the full SLSQP optimizer over unicycle
controls; path/terminal-goal, heading, control-effort, smoothness, and progress cost terms;
the static-obstacle soft-clearance term; the occupancy-grid term; the speed cap; the
symmetry-breaking avoidance-turn bias; and a **positive soft pedestrian clearance penalty**
(`W_soft = 4.5`, logistic `_soft_collision_cost`). A B-OFF arm is therefore a genuine
soft-cost social planner (equivalent to the base NMPC-Social pedestrian handling), **not** a
degenerate planner with pedestrian avoidance switched off. This is the maintainer's binding
condition: the B contrast measures *hard-constraint vs. soft-penalty avoidance*, not
*avoidance vs. breakage*. The shipping adapter's `pedestrian_clearance_weight = 0.0` would
have produced exactly the degenerate B-OFF the maintainer warned against; §6.2 verifies the
functional value empirically before any campaign.

---

## 2. Matched budgets

### 2.1 Observation contract
All four arms consume the **SocNav structured observation** via
`NMPCSocialPlannerAdapter._extract_state` -> `_socnav_fields` (robot pose/heading/speed/radius,
current + next goal, pedestrian positions/velocities/count/radius, and the occupancy-grid
payload). Because all arms are the same adapter, the observation mode is identical **by
construction** — no per-arm adapter, no per-arm observation access. This is the parity the
#5353 capability matrix must confirm, and here it is a tautology rather than a claim.

### 2.2 Action space / kinematics
Identical across arms: differential-drive unicycle `(v, w)`, `v in [0, speed_cap]`,
`w in [-1.1, 1.1] rad/s`, `max_linear_speed = 0.9 m/s`; rollout integrator
`theta <- wrap(theta + w*dt)`, `x <- x + [v*cos(theta)*dt, v*sin(theta)*dt]`, `dt = 0.25 s`,
horizon 6 steps (1.5 s look-ahead).

### 2.3 Runtime budget per decision
Identical solver budget in every arm: SLSQP, `solver_max_iterations = 40`,
`solver_ftol = 1e-3`, warm-start on, `fallback_to_stop` on. Factor B changes the *number of
constraints* handed to the same solver under the same iteration cap, so per-decision compute
is matched to within solver-internal constraint-evaluation cost (recorded in diagnostics, not
re-tuned).

### 2.4 Tuning protocol (and why it is equal)
**No per-arm tuning.** Every arm inherits the single frozen shared config in §1.1; the arms
differ only in the two mechanism toggles and the mechanically-implied `W_soft`/soft-weight
bookkeeping in §1.3. Equality is guaranteed by construction: there is one config, not four
tuned configs. The only free calibration in the entire design — `W_soft` for the B-OFF arms
— is fixed at the base default `4.5` before any results are seen and is shared by both B-OFF
arms, so it cannot be adjusted to advantage either the A or the B contrast.

### 2.5 Scenario matrix (concrete, existing, failure-expressing)
**`configs/scenarios/classic_interactions_francis2023.yaml`** (merges
`classic_interactions.yaml` + `francis2023.yaml`; the exact scenario-matrix sha256 is
recorded when the campaign config lands, per §7). Justification — both factors have strata
where their mechanism is load-bearing, so the matrix can express A, B, **and** A×B effects
rather than saturating:

- **Static-structural / local-minimum stratum** (expresses constraint-handling and
  local-minimum failures): `classic_bottleneck`, `classic_realworld_bottleneck`,
  `classic_doorway`, `classic_head_on_corridor`, `classic_cross_trap`,
  `classic_t_intersection`, `francis2023_narrow_hallway`, `francis2023_narrow_doorway`,
  `francis2023_blind_corner`, `francis2023_entering/exiting_*`.
- **Dynamic-interaction stratum** (expresses prediction-relevant failures):
  `francis2023_frontal_approach`, `francis2023_pedestrian_overtaking`,
  `francis2023_robot_overtaking`, `francis2023_circular_crossing`,
  `francis2023_perpendicular_traffic`, `francis2023_parallel_traffic`,
  `francis2023_crowd_navigation`, `classic_group_crossing`, `classic_urban_crossing`,
  `classic_merging`.

A matrix with only easy scenarios would floor/ceiling all arms and mask the contrasts; this
matrix deliberately spans the failure modes each mechanism targets.

### 2.6 Seed budget with a power argument
- **Seed set: `paper_eval_s30`** (30 seeds, `111..140` inclusive) from
  `configs/benchmarks/seed_sets_v1.yaml`.
- **Design is fully paired**: each `(scenario, seed)` cell is played by all four arms, so the
  2x2 is closed within every cell. The relevant sampling variance for each contrast is the
  **within-pair** difference variance, not the (much larger) between-arm variance.
- **Qualitative anchor from prior data (cited qualitatively only, per the issue):** existing
  *cross-planner* S30 evidence shows constraint-first structured planners separated with
  disjoint confidence intervals **above** prediction-equipped planners — a large but
  **confounded** cross-planner gap (planners differ in adapter, observation access, and
  tuning). The matched within-planner mechanism contrast targeted here is expected to be
  **substantially smaller** than that confounded gap; the paired design is what makes a small
  effect resolvable.
- **Power sketch:** with ~36 scenarios x 30 seeds ~= **1,080 paired episodes per arm**, a
  paired binary test on collision-free completion (McNemar / hierarchical paired resample,
  §4) resolves an absolute completion difference on the order of **0.03-0.05** at
  power >= 0.8 for the discordant-pair rates these failure-expressing strata produce. S30 is
  the primary schedule. **S20 -> S30 escalation** is predeclared: if an S20 interim contrast is
  directionally clear but its CI straddles the practical-effect threshold, expand to the full
  S30 (no other extension is permitted — see §5).

---

## 3. Endpoints (typed ledger)

- **Primary — collision-free completion.** The repo `success` / `success_rate` semantics:
  route completion before `horizon` **and** zero total collisions
  (`wall_collisions + agent_collisions + human_collisions`). Read from the typed event
  ledger and reconciled via `event_ledger_reconciliation`; per-arm rows must carry the
  typed-ledger fields (no imputed successes).
- **Secondary — near-miss exposure, normalized.** `near_misses` = count of steps with
  `0 <= min_clearance < D_NEAR` (`NEAR_MISS_DIST = 0.50 m`,
  `robot_sf/benchmark/constants.py`), **normalized per interaction opportunity** (per episode
  step with >=1 pedestrian within sensing range). The opt-in TTC near-miss diagnostic
  (`near_miss_ttc`) is reported as a robustness companion; it must be deterministic
  (`near_miss_determinism`) or the run is invalid (§5).
- **Secondary — censored time-to-goal.** Time-to-goal with **right-censoring at `horizon`**
  for non-completions, analyzed via a survival/AFT treatment. Note: the repo's
  `time_to_goal_norm` imputes `1.0` for failures; this pre-registration explicitly adopts the
  **censored** treatment (failures = right-censored, not mean-imputed) for the secondary
  survival estimand, using `time_to_goal_norm_success_only` for the observed-completion times.

---

## 4. Estimands, contrasts, and multiplicity

Let the collision-free-completion indicator for arm `Aab` in cell `(scenario s, seed k)` be
`Y_ab(s,k)`. All contrasts are formed **at the paired `(planner, scenario, seed)` level** —
the same base planner and the same `(s,k)` produce all four arms, so every contrast is a
within-cell difference.

- **Factor A main effect:** `d_A = mean_{s,k}[ 0.5*(Y_11 + Y_10) - 0.5*(Y_01 + Y_00) ]`
  (prediction on - off, marginal over B).
- **Factor B main effect:** `d_B = mean_{s,k}[ 0.5*(Y_11 + Y_01) - 0.5*(Y_10 + Y_00) ]`
  (constraint on - off, marginal over A).
- **A×B interaction:** `d_AB = mean_{s,k}[ (Y_11 - Y_01) - (Y_10 - Y_00) ]`.

**Inference (analysis machinery = #5351, hierarchical paired):** hierarchical family
resampling that respects the nested structure — resample **scenario strata**, then scenarios
within stratum, then seeds within scenario — to produce cluster-robust CIs and paired
p-values for each contrast. Practical-effect thresholds and the practical-vs-statistical
reporting follow #5351's convention.

**Multiplicity policy:** **Holm-Bonferroni across the 3 primary contrasts** (`d_A`, `d_B`,
`d_AB`) at family alpha = 0.05 on the **primary** endpoint (collision-free completion). The two
secondary endpoints (normalized near-miss, censored time-to-goal) are each reported with
their own 3-contrast Holm family, labeled **secondary / supporting** — they do not gate the
primary conclusion.

---

## 5. Stop rules

- **Adverse / null result is a binding narrowing outcome (verbatim, per the promotion
  conditions):** *"an explicit adverse-result stop rule — a null or ambiguous result is a
  binding outcome (it forces claim narrowing downstream), not a reason to extend the
  search."* Concretely: a null or CI-straddling result on `d_A`, `d_B`, or `d_AB` is
  recorded as the finding and forces downstream claim narrowing. It does **not** authorize
  adding arms, adding seeds beyond the predeclared S20->S30 escalation, re-tuning `W_soft` or
  any cost weight, swapping the scenario matrix, or re-running to chase significance.
- **Compute abort criteria:** abort a run if per-arm GPU lifecycle / subprocess isolation
  fails (coordinate with the #4826-class isolation gate that already killed prior campaigns);
  if any arm's solver `fallback_to_stop` rate exceeds a preregistered ceiling (default **> 60 %**
  of decisions) indicating a degenerate planner; or if the fidelity smoke (§6) has not passed
  for the exact config being submitted.
- **What invalidates a run (row-level fail-closed):** scenario-matrix sha256 != the
  preregistered hash; seed set != `paper_eval_s30` (`111..140`); any arm config differing from
  the §1.1 shared config in **anything other than the two toggles + the implied
  soft-weight bookkeeping**; the uncertainty envelope accidentally enabled on any arm;
  non-deterministic near-miss ledger (`near_miss_determinism` fails); or missing typed-ledger
  collision/near-miss/time fields.

---

## 6. Fidelity checks (pre-campaign, CPU)

Both checks must pass on the **exact** submitted config before any GPU submission.

### 6.1 Toggle-effect smoke (each factor demonstrably changes behavior)
On **2 scenarios** — one static-structural (`classic_doorway`) and one dynamic
(`francis2023_circular_crossing`) — run all four arms for a small fixed seed set and compare
**per-arm decision traces** (first-step `(v, w)` commands, the pedestrian-future positions
fed to the planner, and per-decision constraint-activation / solver stats from
`diagnostics()`). Assertions:

- **Factor A bites:** A-on vs A-off produce **different predicted-future pedestrian
  positions** (non-zero forward displacement vs. frozen-at-current) wherever pedestrians have
  non-zero velocity, and this changes at least some first-step commands. Where all
  pedestrians are static, A-on and A-off traces should coincide (a correctness sanity check
  on the freeze path).
- **Factor B bites:** B-on evaluates a **non-empty active pedestrian constraint** (non-zero
  constraint evaluations in `_optimizer_constraints`) and carries soft pedestrian weight
  `0.0`; B-off returns **no** hard constraint and carries `W_soft = 4.5` with a non-zero soft
  pedestrian cost. Traces must differ where a pedestrian is within margin.
- **Purity:** toggling A must not change constraint machinery selection, and toggling B must
  not change the prediction mode — the two knobs are orthogonal.

### 6.2 B-OFF functionality check (the maintainer's load-bearing condition)
Verify that the B-OFF arms (A00, A10) are **functional planners**, not degenerate ones:
non-trivial completion on the smoke scenarios, non-degenerate command statistics
(`nonzero_command_count > 0`, `mean_abs_linear > 0`, `fallback_stop_count` not saturated),
and demonstrable soft pedestrian avoidance (min-clearance distribution not collapsed to
systematic collision). **If a B-OFF arm collides trivially or stalls everywhere, the B
contrast is invalid and the campaign is blocked** until `W_soft` is set to a functional
shared value (re-preregistered before results are seen).

---

## 7. Config landing and evidence registry

The 2x2 campaign config (four arm keys + the shared base config + the pinned scenario matrix
and `paper_eval_s30` seed set) lands with its **sha256 in the evidence registry**, plus a
static preflight/identity test that asserts: config loads through `load_campaign_config`;
scenario-matrix hash matches the preregistered value; `paper_eval_s30` resolves exactly the
30 seeds `111..140`; the four arm configs differ **only** in `prediction_enabled`,
`hard_pedestrian_constraints_enabled`, and the implied soft pedestrian weight; and the
uncertainty envelope is disabled on every arm. No campaign submission, Slurm/GPU execution,
or row archiving occurs under this pre-registration.

---

## 8. Open implementation preconditions (binding; design/impl may start now, compute is gated)

1. **Factor A toggle does not exist.** Add `prediction_enabled` (or
   `pedestrian_prediction_mode: {constant_velocity | frozen}`) that, when off, zeroes the
   pedestrian velocity used by **both** `NMPCSocialPlannerAdapter._predict_pedestrians` (soft
   path) and `ConstantVelocityPedestrianPredictor.predict` (hard path). Minimal change; no
   optimizer change.
2. **Factor B toggle does not exist.** The shipping adapter always adds the hard constraint
   and hardcodes `pedestrian_clearance_weight = 0.0` in `_to_nmpc_config`. Add
   `hard_pedestrian_constraints_enabled` that, when off, returns `()` from
   `_optimizer_constraints` **and** passes the positive shared `W_soft` through to the cost.
   Expose `pedestrian_clearance_weight` in `build_prediction_mpc_config` converters.
3. **Single unified adapter/config for all four arms.** Preferred: extend
   `PredictionMPCConfig` with the two booleans + the `W_soft` passthrough so all four arms are
   one adapter with matched weights (capability parity by construction). This keeps §2.1
   parity a tautology.
4. **Arm YAMLs + preflight identity test** under `configs/` per §7, registered with sha256.
5. **Fidelity smoke module** (CPU) implementing §6 on the two named scenarios.
6. **Dependency coupling:** confirmatory analysis consumes **#5351** (hierarchical paired
   inference); parity is cross-checked against the **#5353** capability matrix. Both are open;
   design and arm implementation can proceed now, compute goes through the ops queue and
   respects the #4826-class per-arm GPU lifecycle gate.

---

## 9. Relation to #4830 (shared conventions, disjoint scope)

**Shared conventions.** #4830 is the already-registered paired-factorial measurement vehicle
(planner x {wrapper_off, wrapper_on}, paired-report builder #4598, per-cell on/off deltas
with uncertainty). This design reuses its conventions: paired factorial with per-cell deltas
and uncertainty; hierarchical paired inference (#5351); scenario-stratum-respecting
resampling; and the same evidence-registry + sha256 config-hash discipline.

**Disjoint scope (must not duplicate).** #4830 varies an **external safety-wrapper** on/off
across a **multi-planner roster** — the co-design-gain leg (does wrapping *any* planner
help). #5355 **fixes one base planner** and varies **two internal mechanisms** (prediction,
constraint-handling), matched-by-construction. #4830 varies the planner and holds the
mechanism family fixed; #5355 holds the planner fixed and varies the mechanisms. **No shared
arms, no shared config, no shared rows.** #5355 does not measure the safety-wrapper leg, and
#4830 does not absorb the prediction×constraint factorial. The two pre-registrations are
coordinated (same pairing/analysis conventions) but scope-disjoint.

---

## 10. Out of scope

No changes to existing planner arms or frozen release data; no claim promotion (the result
feeds interpretation regardless of direction — a null on Factor A is as informative as a
positive one); no GPU submission under this note.
