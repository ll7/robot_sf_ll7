# Collision Causality, Online Risk, and Scenario Discovery

> **Status:** research synthesis and implementation roadmap, 2026-07-12.
> **Evidence status:** design-stage / diagnostic-only. No collision cause, collision probability,
> planner ranking, or Chapter 7 claim is established by this document.
> **Claim boundary:** the proposed collision diagnosis attributes causal contribution only under an
> explicit simulator and counterfactual model. It does not determine legal, moral, or human blame.
> The proposed online estimator reports conditional model risk, not a safety guarantee or real-world
> crash probability.
> **Confidence:** about 0.88 that the architecture is a sound next research program. The main
> uncertainties are counterfactual replay fidelity, availability of action-decision traces, and the
> number of independent seeds required to make case selection statistically stable.

This note defines one connected research program: explain *why* a collision occurred, estimate
collision risk before contact for each candidate action, and use those explanations and risk traces
to select reproducible worked examples for dissertation Chapter 7.

Live tracking is owned by [epic #5440](https://github.com/ll7/robot_sf_ll7/issues/5440).

Related local foundations:

- [failure-mechanism classifier](issue_2012_failure_mechanism_classifier.md) and
  [taxonomy](issue_2220_failure_mechanism_taxonomy.md);
- [counterfactual mechanism taxonomy](issue_2547_counterfactual_mechanism_taxonomy.md) and the
  `robot_sf/benchmark/counterfactual_pair.py` pair evaluator;
- `robot_sf/benchmark/critical_intervals.py` for event-centred trace windows;
- [rare-event probability method boundary](issue_3292_rare_event_probability_plan.md);
- [adversarial generation roadmap](issue_2468_adversarial_generation_roadmap.md);
- `scripts/tools/analyze_scenario_seed_sensitivity.py` and closed issues
  [#1608](https://github.com/ll7/robot_sf_ll7/issues/1608) and
  [#1609](https://github.com/ll7/robot_sf_ll7/issues/1609);
- open learned-risk execution [#1472](https://github.com/ll7/robot_sf_ll7/issues/1472), oracle-gap
  analysis [#5302](https://github.com/ll7/robot_sf_ll7/issues/5302), transfer analysis
  [#5303](https://github.com/ll7/robot_sf_ll7/issues/5303), chance-constrained model predictive
  control [#5307](https://github.com/ll7/robot_sf_ll7/issues/5307), quality-diversity generation
  [#5308](https://github.com/ll7/robot_sf_ll7/issues/5308), and hierarchical release statistics
  [#5351](https://github.com/ll7/robot_sf_ll7/issues/5351).

## 1. Questions and review method

### Research questions

1. **Collision causality:** How can a Robot SF trace support a reproducible, programmatic account of
   the collision mechanism, the earliest avoidable unsafe robot action, other contributing actors or
   environment conditions, and uncertainty about that attribution?
2. **Online collision risk:** How should Robot SF estimate, at each decision point and for each
   candidate robot action or trajectory, the probability of contact within a declared horizon while
   preserving calibration, latency, and model-provenance boundaries?
3. **Interesting worked examples:** How can campaigns identify seed-dependent outcome flips and
   planner upsets without circular planner-strength definitions, multiple-comparison cherry-picking,
   or reliance on broken/degraded evidence rows?

The questions are feasible because the repository already owns trace, event, counterfactual,
critical-interval, prediction, and campaign-analysis primitives. They are scientifically relevant
to planner diagnosis and Chapter 7, and implementable as separate bounded tasks. They are not yet
answerable from existing aggregate results alone: causal attribution needs controlled replay,
online probability needs a declared conditional target and calibration set, and scenario selection
needs more than three seeds for confirmatory claims.

### Method

This is a scoping review and design synthesis, not a PRISMA systematic review or meta-analysis.
Searches were run on 2026-07-12 over OpenAlex and Crossref using 24 query variants spanning causal
failure explanation, accident responsibility, cyber-physical-system debugging, collision-risk
assessment, chance constraints, reachability, trajectory forecasting, adaptive stress testing,
scenario generation, random-seed sensitivity, and quality-diversity search. OpenAlex returned 514
deduplicated candidate records from 335,281 broad indexed matches. Exact-title and DOI checks in
Crossref produced a 30-source primary corpus. Broad hit counts reflect noisy discovery queries and
must not be interpreted as screened studies.

All 30 included records had title, author, year, venue, and DOI cross-checked between OpenAlex and
Crossref. Full text was inspected for six pivotal methods: CPSDebug, Responsibility-Sensitive Safety
(RSS), Dynamic Risk-Aware MPPI, online reachability verification, the black-box validation survey,
and quality-diversity scenario generation. Local full text was also inspected for adaptive stress
testing and social-navigation complexity. Remaining claims are restricted to verified metadata and
available abstract-level descriptions. The source table in Section 7 records this boundary.
Retraction/correction status was not checked against a dedicated retraction database; that check
remains mandatory during dissertation source intake.

Inclusion prioritized original method papers, peer-reviewed surveys, and directly applicable
robotics or cyber-physical-system work from 2010-2026, plus older foundational work. Blog posts and
vendor descriptions were excluded. The synthesis also inspected current `robot_sf_ll7` code,
context notes, and live GitHub issues. No human-subject data were collected; institutional review is
not applicable to this design note.

## 2. What exists locally, and what is missing

| Existing surface | What it already proves | Missing capability |
| --- | --- | --- |
| `failure_mechanism_classifier.py` | Conservative rule-based outcome labels over matched fixed/long-horizon rows | No action-level cause, causal graph, or branching replay |
| `failure_mechanism_taxonomy.py` | Shared mechanism vocabulary and confidence classes | No test that a labelled mechanism caused contact |
| `critical_intervals.py` | Closest approach, time-to-collision (TTC), braking, collision/near-miss, recovery, and mode-switch windows | No last-avoidable-state or first-unsafe-action computation |
| `counterfactual_pair.py` and #2924 | Tests whether a predeclared intervention activated a mechanism and moved an outcome in the expected direction | One pair is diagnostic, not an actual-cause proof; it does not branch from an identical mid-episode state |
| Event ledger and collision reconciliation | Typed event and collision semantics | Planner observations, predictions, candidates, scores, guards, and applied commands are not yet joined into one causal trace contract |
| `scenario_flakiness.py` and seed-sensitivity script | Flags outcome variation and hard/easy seeds | Current threshold/range rules are exploratory and do not quantify posterior uncertainty or multiplicity |
| Scenario-criticality and adversarial search | Finds severe or boundary cases | Failure discovery is not probability estimation; severity alone does not give diverse, explanatory cases |
| Issue #1472 learned-risk contract | Requires Brier score, calibration, AUROC/AUPRC, false-negative rate, and hard-guard precedence | The durable trace campaign and shared action-conditioned online-risk API remain absent |
| Issue #3292 rare-event plan | Defines probability language, target distributions, denominators, and direct Monte Carlo for episode-level failure frequency | It does not estimate short-horizon conditional risk online for a candidate robot action |
| Issues #5302, #5303, #5308, and #5351 | Oracle regret, cross-planner transfer, quality-diversity generation, and hierarchical statistics | No integrated selector for Chapter 7 case capsules |

The repository therefore needs integration, not a second taxonomy or another aggregate score.

## 3. Programmatic collision root-cause analysis

### 3.1 Replace “who is at fault?” with four distinct outputs

Accident analysis, formal responsibility models, and ethics literature do not support deriving legal
or moral fault from simulator geometry alone. Structural causal models formalize actual causes and
degrees of responsibility under explicit counterfactual assumptions
([Halpern & Pearl, 2005](https://doi.org/10.1093/bjps/axi147);
[Chockler & Halpern, 2004](https://doi.org/10.1613/jair.1391)). Systems-theoretic accident analysis
instead asks how inadequate control and interacting constraints produced the loss
([Leveson, 2003](https://doi.org/10.1016/S0925-7535(03)00047-X)). Ethical liability remains a
separate normative question ([Hevelke & Nida-Rümelin, 2015](https://doi.org/10.1007/s11948-014-9565-5)).

Every Robot SF incident report should therefore separate:

1. **Observed reconstruction:** actors, states, geometry, commands, events, and contact facts.
2. **Proximate mechanism:** for example late braking, oscillatory avoidance, prediction miss, route
   trap, pedestrian reciprocity mismatch, or simulator/metric artifact.
3. **Causal contribution under model:** whether a declared intervention would have prevented
   contact while holding specified variables or policies fixed.
4. **Normative fault:** always `not_assessed` in Robot SF artifacts.

RSS is useful as an example of model-scoped responsibility: it defines a dangerous situation and a
proper response under explicit behavioural assumptions, rather than inferring responsibility from
contact alone ([Shalev-Shwartz et al., 2017](https://doi.org/10.48550/arXiv.1708.06374)). Its road
rules cannot simply be transplanted into pedestrian-rich social navigation, but its separation of
danger, response duty, and collision is valuable.

### 3.2 A layered causal-analysis pipeline

The recommended system has six layers.

**A. Reconstruct one authoritative timeline.** Join the typed event ledger to robot and pedestrian
states, map geometry, observations, forecast distributions, candidate trajectories/actions,
candidate scores, selected command, guard/arbitration decisions, feasible actuator command, and
contact geometry. Each field must record source and availability. Missing planner-internal fields
produce `unknown`, not reconstructed guesses.

**B. Build a temporal causal graph.** Nodes should cover scenario initialisation, observation,
prediction, candidate generation, scoring/selection, safety guard, command conversion, actuation,
robot dynamics, pedestrian response, and collision metrics. Edges describe the implemented data
flow, not post-hoc statistical correlation. CPSDebug shows that mined expected properties, first
violation times, and component mapping can localize failure propagation in cyber-physical models
([Bartocci et al., 2021](https://doi.org/10.1007/s10009-020-00599-4)); it is a useful localization
stage, but suspicious signals are not automatically actual causes.

**C. Locate four timestamps.** For each incident compute:

- `t_danger`: first declared safety-margin or temporal-logic robustness breach;
- `t_uca`: first *unsafe control action*, using four systems-theoretic forms: a needed action was
  absent, an unsafe action was provided, timing/order was unsafe, or an action continued/stopped for
  an unsafe duration;
- `t_inevitable`: first state from which no admissible robot action in the declared action set avoids
  contact under the declared response model;
- `t_contact`: first authoritative collision event.

The useful answer to “which action was false?” is the earliest avoidable unsafe control action, not
the final command before contact. If `t_inevitable <= t_uca`, the system must not assign a planner
action as the collision cause; contact was already unavoidable under that model.

**D. Branch from frozen states.** At candidate decision points between `t_danger` and
`t_inevitable`, restore the exact simulator snapshot and replace only the selected robot action with
admissible alternatives: emergency braking, bounded steering/velocity samples, other candidates
already generated by the planner, or a declared short-horizon oracle. Re-run with fixed simulator
random state. Causal testing similarly treats controlled interventions as the evidence needed to
distinguish association from causal effects ([Johnson et al., 2020](https://doi.org/10.1145/3377811.3380377)).

Because pedestrians react to the robot, a single frozen pedestrian trajectory is insufficient.
Report results under at least two clearly named response assumptions when possible: replayed
pedestrian motion and closed-loop pedestrian response. Do not average them into one verdict.

**E. Find minimal sufficient interventions.** If multiple changes are necessary, retain the minimal
intervention sets and interaction terms. A degree-of-responsibility or Shapley-style allocation can
be a secondary summary, but individual intervention outcomes and assumptions remain authoritative.
The analyser must abstain when multiple explanations are observationally equivalent, replay is
nondeterministic, or the action set excludes plausible avoidance actions.

**F. Emit a reviewable cause report.** The minimum report contains observed facts, the four
timestamps, unsafe-action class, intervention table, avoidable/unavoidable/unknown verdict,
contributing components, competing explanations, confidence, assumptions, missing fields, and
`normative_fault: not_assessed`.

### 3.3 Initial mechanism classes

The report should reuse current taxonomy where possible and add cause-location classes only where
the evidence path differs:

- observation/perception omission or delay;
- prediction distribution miss or miscalibration;
- candidate-generation omission;
- candidate scoring/selection error;
- guard/arbitration omission or unsafe override;
- command conversion, actuation, or kinematic infeasibility;
- route/topology trap;
- pedestrian interaction or reciprocity mismatch;
- scenario infeasibility / collision already unavoidable;
- simulator, logging, or metric artifact;
- unknown or interacting causes.

### 3.4 Validation and stop rule

Validation begins with injected-fault fixtures whose responsible component and activation time are
known. Report class precision/recall, temporal localization error, avoidability accuracy,
counterfactual replay determinism, abstention coverage, and reviewer agreement. Then use ambiguous
multi-actor fixtures where the correct result is `unknown` or an interaction set.

Stop and revise if identical snapshot replays are not deterministic enough to preserve the baseline
outcome, if the known injected cause is outside the top reported explanation in more than 10% of
simple fixtures, or if the analyser emits high-confidence single-cause labels for deliberately
ambiguous fixtures. No real-episode causal claim is promoted before those gates pass.

## 4. Online collision-probability estimation

### 4.1 Define the target before choosing a model

At decision time `t`, the core estimand should be:

```text
P(contact in (t, t + H] |
  observed history h_t,
  candidate robot action/trajectory u[t:t+H],
  pedestrian prediction model M,
  geometry and uncertainty version V)
```

It is horizon-specific, candidate-action-conditioned, and model-versioned. Outputs should include
joint probability of any contact, per-actor marginal contributions, a first-passage-time
distribution, uncertainty or out-of-distribution status, and compute latency. Per-actor marginals
must not be summed as if agents were independent. The joint event should be computed from joint
samples where available; a union bound may be reported separately as a conservative bound.

Classical velocity obstacles and reciprocal collision avoidance identify unsafe velocity sets, not
calibrated probabilities ([Fiorini & Shiller, 1998](https://doi.org/10.1177/027836499801700706);
[van den Berg et al., 2011](https://doi.org/10.1007/978-3-642-19457-3_1)). Reachability analysis can
provide a set-based verification result under modelled behaviour and uncertainty, also not a
probability ([Althoff & Dolan, 2014](https://doi.org/10.1109/TRO.2014.2312453)). These distinctions
must remain visible in the API.

### 4.2 Implement an evidence ladder

1. **Deterministic signals:** minimum separation, TTC, velocity-obstacle membership, and reachable
   collision flag. These are interpretable baselines and guard signals, labelled non-probabilistic.
2. **Analytic or Monte Carlo constant-velocity risk:** roll out exact robot footprint against
   explicitly parameterized pedestrian state noise. This provides the first auditable probability
   baseline, though its behavioural model is weak.
3. **Multimodal forecast Monte Carlo:** sample joint pedestrian futures, roll each candidate robot
   trajectory, and estimate first contact. Trajectron++ is representative of dynamically feasible,
   multimodal heterogeneous-agent forecasting
   ([Salzmann et al., 2020](https://doi.org/10.1007/978-3-030-58523-5_40)). Risk-sensitive control
   and Dynamic Risk-Aware MPPI demonstrate how multimodal forecasts and Monte Carlo joint collision
   probabilities can enter action selection
   ([Nishimura et al., 2020](https://doi.org/10.1109/IROS45743.2020.9341469);
   [Kim et al., 2025](https://doi.org/10.1109/IROS60139.2025.11246822)).
4. **Calibrated learned risk:** consume trace/history features and prediction outputs, but remain an
   auxiliary cost or warning until calibration and distribution-shift gates pass. This is the role
   of existing issue #1472, not a duplicate model-training issue.
5. **Conformal or reachable upper bound / shield:** provide a separate guarantee-like output under
   explicit assumptions. Adaptive conformal prediction can wrap dynamic-agent uncertainty for
   safe partially observed planning ([Sheng et al., 2024](https://doi.org/10.1109/LRA.2024.3468092)).
   Chance constraints similarly bound collision risk under a declared uncertainty model
   ([Blackmore et al., 2011](https://doi.org/10.1109/TRO.2011.2161160)). Neither should be labelled
   an empirical calibrated probability without separate validation.

### 4.3 Calibration and runtime contract

Compare estimators on identical histories, forecast samples, candidate actions, geometry, and
horizons. Required metrics are Brier score, log loss, reliability curves, calibration error with
bin counts exposed, area under the precision-recall curve for rare contacts, false-negative rate at
predeclared operating points, time-to-warning, action sensitivity, and horizon monotonicity. AUROC
alone is insufficient for rare events. Report calibration by scenario family, planner, density,
prediction model, and horizon, plus out-of-distribution abstention.

Runtime is part of validity: record p50/p95/p99 latency, deadline misses, forecast sample count, and
candidate count. A risk estimator that misses the local-planner control deadline is offline analysis,
not an online system. A 100 ms target is the default research gate unless the consuming planner has
a stricter measured budget; the gate must not be relaxed after seeing results.

Case-control sampling, adversarial enrichment, and importance sampling alter prevalence. Calibrate
only after reweighting to a declared target distribution and retain target/proposal density
provenance. The black-box validation literature distinguishes falsification, most-likely failure
search, and probability estimation; finding a collision does not estimate its probability
([Corso et al., 2021](https://doi.org/10.1613/jair.1.12716)). Existing issue #3292 remains the
probability-language gate for episode distributions.

Hard collision guards remain authoritative until a separate safety argument proves otherwise. The
learned score, calibrated model probability, reachable-set warning, and formal bound are distinct
fields, never silently collapsed into `risk`.

## 5. Finding interesting Chapter 7 scenarios

### 5.1 Evidence gates before interestingness

A row is eligible only when it has exact typed collision/event semantics, native non-degraded
execution, matching scenario/seed/config provenance, stable planner identity, sufficient trace
coverage, and a frozen code/config SHA. Release 0.0.2 collision-derived fields remain ineligible
while [#5097](https://github.com/ll7/robot_sf_ll7/issues/5097) is unresolved. A new commit invalidates
the selection evidence for that case.

### 5.2 Separate interpretable discovery signals

Do not create one hidden weighted “interestingness score.” Keep a component vector, apply evidence
gates first, then use Pareto and diversity selection.

**Seed flip.** For one planner and scenario, success/collision class changes across seeds. Estimate a
beta-binomial or hierarchical posterior for success and flag high posterior uncertainty/entropy or
substantial between-seed variation. Three seeds can discover a candidate but cannot establish a
stable seed-sensitive scenario. Random-seed literature shows that small samples and selective
reporting can materially distort conclusions
([Colas et al., 2018](https://doi.org/10.48550/arXiv.1806.08295);
[Henderson et al., 2018](https://doi.org/10.1609/aaai.v32i1.11694);
[Agarwal et al., 2021](https://doi.org/10.48550/arXiv.2108.13264)).

**Planner inversion or upset.** Define planner strength from held-out scenarios or leave-one-family-
out estimates, then find a matched scenario+seed where a high-skill planner fails and a lower-skill
planner succeeds. The strength estimate must exclude the candidate family/cell to avoid selecting a
case using the same outcome that defines “strong.” Report the constraints-first outcome difference,
held-out skill gap, uncertainty, and oracle regret. Bradley-Terry/Elo-style paired skill estimates
are suitable summaries if the raw paired outcomes remain available.

**Disagreement and regret.** Use cross-planner outcome entropy, per-episode oracle improvement, and
family-specific rank reversals. This directly complements #5302 and #5303.

**Boundary and mechanism value.** Prefer reproducible cases near a temporal-logic/clearance boundary,
with a non-common mechanism, or with a clear causal divergence. Adaptive stress testing finds
likely failure trajectories under an explicit stochastic disturbance model
([Koren et al., 2018](https://doi.org/10.1109/IVS.2018.8500400)); Scenic, VerifAI, and AsFault
provide complementary programmatic or search-based scenario generation
([Fremont et al., 2019](https://doi.org/10.1145/3314221.3314633);
[Dreossi et al., 2019](https://doi.org/10.1007/978-3-030-25540-4_25);
[Gambi et al., 2019](https://doi.org/10.1145/3293882.3330566)). These methods find candidates, but
controlled replay is still needed to explain *why* planners diverge.

**Scenario diversity.** Density, narrow geometry, policy mix, directionality, and kinematics are
meaningful social-navigation descriptors
([Stratton et al., 2024](https://doi.org/10.1109/LRA.2024.3502060)). Quality-diversity search can fill
cells in a declared behaviour-descriptor archive rather than returning only the severest failure
([Fontaine & Nikolaidis, 2021](https://doi.org/10.15607/RSS.2021.XVII.036)). DynaBARN and
scenario-based vehicle-safety work reinforce the need to characterize dynamic scenario difficulty
and scenario abstraction explicitly
([Nair et al., 2022](https://doi.org/10.1109/SSRR56537.2022.10018758);
[Menzel et al., 2018](https://doi.org/10.1109/IVS.2018.8500406);
[Riedmaier et al., 2020](https://doi.org/10.1109/ACCESS.2020.2993730)).

### 5.3 Statistical and experimental safeguards

- Use common initial random numbers for matched planner comparisons, but acknowledge that reactive
  pedestrians diverge after planner actions diverge.
- Use hierarchical paired bootstrap/modeling over scenario family, scenario, seed, and planner;
  align confirmatory claims and multiplicity handling with #5351.
- Split discovery and confirmation seeds. The confirmation set must not be used to tune thresholds,
  captions, or case choice.
- Report all candidates passing the predeclared gate before selecting the final diverse subset.
- Treat old #1608 hard/easy range thresholds as triage signals only.
- Include negative controls: cases where seed changes trajectories but not outcomes, and apparent
  planner inversions that disappear after held-out strength estimation.

### 5.4 Chapter 7 case-capsule set

The first target set is four to six compact capsules, not dozens of disconnected screenshots:

1. same planner and scenario, hard seed versus easy seed;
2. same scenario and seed, expected strong planner failure versus weak planner success;
3. near-identical approach with different first unsafe action and causal branch outcome;
4. near miss where online risk rises early enough to distinguish two candidate actions;
5. unexpected recovery where an aggregate metric obscures the critical interval;
6. optional metric-disagreement or multi-actor ambiguity case whose correct causal output is
   abstention.

Each capsule should consume the separate Chapter 7 visualization contract: static map and metre
scale, complete actor trajectories, shared marked timestamps, actor footprints at those timestamps,
critical-interval/risk panels, concise “why this frame” annotations, and paired planners or seeds on
identical axes. The causal report supplies the explanation; the visualization does not infer it.

## 6. Issue decomposition and execution order

One epic should own the research question and links; bounded child issues should own implementation.
The intended dependency order is:

1. [#5441 collision causal-graph and report contract](https://github.com/ll7/robot_sf_ll7/issues/5441);
2. [#5442 mid-episode snapshot/branching replay and last-avoidable-action analysis](https://github.com/ll7/robot_sf_ll7/issues/5442);
3. [#5443 attribution validation on injected and ambiguous faults](https://github.com/ll7/robot_sf_ll7/issues/5443);
4. [#5444 action-conditioned online-risk API plus deterministic/Monte Carlo baselines](https://github.com/ll7/robot_sf_ll7/issues/5444);
5. [#5445 matched calibration comparison](https://github.com/ll7/robot_sf_ll7/issues/5445), integrating rather than duplicating #1472 and #5307;
6. [#5446 seed-flip/planner-inversion miner](https://github.com/ll7/robot_sf_ll7/issues/5446), integrating #5302/#5303/#5308/#5351;
7. [#5447 Chapter 7 case-capsule synthesis](https://github.com/ll7/robot_sf_ll7/issues/5447) after pinned-SHA evidence exists.

Exactly one writer should own each implementation branch. External validation begins only after the
head SHA is frozen. A validation issue may not repair and approve the same head. No new outcome
campaign should be launched merely to populate figures before the current long campaign and the
required trace/provenance contracts are resolved.

## 7. Literature verification matrix

`FT` means full text inspected in this review; `MA` means metadata and abstract verified. Evidence
levels are methodological relevance to this design, not a quality score for the publication.

| Theme | Source | Access | Contribution and boundary |
| --- | --- | --- | --- |
| Causality | [Halpern & Pearl (2005)](https://doi.org/10.1093/bjps/axi147) | MA | Structural actual-cause semantics; conclusions depend on chosen model and contingencies |
| Causality | [Chockler & Halpern (2004)](https://doi.org/10.1613/jair.1391) | MA | Degree of responsibility/blame formalism; normative blame needs additional epistemic inputs |
| Accident analysis | [Leveson (2003)](https://doi.org/10.1016/S0925-7535(03)00047-X) | MA | Systems-theoretic unsafe-control framing; not an automated trace algorithm |
| Failure localization | [Bartocci et al. (2021)](https://doi.org/10.1007/s10009-020-00599-4) | FT | Passing/failing traces, mined properties, first violations, component propagation; suspicious cause is not actual cause |
| Causal testing | [Johnson et al. (2020)](https://doi.org/10.1145/3377811.3380377) | MA | Intervention-based software testing; requires controllable variables and valid oracle |
| Responsibility model | [Shalev-Shwartz et al. (2017)](https://doi.org/10.48550/arXiv.1708.06374) | FT | Dangerous situation and proper-response separation; automotive assumptions are not social-navigation law |
| Ethics | [Hevelke & Nida-Rümelin (2015)](https://doi.org/10.1007/s11948-014-9565-5) | MA | Shows crash responsibility is normative and distributed; not computable from Robot SF alone |
| Collision geometry | [Fiorini & Shiller (1998)](https://doi.org/10.1177/027836499801700706) | MA | Velocity-obstacle unsafe set; deterministic, not calibrated probability |
| Reciprocal avoidance | [van den Berg et al. (2011)](https://doi.org/10.1007/978-3-642-19457-3_1) | MA | Reciprocal multi-agent avoidance; modelling assumption, not observed responsibility |
| Chance constraints | [Blackmore et al. (2011)](https://doi.org/10.1109/TRO.2011.2161160) | MA | Probabilistic constraints under declared uncertainty; bound quality follows model assumptions |
| Reachability | [Althoff & Dolan (2014)](https://doi.org/10.1109/TRO.2014.2312453) | FT | Online set-based verification; guarantee under reachable-set model, not empirical probability |
| Risk survey | [Lefèvre et al. (2014)](https://doi.org/10.1186/s40648-014-0001-z) | MA | Connects motion prediction and risk assessment; intelligent-vehicle scope |
| Criticality | [Lin & Althoff (2023)](https://doi.org/10.1109/IV55152.2023.10186673) | MA | Library of criticality measures; measures are signals, not causal verdicts |
| Forecasting | [Salzmann et al. (2020)](https://doi.org/10.1007/978-3-030-58523-5_40) | MA | Multimodal dynamically feasible trajectory samples; forecast calibration still needs validation |
| Uncertainty bounds | [Sheng et al. (2024)](https://doi.org/10.1109/LRA.2024.3468092) | MA | Adaptive conformal prediction for dynamic-agent planning; coverage assumptions must be exposed |
| Online risk control | [Kim et al. (2025)](https://doi.org/10.1109/IROS60139.2025.11246822) | FT | Efficient Monte Carlo joint collision probability in MPPI; independence approximations need testing |
| Risk-sensitive control | [Nishimura et al. (2020)](https://doi.org/10.1109/IROS45743.2020.9341469) | MA | Multimodal human forecasts in action selection; transfer to Robot SF is untested |
| Scenario abstraction | [Menzel et al. (2018)](https://doi.org/10.1109/IVS.2018.8500406) | MA | Functional/logical/concrete scenario levels; vehicle-domain terminology needs adaptation |
| Scenario survey | [Riedmaier et al. (2020)](https://doi.org/10.1109/ACCESS.2020.2993730) | MA | Scenario-based safety-assessment landscape; broad rather than social-navigation-specific |
| Stress testing | [Koren et al. (2018)](https://doi.org/10.1109/IVS.2018.8500400) | FT | Likely failure-trajectory search under stochastic disturbances; search yield is not probability |
| Validation survey | [Corso et al. (2021)](https://doi.org/10.1613/jair.1.12716) | FT | Separates falsification, likely failure search, and probability estimation |
| Scenario language | [Fremont et al. (2019)](https://doi.org/10.1145/3314221.3314633) | MA | Probabilistic scene specification; does not choose scientifically interesting cases by itself |
| Formal analysis | [Dreossi et al. (2019)](https://doi.org/10.1007/978-3-030-25540-4_25) | MA | Formal design/analysis toolkit; system integration required |
| Search-based testing | [Gambi et al. (2019)](https://doi.org/10.1145/3293882.3330566) | MA | Procedural road testing; road-domain objectives need adaptation |
| Statistical reporting | [Agarwal et al. (2021)](https://doi.org/10.48550/arXiv.2108.13264) | MA | Distributional and interval-aware RL evaluation; does not prescribe scenario case selection |
| Seed power | [Colas et al. (2018)](https://doi.org/10.48550/arXiv.1806.08295) | MA | Power consequences of seed count; assumptions differ from nested scenario campaigns |
| Reproducibility | [Henderson et al. (2018)](https://doi.org/10.1609/aaai.v32i1.11694) | MA | Implementation and random-seed sensitivity; aggregate RL focus |
| Social-navigation complexity | [Stratton et al. (2024)](https://doi.org/10.1109/LRA.2024.3502060) | FT | Density, geometry, policy/directionality/kinematics descriptors; fixed-seed comparisons do not establish seed sensitivity |
| Dynamic benchmark | [Nair et al. (2022)](https://doi.org/10.1109/SSRR56537.2022.10018758) | MA | Dynamic-environment difficulty benchmark; not causal analysis |
| Quality diversity | [Fontaine & Nikolaidis (2021)](https://doi.org/10.15607/RSS.2021.XVII.036) | FT | Diverse failure archive over declared descriptors; diversity depends on descriptor choice |

Nine sources already have DOI-matched entries in the dissertation bibliography: van den Berg et
al., Lin and Althoff, Menzel et al., Riedmaier et al., Koren et al., Agarwal et al., Colas et al.,
Stratton et al., and Nair et al. The remaining sources require normal Zotero/BibLaTeX intake and DOI
verification before manuscript citation; this note deliberately does not modify the currently dirty,
Zotero-managed bibliography.

## 8. Synthesis, contradictions, and open gaps

Three evidence streams converge. Formal causality requires interventions and an explicit model;
systems safety requires looking earlier than contact at unsafe control and control constraints; and
trace-debugging methods can localize first violations but cannot alone prove actual cause. Together
they support the layered pipeline in Section 3.

Risk literature contains an important apparent contradiction: velocity-obstacle/reachability
methods offer crisp unsafe/safe outputs, while probabilistic forecasting and Monte Carlo methods
offer graded risk. The outputs are not competitors on one scale. One is a set or guarantee under a
model; the other is a calibrated conditional frequency target. Robot SF should expose both and test
their disagreements.

Scenario literature similarly separates severity optimization, likely-failure search, probability
estimation, and behavioural diversity. A scenario can be severe but common, rare but uninformative,
diverse but benign, or explanatory despite modest severity. This is why Chapter 7 selection needs a
component vector and diversity archive rather than one ranking.

The largest empirical gaps are:

1. no deterministic mid-episode snapshot/replay proof across interactive pedestrian models;
2. incomplete planner decision traces for observations, forecast samples, candidates, scores, and
   guard outcomes;
3. no injected-fault ground-truth suite for causal attribution;
4. no shared candidate-action collision-risk API or matched calibration corpus;
5. no statistically defensible, held-out definition of planner strength for upset mining;
6. insufficient confirmation seeds for Chapter 7 seed-flip claims;
7. model-dependent pedestrian counterfactuals with no accepted reference policy.

These gaps are the reason for the issue split. They are not grounds to claim that the methods already
work.

## 9. Research integrity and responsible use

The program has low dual-use risk but meaningful misinterpretation risk. “Fault” labels could be
misused as claims about human pedestrians, planner developers, or real-world liability. Artifacts
must therefore use model-scoped language, preserve competing explanations, expose abstention, and
set `normative_fault: not_assessed`. No simulated collision probability may be described as
real-world operational risk without an explicit external-validity study.

**AI disclosure:** This research note was produced with AI-assisted literature discovery, metadata
verification, source triage, repository inspection, synthesis, and drafting. Source metadata were
cross-checked programmatically; pivotal full texts were inspected; no claim of exhaustive review is
made. Human maintainer review is required before paper-facing use.
