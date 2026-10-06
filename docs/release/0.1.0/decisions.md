# Release 0.1.0: decision ledger

New decisions are added here when they are made, each with its enforcing test or "none yet".

This file records the decisions taken while planning benchmark release 0.1.0
(the milestone formerly called 0.0.9), starting on 2026-10-01. For each
decision it states the question, the choice, the reason, who decided, what was
rejected, the evidence, where it is implemented, and which test or gate
enforces it.

How to read the fields:

- **Decided by.** "author" means the repository author ruled. "orchestrator
  (delegated)" means the orchestrating agent decided under the author's
  standing delegation; the author reviews these afterwards. "proposed" means
  written down but not yet ruled on.
- **Evidence.** Issues and pull requests in this repository are written `#N`;
  dissertation repository items are written `diss#N`. Research notes are
  files under `docs/context/research_results/` in ll7/diss, each with a
  verification header saying which numbers were checked against primary
  sources.
- **Enforced by.** "none yet: X" names what should enforce it. Almost all
  0.1.0 decisions are plans; their tests come with the implementing PRs.

Related: the 0.0.8 ledger (`docs/release/0.0.8/decisions.md`), whose
disclosures (wall stand-off, doorway queue, 0.65 m/s, disc size, no steering,
no reverse, grid offset, TTC) are the starting point for this release.

Verification snapshot: GitHub issue/PR metadata and code checked on 2026-10-01. The evening author rulings are recorded explicitly. Measurements and literature numbers are source-attributed planning inputs; this records lane executed no new simulation. Evening author rulings apply where stated.

## Release identity

### D-001: The next release is named 0.1.0, not 0.0.9
- **Date:** 2026-10-01
- **Question:** The planned changes (pedestrian body, speed, wall law, contact,
  pedestrian-robot interaction, reverse driving, retrained learned and
  forecast models) make results incomparable with 0.0.x. Keep the milestone
  name 0.0.9?
- **Choice:** Name it 0.1.0. Release notes state that results are not
  comparable cell by cell with any 0.0.x release. It gets its own sealed
  evaluation seeds, its own full campaign and its own intake.
- **Reason:** Under version-numbering conventions a change that makes results
  incomparable belongs in the middle number; the name tells readers at once
  that this is a new benchmark generation, not a patch.
- **Decided by:** author (answer "0.1.0 (Recommended)", chat, 2026-10-01
  14:32), after saying on 14:19 "0.0.9 will be a new release with completly
  different behavior and probably result. We might change it to 0.1.0".
- **Alternatives:** keep 0.0.9.
- **Evidence:** milestone #11 renamed (description: "Renamed from 0.0.9 on
  2026-10-01 (author decision)"); #10074 comment 5933617449; 0.1.0 scope
  proposal.
- **Implemented in:** milestone rename. Many issue titles still say "0.0.9"
  (#10061, #10068, #10070, #10071, #10079).
- **Enforced by:** none yet: the release identity and notes check that the
  version is 0.1.0 and that a non-comparability sentence is present.
- **Reopen:** New material evidence or an explicit author ruling.

### D-002: 0.0.8 is finished first; 0.1.0 never delays it
- **Date:** 2026-10-01
- **Question:** How do the two releases relate?
- **Choice:** Finish 0.0.8 under its freeze rule. Write the thesis on 0.0.8; switch to 0.1.0 only if it ships before submission would happen anyway. The stop-or-continue check decides the switch. Parallel 0.1.0 work does not delay 0.0.8.
- **Reason:** "we finish 0.0.8 to make sure we have decent results, but 0.0.9
  will be a new release" (author, chat, 2026-10-01 14:19).
- **Decided by:** author.
- **Alternatives:** skip 0.0.8 and go straight to the new model.
- **Evidence:** Author records-lane instruction, 2026-10-01 evening; earlier chat 2026-10-01 14:19–14:25; 0.0.8 D-079.
- **Implemented in:** process.
- **Enforced by:** none yet: record the selected thesis release at the stop-or-continue check.
- **Reopen:** New material evidence or an explicit author ruling.

### D-003: Scope: a core set, cheap coupled items, everything else optional
- **Date:** 2026-10-01
- **Question:** The milestone currently holds 38 issues and 32 PRs (70 total, including closed items). Which must ship?
- **Choice:** (proposed scope; the author ruled on the decision items in D-008,
  D-012, D-013, but has not explicitly approved the lists)
  - **Core:** #10074 (with PRs #10073, #10075), #10061 merged with #10017,
    #10065, #10079, #10082, #10034 item B5, #10049, #10071, the #10000 gate
    run, and (by author decision) #10068.
  - **Coupled (cheap now, expensive later):** #10035, #10051, #10062 items
    1-2, #10028, #10037, #10027 items 1, 2 and 5, #10040, plus the forecast
    retrains #10033 and #10018 (D-013) and the loader fix #10083 (D-020).
  - **Independent:** hygiene, CI and seed-guard items (#10036, #10022, #10050,
    #10060, #10055, #10059, #10006, #10048, #10070, #10076, #10030, #10016);
    #10055 and #10059 are required before the sealed campaign.
  - **Deferred:** #10012 and research-tooling PRs (#9832, #9895, #9896, #9897,
    #9903, #9907).
- **Reason:** "Completely different behaviour" keeps growing unless the core
  is fixed early. The release ships when the pedestrian model passes the
  validation suite (D-007), not when every milestone item is done.
- **Decided by:** proposed by the orchestrator's scope triage (2026-10-01).
- **Alternatives:** ship every milestone item.
- **Evidence:** 0.1.0 scope proposal in decisions_0_1_0.md and gaps.md, private-ops review archive. The schedule estimate in the draft was not independently verified and is not an adopted deadline. GitHub milestone #11 inventory checked on 2026-10-01.
- **Implemented in:** milestone #11; native blocked-by links (#10071,
  #10033, #10018 blocked by the environment issues, including #10068).
- **Enforced by:** none yet: native dependency links; a scope list frozen
  in this file.
- **Reopen:** New material evidence or an explicit author ruling. Proposed choices remain open for author adoption.

### D-004: Pedestrian radius 0.28 m nominal, 0.25-0.30 m sensitivity, one parameter everywhere
- **Date:** 2026-10-01
- **Question:** Pedestrians are rigid discs of radius 0.40 m (collision,
  metrics, placement) and 0.35 m (force kernel). Real shoulder half-width is
  about 0.23-0.26 m. What radius?
- **Choice:** Nominal physical radius 0.28 m, sensitivity 0.25-0.30 m. One
  parameter is read by contact, the force kernel, the collision metric,
  placement, the occupancy grid and all planner configs. The behavioural
  avoidance margin is a separate, named force parameter, not an inflated body.
  Appendix wording: an engineering approximation informed by adult shoulder
  breadth, not a measured human radius; overlap defines contact.
- **Reason:** A disc whose overlap is called contact should follow body
  breadth. ANSUR II bideltoid breadth: men 51.04 cm (SD 3.25, n = 4,082) ->
  0.255 m; women 45.03 cm -> 0.225 m; men P95 56.7 cm -> 0.284 m (not
  re-checked). Helbing, Farkas and Vicsek 2000 use diameters 0.5-0.7 m;
  CrowdNav uses 0.3 m. Packing: non-overlapping discs reach 3.68 /m^2 at
  0.28 m but only 1.80 /m^2 at 0.40 m, below the 3.3 /m^2 supply density of
  Seyfried et al. 2009, so no wall law can reproduce narrow-door data with
  0.40 m discs (#10073). Radii fitted only to match flow hide missing
  mechanisms.
- **Decided by:** orchestrator proposal (#10074 comment, 10:25); author
  included it in the joint calibration (D-006) on 2026-10-01 14:32. The radius
  is formally "chosen from the results" within the range.
- **Alternatives:** keep 0.40 / 0.35 m (only defensible as an
  occupied-space envelope, and then the metric must be called envelope
  overlap); shrink until flow matches; 0.30 m as nominal (Zanlungo 2R = 0.6 m
  includes avoidance).
- **Evidence:** #10074 comments 5929476286 and 5929692628 (source check);
  diss research note `2026-10-01_chatgpt_pedestrian_radius_evidence.md`;
  #10073 calibration round.
- **Implemented in:** not yet (#10074 step 2, #10034 B5, #10079 D6).
- **Enforced by:** none yet: a test that fails if any consumer uses a
  literal 0.35 or 0.40 (scope acceptance 1).
- **Reopen:** New material evidence or an explicit author ruling.

### D-005: Desired speed N(1.3, 0.2) m/s per pedestrian with a separate cap
- **Date:** 2026-10-01
- **Question:** In 0.0.8 every crowd pedestrian walks at 0.65 m/s (spawn speed
  0.5 x 1.3), which is both target and cap. What speed model?
- **Choice:** Desired speed drawn per pedestrian from N(1.30, 0.20) m/s
  (sensitivity N(1.34, 0.26)); a separate cap of 1.3 x the pedestrian's own
  desired speed; relaxation time 0.5 s. No explicit group slowdown: check that
  groups end up 0.04-0.08 m/s slower per member, as measured by Moussaid et al.
  2010. 0.65 m/s survives only as a labelled `slow` tier. Fix the tier
  docstring citation (Moussaid 2009, not 2010).
- **Reason:** Free adult walking speed is 1.29 +- 0.19 m/s (Moussaid et al.
  2009, n = 40) and 1.34 m/s mean / 0.26 SD (Weidmann 1993). Moussaid et al.
  2010 also simulate with N(1.3, 0.2). A shared cap equal to the desired
  speed stops pedestrians catching up after being slowed. Dense-crowd
  slowdown is produced by the interaction forces, not by the desired speed.
- **Decided by:** author: asked "so this means we definitly should change to
  the literature-based option in the benchmark?" (13:14); the orchestrator
  answered "yes for 0.0.9 ... as part of the joint calibration"; the author
  then included speed N(1.3, 0.2) with a separate cap in the joint
  calibration (14:32).
- **Alternatives:** flip the existing `typical` tier alone (the cap
  is still tied to the desired speed); keep 0.65 m/s.
- **Evidence:** #10074 comment 5932171567; diss research note
  `2026-10-01_chatgpt_pedestrian_dynamics.md`; validation baseline (#10075).
- **Implemented in:** not yet; depends on #10083 (D-020).
- **Enforced by:** none yet: V1 in the validation suite (simulated mean
  1.29 +- 0.05 m/s, SD 0.15-0.25 on 30 dev seeds, as proposed in the scope).
- **Reopen:** New material evidence or an explicit author ruling.

### D-006: Joint calibration of radius, speed, wall law and contact, time-boxed to 5 working days
- **Date:** 2026-10-01
- **Question:** The wall stop happens where the wall force equals v0/tau. At
  v0 = 1.3 that balance doubles, and the radius sets the achievable density.
  Calibrate one parameter at a time?
- **Choice:** Calibrate together: radius (0.28 nominal, 0.25-0.30) x speed
  N(1.3, 0.2) with separate cap x wall law x contact rule. Time box: 5 working
  days. Hard gates: V1 (speed), V2 (apertures), overlap and wall penetration.
  V3-V8 flow comparisons are reported as differences with reasons.
- **Reason:** One-at-a-time changes were measured to fail (#10073: no wall
  law alone passes). The time box limits the largest schedule risk.
- **Decided by:** author; joint calibration, friction off, the 5-working-day time box and hard/soft gate split explicitly confirmed in the 2026-10-01 evening records-lane ruling.
- **Alternatives:** sequential calibration; hard gates on every V
  case.
- **Evidence:** #10074 comment 5933617449; scope proposal (risk 1).
- **Implemented in:** not yet.
- **Enforced by:** none yet: the validation suite's hard gates (D-007).
- **Reopen:** New material evidence or an explicit author ruling.

### D-007: The validation suite is the acceptance gate for the pedestrian model
- **Date:** 2026-10-01
- **Question:** How is the pedestrian model judged?
- **Choice:** One reusable validation suite reproduces published experiments
  with their own measurement definitions and prints: measured quantity |
  published value (source, table/figure) | simulated value | difference |
  known reason. Cases V1-V9 (free speed; single walker through apertures;
  narrow bottleneck; wide bottleneck; obstacle circumvention; head-on
  avoidance onset; lane formation; wall distance; robot passing as context).
  It runs on the unchanged 0.0.8 model (baseline for the appendix, rerunnable
  with one command at any sha, including the 0.0.8 freeze sha) and on each
  candidate. Where the simulator cannot reproduce a setup, the suite says so.
  0.1.0 ships when the hard gates pass. Thesis wording: "set from published
  measurements and checked against them", not "calibrated", unless a value is
  actually fitted.
- **Reason:** "It would be nice to ground the pedestrian simulation in some
  real data snapshots ... and then in the appendix we can explicitly describe
  what is different and why" (author, chat, 2026-10-01 08:26). The same table
  serves 0.0.8 (shows the gaps) and 0.1.0 (shows agreement).
- **Decided by:** author (grounding and appendix); orchestrator (suite as the
  ship gate, wording rule).
- **Alternatives:** a single "human clearance" number; copying
  published coefficients.
- **Evidence:** #10074 body; #10075; diss research notes of 2026-10-01; diss
  issue #3017 (consumer).
- **Implemented in:** draft #10075 (stacked on #10073); V2/V4-V6 and the
  single-radius plumbing incomplete.
- **Enforced by:** none yet: the suite's own hard-gate assertions, run in
  the release rehearsal.
- **Reopen:** New material evidence or an explicit author ruling.

### D-008: Contact rule: per-step velocity projection (replaces the capped contact force)
- **Date:** 2026-10-01
- **Question:** Nothing prevents pedestrian discs from overlapping. How should
  pedestrians be kept apart?
- **Choice:** (final) Each 0.1 s step, after the social-force velocity update
  and the speed cap, find every pedestrian pair and pedestrian-wall pair whose
  predicted end-of-step gap would be negative, and apply equal and opposite
  normal velocity changes that bring the predicted gap to zero (a one-step
  Maury-Venel projection). Repeat over the set for a few passes: the smallest
  count that meets the targets, starting at 3, at most 10. Store the corrected
  velocity. The robot is excluded (its collisions are measured on executed
  trajectories as before). Friction off (D-009). No substeps. Log residual
  overlap rather than iterating to zero. Report the step-time overhead on the
  release matrix; cost at most 15 %.
- **History:** At 14:32 the author chose a contact force: "Contact force sounds
  right, but what does 'plus friction' mean? Ideally no substepping if
  possible, couldn't we cap the maximum pushing apart force?" The orchestrator
  posted it as a Helbing-Farkas-Vicsek body force acting only during overlap,
  capped per step at about overlap/dt^2. Verification then showed the capped
  force fails, and at 15:00 the author approved the switch: "You made a good
  case to change it to a per-step velocity projection. If this is not too
  costly and gives us roughly the desired outcome, I am fine. I mean we don't
  need perfect overlap protection, just the obvious overlap should be
  minimized."
- **Reason:** In small verification simulations the capped force failed even
  for one head-on pair at 1.3 m/s: overlap reached 0.26 m and the pair bounced
  apart elastically, because the force acts only after overlap and ignores the
  approach speed. The separate five-person probe figures are chat-only and were not independently verifiable in the supplied artifacts, so they are not admitted here as measured evidence. The projection limits approach speed instead and requires no substeps. Maury and Venel 2011 (ESAIM M2AN 45(1),
  DOI 10.1051/m2an/2010035) define the projection; the external answer
  recommended it with 0.01 s substeps, which the author's no-substep
  requirement and the swept-motion argument make unnecessary.
- **Decided by:** author; the velocity-projection switch, minimise-obvious-overlap tolerance, 3–10 passes and cost at most 15% explicitly confirmed in the 2026-10-01 evening records-lane ruling. The robot exclusion and wall-pair implementation details are delegated design parameters.
- **Alternatives:** capped contact force (fails as above); HFV
  contact force with friction and substeps (stiff, needs h < 0.0365 s);
  post-step position separation (jerky, can push through walls); a
  collision-free speed model (a different model); stronger exponential
  repulsion (no guarantee).
- **Evidence:** #10074 comments 5933617449 (capped force) and 5934117678
  (switch); diss research note `2026-10-01_chatgpt_pedestrian_contact_model.md`
  (its header still describes the capped-force plan); diss#3030.
- **Implemented in:** not yet.
- **Enforced by:** none yet: the overlap acceptance (D-010) and a step-time
  overhead measurement in the validation suite.
- **Reopen:** New material evidence or an explicit author ruling.

### D-009: Sliding friction is off
- **Date:** 2026-10-01
- **Question:** Helbing's contact model has a tangential friction term. Include
  it?
- **Choice:** Off by default. Turn it on only if the joint calibration shows
  bottleneck flow needs it.
- **Reason:** Friction matters mainly for clogging and arching at panic
  densities, far above the benchmark's; it adds parameters without benefit.
- **Decided by:** author; friction off explicitly confirmed in the 2026-10-01 evening records-lane ruling.
- **Alternatives:** friction on by default.
- **Evidence:** #10074 comments 5933617449 and 5934117678.
- **Implemented in:** not yet.
- **Enforced by:** none yet: the manifest records the contact rule and
  friction setting (D-019).
- **Reopen:** New material evidence or an explicit author ruling.

### D-010: Overlap and wall-penetration acceptance (revised)
- **Date:** 2026-10-01
- **Question:** When is overlap "obvious"?
- **Choice:** Non-group pairs: no pair-step below 0.45 m between centres (or a
  disclosed handful); centre distance below 2r in under 0.1 % of non-group
  pair-steps. Group pairs reported separately, with the group target spacing
  above 2r (D-014). Wall penetration at most 2 cm, or disclosed. Baseline to
  beat (0.0.8 model, dev seeds, release matrix): non-group pair-steps below
  0.80 m 0.34 %, below 0.45 m 0.016 %; group pairs 55.8 % / 0.22 %; minimum
  0.028 m.
- **Reason:** Shoulder-to-shoulder walking (0.5-0.6 m between centres) is
  plausible; deep overlap is not.
- **Decided by:** orchestrator, implementing the author's tolerance ("just the
  obvious overlap should be minimized").
- **Alternatives:** Keep the original 1% overlap bar; require perfect non-overlap regardless of cost.
- **Supersedes:** the first bar of at most 1 % of non-group pair-steps below 2r
  (posted with the capped force; also in the scope proposal's acceptance 5).
- **Evidence:** #10074 comment 5934117678; overlap measurement lane (dev seeds
  1001-1030, 1,530 episodes).
- **Implemented in:** not yet (overlap metric in #10075).
- **Enforced by:** none yet: the suite's overlap gate.
- **Reopen:** New material evidence or an explicit author ruling.

### D-011: The obstacle (wall) law is refit after the radius, against behaviour targets
- **Date:** 2026-10-01
- **Question:** The released wall law stops lone pedestrians before openings
  up to about 3 m. How is it replaced?
- **Choice:** Fit an own wall law to behaviour targets (a lone walker passes
  every opening of 1.0 m and wider without stopping, with modest slowing;
  narrow and wide bottleneck flow as reported differences; wall clearance and
  obstacle circumvention, recording both centre and edge distance). Do not
  copy published coefficients. `legacy_v1` stays selectable and reproduces
  0.0.8 byte for byte.
- **Reason:** The vendored law is a shifted inverse-power potential and is not
  the gradient of its stated potential; published coefficients differ in
  distance origin and unit. #10073 showed no wall law passes at 0.40 m, so the
  wall law comes after the radius.
- **Decided by:** orchestrator (delegated); author for the doorway acceptance
  (#10074 comment 5932040814).
- **Alternatives:** copy Helbing or Moussaid coefficients; the 0.3x
  probe value.
- **Evidence:** #10061 comments (2026-10-01); #10073; diss note
  `2026-10-01_chatgpt_pedestrian_calibration_literature_routed.md`.
- **Implemented in:** draft #10073 (opt-in profiles; no profile accepted).
- **Enforced by:** none yet: V2 aperture gate; slice crossing on dev seeds.
- **Reopen:** New material evidence or an explicit author ruling.

### D-012: Pedestrian-robot activation scales with robot size; collisions are attributed
- **Date:** 2026-10-01
- **Question:** How should the pedestrian-robot interaction and collision attribution change?
- **Choice:** Activation = robot radius + pedestrian radius + an edge onset
  distance. Compare onsets giving 2.65 / 3.0 / 3.5 m on dev seeds as a
  sensitivity analysis; choose the default by passing clearance (reference
  0.83 +- 0.27 m, Vassallo et al. 2017) and absence of implausibly early
  swerving; report outcomes across the range. Add collision attribution to
  each row (robot moving or stationary, closing agent, approach direction).
- **Reason:** The force is radial and has no anticipatory side selection; attribution is absent. The current cutoff already scales with radii: `PedRobotForce.__call__` uses activation_threshold + force radius + robot radius (about 3.35 m with defaults). The 2.0 m centre-cutoff/0.65 m approach argument in #10065 is therefore not an established premise. Candidate activation distances must be reinterpreted against the effective cutoff and the distinct collision radius before choosing a default.
- **Decided by:** author ("this should probably increased. Here we need good arguments to increase this by how much",
  relayed in #10065 comment 5925990390, 2026-10-01 06:28 UTC);
  orchestrator for the summary "the bigger problem is the 2.0m reaction distance"
  and the sensitivity design. The summary is not part of the author's quote.
- **Alternatives:** a single literature onset value (none exists);
  fixing collinear encounters (author: rare).
- **Evidence:** #10065 comments 5925930814 and 5925990390.
- **Implemented in:** not yet; blocked by #10074.
- **Enforced by:** none yet: attribution fields present in every collision
  row; the sensitivity table.
- **Reopen:** New material evidence or an explicit author ruling.

### D-013: Forecast models are retrained in parallel with PPO
- **Date:** 2026-10-01
- **Question:** predictive MPPI (#10033) and prediction_planner (#10018)
  forecast models were trained on 0.65 m/s pedestrians.
- **Choice:** Retrain them in parallel with PPO on the frozen environment. If
  a retrain fails, fall back to a disclosure.
- **Reason:** Forecasts trained on 0.65 m/s pedestrians would be far out of
  distribution at about 1.3 m/s.
- **Decided by:** author ("Retrain in parallel (Recommended)", chat,
  2026-10-01 14:32).
- **Alternatives:** keep and disclose; drop both arms.
- **Evidence:** #10074 comment 5933617449.
- **Implemented in:** native links: #10033 and #10018 blocked by the
  environment issues.
- **Enforced by:** none yet: registry entries with SHA-256 for the new
  forecast models before any campaign uses them.
- **Reopen:** New material evidence or an explicit author ruling.

### D-014: Group spacing stays above 2r; the gaze-force singularity is fixed
- **Date:** 2026-10-01
- **Question:** Group pairs drive the overlap (55.8 % of group pair-steps
  below 0.80 m; minimum 0.028 m). Why, and what changes?
- **Choice:** Cap or regularise the gaze term near waypoints; implement or
  remove `fov_phi` (declared, unused); make cohesion vanish at or above contact
  distance; keep the group target spacing above 2r (0.56 m at r = 0.28 m). The
  projection (D-008) then guarantees the floor. Fix #10027 items 1, 2 and 5.
- **Reason:** Gaze magnitude scales with distance-to-centroid divided by
  distance to the pedestrian's own waypoint (epsilon 1e-6), so it blows up
  near a waypoint; the deepest event pulled a walking member into a standing
  one. Cohesion stays attractive below contact; intra-group repulsion acts only
  below 0.55 m with factor 1. Published pair spacing is 0.54-0.78 m
  (Moussaid et al. 2010, Table 1, tracked positions) and the Zanlungo 2017
  dyad mean is 0.71 m.
- **Decided by:** orchestrator (delegated).
- **Alternatives:** a full Moussaid-2010 group rewrite as a new
  calibration axis.
- **Evidence:** #10027 comment 5933982284; #10074 comment 5934117678.
- **Implemented in:** not yet.
- **Enforced by:** none yet: group-pair overlap reported by the suite;
  a unit test that gaze stays bounded at zero waypoint distance.
- **Reopen:** New material evidence or an explicit author ruling.

### D-015: Limited reverse driving is included
- **Date:** 2026-10-01
- **Question:** Include reverse driving (#10068) in 0.1.0 or defer it?
- **Choice:** Include. Add a separate reverse cap, versioned and recorded in
  the plant identity so 0.0.8 stays reproducible. Planners that sample
  commands get backward candidates with rear clearance checks. A dev-seed
  comparison runs reverse off vs 0.3 m/s vs 0.5 m/s; the release default is
  chosen from the measured trade-off (escapes gained vs collisions while
  reversing). The cap is a declared design choice, not a standards value. It
  must land before the PPO and forecast retrains.
- **Reason:** Backing up is the realistic escape from a nose-to-nose standoff;
  the published PPO was trained with reverse; a later change would force
  another retrain.
- **Decided by:** author: "Include" (chat, 2026-10-01 14:32), against the
  orchestrator's recommendation to defer (planner work across 5 samplers,
  about a week of critical-path risk). Earlier: "0.3 m/s sound good as well"
  (06:27).
- **Alternatives:** defer to a later release; 1.0 m/s cap.
- **Evidence:** #10068 body and comments; no verified standard gives a
  reverse-specific limit (ISO 13482:2014, ISO 3691-4:2023); one manufacturer
  example reverses at 0.30 m/s.
- **Implemented in:** not yet. #10064 (its prerequisite) is fixed by #10066.
- **Enforced by:** none yet: plant identity records the reverse cap; the
  comparison report.
- **Reopen:** New material evidence or an explicit author ruling.

### D-016: PPO is retrained after the environment freeze
- **Date:** 2026-10-01
- **Question:** The retrained 0.0.8 PPO still collides in 37 % of dev episodes
  (85 % with static geometry; variant B holds 2.0 m/s on 97-100 % of steps;
  the action std grows during training).
- **Choice:** Retrain on the frozen 0.1.0 environment and plant (reverse,
  fixed rasteriser, new pedestrians). Candidate changes, measured one at a
  time: speed cost near static geometry, bounded or annealed policy std,
  training on release geometry and spawn rules, a narrow-scenario curriculum.
  Recipe work may start on a provisional environment. Acceptance: on the
  240-episode dev matrix, PPO is not dominated by both orca and social_force
  on success and collisions together; collision count at most the maximum
  among the other admitted arms; registry entry with SHA-256 and a durable
  artifact. Pre-declared fallback: report PPO as dominated, rather than miss
  the date.
- **Reason:** A learned policy must see the final plant in training.
- **Decided by:** author ("more training for 0.0.9 could become interesting",
  2026-10-01 07:28); orchestrator for acceptance and ordering.
- **Alternatives:** reuse the 0.0.8 retrain.
- **Evidence:** #10071; #10077 evaluation table.
- **Implemented in:** not yet; #10071 blocked by #10082 and the environment
  issues.
- **Enforced by:** none yet: paired sign tests and the cap-signature slice
  in #10071's acceptance.
- **Reopen:** New material evidence or an explicit author ruling.

### D-017: The occupancy-grid rasteriser is fixed before the PPO retrain
- **Date:** 2026-10-01
- **Question:** The circle rasteriser shifts pedestrians half a cell.
- **Choice:** Test against cell centres; fix before #10071.
- **Reason:** A model trained on the shifted grid would face a distribution
  shift after the fix.
- **Decided by:** orchestrator (delegated).
- **Alternatives:** fix after the retrain.
- **Evidence:** #10082.
- **Implemented in:** not yet.
- **Enforced by:** none yet: centroid within 0.25 cell of the true position;
  circle and polygon rasterisers agree (both fail on base).
- **Reopen:** New material evidence or an explicit author ruling.

### D-018: Metrics measure from footprints; thresholds are reported as sensitivity ranges
- **Date:** 2026-10-01
- **Question:** Several metrics measure from the robot centre (#10079 D1-D6).
- **Choice:** Footprint clearance for collisions; disc-contact TTC with the
  closing speed only; edge-based or removed space compliance; clearance-based
  thresholds in the analysis tools; one radius source. Keep the 0.50 m
  near-miss gap as the SNQI input, but report near-miss 0.10 / 0.20 / 0.30 /
  0.50 m, comfort 0.30 / 0.50 / 0.80 / 1.20 m and contact TTC 1 / 2 / 3 s.
  Bind SNQI anchors to a digest of the metric definitions (#10049).
- **Reason:** No single threshold is established for a 1 m-radius robot at up
  to 2 m/s; centre distances make "near miss" mean deep contact.
- **Decided by:** orchestrator (delegated).
- **Alternatives:** single thresholds presented as validated.
- **Evidence:** #10079; diss note `2026-10-01_chatgpt_safety_metric_thresholds.md`.
- **Implemented in:** not yet.
- **Enforced by:** none yet: each of D1, D2, D4, D5, D6 with a test that
  fails on base; anchor-digest mismatch refused.
- **Reopen:** New material evidence or an explicit author ruling.

### D-019: The release manifest records the effective pedestrian physics
- **Date:** 2026-10-01
- **Question:** The thesis intake checks pedestrian-model facts against the
  release bundle, and no bundle records them (15 fields unverified).
- **Choice:** Record effective values read from the live simulator objects
  (one sample per scenario): radii (contact, force, metric, placement, grid),
  desired-speed model and cap rule, wall law and parameters, contact rule,
  pedestrian-robot force, group-force parameters, robot reverse cap and
  kinematics, TTC and clearance definitions, dt and integrator.
- **Reason:** Read from what ran, so the manifest cannot disagree with the
  run; 0.1.0 changes exactly these facts.
- **Decided by:** orchestrator (delegated).
- **Alternatives:** record config defaults.
- **Evidence:** #10084; diss#3031.
- **Implemented in:** not yet.
- **Enforced by:** none yet: a test that fails if a declared field differs
  from the live value.
- **Reopen:** New material evidence or an explicit author ruling.

### D-020: Scenario-level simulation keys must apply or be refused
- **Date:** 2026-10-01
- **Question:** The scenario loader whitelist drops `ped_speed_tier` and the
  desired-speed fields silently; "typical" rows were byte-identical to native.
- **Choice:** Admit those keys through the loader and refuse unknown or
  unapplied simulation keys loudly. Blocks #10074.
- **Reason:** 0.1.0 uses the speed tier as its default; a silently dropped key
  is a fail-open path.
- **Decided by:** orchestrator (delegated).
- **Alternatives:** set the tier only in code.
- **Evidence:** #10083 (overlap measurement lane).
- **Implemented in:** not yet.
- **Enforced by:** none yet: load a scenario with `ped_speed_tier: typical`
  and assert the live caps are not all 0.65 (must fail on base).
- **Reopen:** New material evidence or an explicit author ruling.

### D-021: Acceptance criteria for 0.1.0
- **Date:** 2026-10-01
- **Question:** Which proposed acceptance checks define readiness for the new benchmark generation?
- **Choice:** (proposed, from the scope triage, revised by D-010)
  1. One radius parameter (D-004).
  2. V1 speed: mean 1.29 +- 0.05 m/s, SD 0.15-0.25, 30 dev seeds.
  3. V2: a lone pedestrian passes every opening of 1.0 m and wider in 30/30
     dev seeds per width; speed drop at 1.0 m at most 0.40 m/s.
  4. Doorway slice: pedestrians cross in every dev seed at 2.2 / 2.8 / 3.6 m;
     slice re-run (author, #10074 comment 5932040814).
  5. Overlap and wall penetration as in D-010.
  6. V3-V8 reported with differences; every V3/V4 difference above 25 %
     carries a written reason.
  7. Pedestrian-robot activation and attribution (D-012).
  8. Metrics (D-018) and anchor digest.
  9. Rasteriser (D-017).
  10. PPO (D-016).
  11. Behaviour-change gate (#10000) receipt on the freeze head with every new
      failure classified, and a posted refute-review MERGE verdict.
  12. `legacy_v1` with the 0.0.8 plant reproduces the 0.0.8 legacy digest byte
      for byte.
  13. Complete campaign rows; zero fallback or degraded rows; new sealed seeds
      never used in development.
  14. A 0.0.8 -> 0.1.0 report attributes each headline change to a declared
      mechanism.
- **Reason:** Make the ship gate explicit while preserving the distinction between author-approved hard gates and the remaining proposed scope checks.
- **Decided by:** proposed as a whole; the author confirmed V1, V2, overlap and wall penetration as the only calibration hard gates, with bottleneck flows reported as differences with reasons. That confirmation does not approve every other proposed scope threshold.
- **Alternatives:** Treat every milestone item or every bottleneck-flow difference as a hard ship gate.
- **Evidence:** 0.1.0 scope proposal; #10074 comments.
- **Enforced by:** none yet: each item names its own check.
- **Reopen:** New material evidence or an explicit author ruling. Proposed choices remain open for author adoption.

### D-022: New sealed evaluation seeds; development on dev seeds only
- **Date:** 2026-10-01
- **Question:** Can previously observed evaluation seeds support a fresh benchmark-generation claim?
- **Choice:** 0.1.0 gets new sealed evaluation seeds, drawn before the
  campaign and never used in development. All 0.1.0 development uses dev seeds
  1001-1030; the 0.0.8 sealed seeds and retired 111-140 stay forbidden for new
  work.
- **Reason:** Same reasoning as 0.0.8 D-049: a band observed during
  development cannot support a fresh release claim.
- **Decided by:** orchestrator, following the author's 0.0.8 seed rule.
- **Alternatives:** Reuse the 0.0.8 sealed set or the published retired band.
- **Evidence:** #10074 ("Dev seeds only"); #10071.
- **Implemented in:** not yet (#10055, #10059 required before the campaign).
- **Enforced by:** none yet: the held-out guard extended to the new band.
- **Reopen:** New material evidence or an explicit author ruling.


See 0.0.8 D-080 for the approved decision-record policy; its tooling is a separate later lane.

### D-023: Enable the hybrid planner repairs by default
- **Date:** 2026-10-05
- **Question:** Should the three independent hybrid repair switches remain opt-in for 0.1.0?
- **Choice:** Current inputs default to planner `physical_static_exclusion_enabled: true`, planner `goal_next_validity_enabled: true`, and environment `include_goal_next_valid: true`. Explicit values always win. Registered released/frozen inputs through 0.0.8 retain false fill-in at the typed builders, with unchanged resolved mappings and canonical digests.
- **Reason:** author decision 2026-10-05.
- **Decided by:** author for activation; orchestrator (delegated) for the builder-level compatibility registry and provenance ruling in the implementing lane.
- **Alternatives:** Retain opt-in defaults; edit frozen YAML; add compatibility keys to hash-bound mappings. Frozen YAML and recorded identities remain protected.
- **Evidence:** #10145, linked #10105 and #9668; [implementation and diagnostic evidence](../../validation/hybrid_defaults/README.md). Development comparisons require every failure classified; they are not release evaluation evidence.
- **Implemented in:** `robot_sf/common/hybrid_defaults.py`, its explicit source registry, typed planner/environment fill-in, and map-runner source context/provenance. Released learned-policy spaces are covered by the same registry. A later 0.1.0 freeze must explicitly record these new defaults together with the other observation migrations.
- **Enforced by:** `test_current_defaults_enable_all_three_switches`; `test_each_explicit_switch_overrides_the_selected_defaults`; `test_scenario_validity_override_wins_or_rejects_non_boolean`; `test_registered_release_full_dataclasses_and_mapping_match_base`; `test_registry_requires_known_source_and_matching_bytes`; `test_release_registry_covers_learned_observation_contract`; `test_release_scenario_environment_dataclasses_match_full_base_dumps`; `test_native_episode_records_legacy_and_current_builder_default_sets` in `tests/planner/test_hybrid_default_compatibility.py`, plus the paired behavior-change comparison gate and `test_missing_candidate_probe_skips_initial_goal_stop_without_speed_cap` for its observer integrity.
- **Reopen:** author: can still be discussed in more detail. Reopen the technical compatibility ruling if a reviewer shows it changes any recorded 0.0.8 identity or behavior.

Development evidence for D-023: `docs/validation/hybrid_defaults/README.md` records the full 3,084-episode comparison. Empty-world has no new failures; the crowded matrix gains 32 net successes but introduces one pedestrian contact and 35 newly failing cells. These diagnostics require review and do not admit a release.
