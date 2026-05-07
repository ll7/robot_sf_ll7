# Benchmark Mechanism Roadmap Design

Date: 2026-05-07

## Goal

Define the next 1-2 month strategic investment area for `robot_sf_ll7`, with a
2-4 week first slice nested inside it.

The priority order is:

1. paper/publication-strength benchmark evidence,
2. stronger planner and policy research platform,
3. reusable open-source library value.

The selected direction is **Benchmark Meaning And Planner Failure Mechanisms**. The program should
turn the current benchmark and h500 evidence from aggregate planner scores into trace-backed
explanations of why planners succeed, time out, collide, or trade safety for completion.

## Context

The current open issue list is narrow and does not by itself capture all strategic opportunities.
The live issues and recent context notes show three important tracks:

- `#1049` h500 trace-backed mechanism pilot, newly opened as a bounded evidence task.
- `#1003` CARLA T1 oracle replay smoke runner, high-priority and agent-ready but scoped as a first
  simulator-transfer proof slice.
- broader h500, policy-search, planner-family, and scenario-certification notes under
  `docs/context/`.

Important evidence sources:

- `docs/context/issue_1044_h500_followup_benchmark_plan.md`
- `docs/context/issue_1045_h500_solvability_mechanisms.md`
- `docs/context/policy_search/reasoning/2026-05-05_h500_research_plan.md`
- `docs/benchmark_spec.md`
- `docs/benchmark_planner_family_coverage.md`
- `docs/scenario_certification.md`
- `docs/context/issue_1001_architecture_seam_audit.md`

## Recommended Program

The recommended 1-2 month program is a paper-first benchmark-interpretation program:

1. Produce trace-backed evidence for representative fixed-vs-h500 cells.
2. Classify the causal mechanism behind each representative outcome.
3. Use those classifications to improve benchmark reporting and paper claims.
4. Create focused follow-up issues only where the evidence justifies planner or tooling work.

The program should answer questions such as:

- Is a fixed-horizon failure a time-budget artifact or a planner failure?
- Does h500 success come from waiting/yielding, delayed continuous progress, replanning/recovery, or
  riskier exposure?
- Which failures are scenario-contract issues, and which are planner-design limitations?
- Which safety and comfort costs appear when the planner is given more time to interact with
  pedestrians?

## First Slice

The first 2-4 week slice should build directly on `#1049`.

Scope:

- Select 3-5 representative planner-scenario-seed cells from the h500 mechanism analysis.
- Include at least one clean budget-relief case, one exposure-enabled completion, and one
  safety-regressed completion.
- Generate fixed-horizon and h500 trace/video evidence where feasible.
- Summarize per-step progress, distance-to-goal, speed/action, pedestrian proximity, near-miss
  timing, collision/timeout/success, and planner mode.
- Update context notes with observed mechanism classifications.

Proof of success:

- At least one representative h500 success has enough trace evidence to support or reject a
  wait-then-go explanation.
- The output clearly separates observed evidence from hypotheses.
- The resulting notes identify follow-up issues for planner improvement, reporting, or scenario
  certification only when those follow-ups are grounded in the trace evidence.

## Follow-On Program Shape

After the pilot, the 1-2 month program should expand only along evidence-backed paths:

- add exposure-aware h500 reporting tables,
- tighten scenario/time-budget/planner-failure classification,
- connect scenario certification to benchmark interpretation,
- add paper-facing language that avoids a single misleading h500 winner table,
- create planner-improvement issues from observed failure mechanisms instead of from aggregate
  scores alone.

Any new code or tooling should serve one of these evidence needs. Avoid broad refactors unless the
pilot exposes a specific blocker in trace generation, reporting, or benchmark interpretation.

## Deferred Alternative: Planner Improvement Program

Why it matters:

- h500 evidence shows persistent deadlocks, route-local minima, near-miss pressure, and
  safety/completion tradeoffs.
- The current policy-search notes already identify promising directions such as recovery-aware
  hybrid planning, comfort-preserving high success, selector safety accounting, and MPC-as-proposer
  behind hard safety filters.
- A successful planner improvement could produce visible research-platform gains.

Why not first:

- Without trace-backed mechanism evidence, planner work risks becoming local tuning against aggregate
  scores.
- A planner win is less paper-strengthening if the repository cannot explain why the previous
  failures occurred.
- The recommended benchmark-mechanism program should produce better-targeted planner issues.

Promotion condition:

- Promote this into the primary track when trace evidence identifies a narrow, repeated planner
  failure mechanism and a targeted slice can prove improvement without raising collision or
  near-miss risk beyond the strict incumbent envelope.

Issue-capture requirement:

- Create a deferred issue or update an existing issue that records the planner-improvement concept,
  why it remains relevant, why it is not first, and what evidence would promote it.

## Deferred Alternative: CARLA Transfer Program

Why it matters:

- CARLA replay can test whether Robot SF scenario conclusions survive a higher-fidelity simulator
  boundary.
- The T0 export/schema stack is mostly in place, and `#1003` is the next executable child issue for
  a T1 oracle replay smoke path.
- Long-term simulator-transfer evidence could strengthen the external validity of the benchmark.

Why not first:

- The immediate paper value is higher from explaining existing Robot SF benchmark outcomes.
- CARLA work is still at the proof-slice stage and may mostly produce optional-runtime
  infrastructure in the first month.
- A narrow smoke runner should proceed when scheduled, but broader CARLA parity should not displace
  the current benchmark-meaning work.

Promotion condition:

- Promote this into the primary track when the T1 smoke path works fail-closed locally and a
  CARLA-capable environment can replay at least one certified scenario enough to support a concrete
  transfer question.

Issue-capture requirement:

- Keep `#1003` as the current implementation slice.
- Add or update a deferred roadmap issue for the broader CARLA transfer program with explicit
  "why relevant", "why not now", and promotion conditions.

## Issue Policy

The roadmap should preserve alternatives without pretending they are current implementation work.

For each strategic track, issue text should include:

- why the track matters,
- why it is or is not the current priority,
- prerequisites,
- proof gates,
- scope boundaries,
- and what evidence would change the priority.

Issue creation or updates should happen after this spec is reviewed. The first implementation plan
should include:

- one primary execution issue for the benchmark-mechanism slice,
- one deferred planner-improvement issue/update,
- one deferred CARLA-transfer issue/update,
- and links between all three.

## Risks

- The h500 pilot may show mixed mechanisms, making the story more nuanced than a clean paper claim.
  That is still useful if the report separates evidence from hypothesis.
- Trace/video generation may require small tooling improvements. Those should stay scoped to the
  pilot's evidence needs.
- Planner-improvement ideas may be tempting to start early. They should wait until the trace pilot
  identifies the strongest target.
- CARLA transfer may attract attention because it is strategically exciting, but the first-order
  paper need is benchmark interpretation.

## Validation Plan

The design itself is validated by:

- checking the referenced context and benchmark docs exist,
- confirming the spec has no unresolved placeholders or contradictory priority statements,
- and committing the design before implementation planning.

Implementation validation will be defined in the follow-up plan. Expected validation surfaces are:

- trace/video artifacts or compact summaries under `docs/context/evidence/`,
- updated context notes for h500 mechanisms and follow-up benchmark interpretation,
- exact commands/configs/commit hashes for generated evidence,
- issue links for the primary and deferred tracks.
