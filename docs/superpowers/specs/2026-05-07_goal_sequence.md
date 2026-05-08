# Goal Sequence — 2026-05-07

Execution queue for the camera-ready paper-claim audit and the h500 benchmark-mechanism slice.
Items are ordered so each one's outputs become safe inputs for the items below it. See the
[benchmark-mechanism roadmap plan](../plans/2026-05-07-benchmark-mechanism-roadmap.md) for the
executable first slice.

1. [#1052 — docs: audit paper claim language against benchmark-set boundary](https://github.com/ll7/robot_sf_ll7/issues/1052)
   First because it defines what the paper is allowed to claim. Everything else should serve that
   claim boundary.
2. [#1051 — bench: audit camera-ready evidence provenance and SNQI trail](https://github.com/ll7/robot_sf_ll7/issues/1051)
   Confirms whether the existing paper evidence is actually complete: PPO provenance, benchmark
   command, SNQI, seed/bootstrap, release inputs.
3. [#1053 — docs: audit durable artifact references for paper evidence](https://github.com/ll7/robot_sf_ll7/issues/1053)
   Follows #1051 because it checks whether the evidence trail is recoverable outside this worktree.
4. [#1054 — bench: audit planner readiness and fallback modes for paper matrix](https://github.com/ll7/robot_sf_ll7/issues/1054)
   Ensures baseline rows are valid evidence, not fallback/degraded caveats.
5. [#1057 — bench: audit semantic blockers before paper failure attribution](https://github.com/ll7/robot_sf_ll7/issues/1057)
   Checks route handoff, SVG geometry, SNQI semantics, metric drift, and fallback/degraded status
   before failures are interpreted as planner behavior.
6. [#1049 — Run trace-backed h500 mechanism pilot](https://github.com/ll7/robot_sf_ll7/issues/1049)
   Core benchmark-mechanism slice. It should run after the paper-risk guardrails above so its
   outputs are interpreted correctly.
7. [#1056 — research: classify h500 failures by scenario, time-budget, and planner mechanism](https://github.com/ll7/robot_sf_ll7/issues/1056)
   Depends on #1049; turns trace evidence into reusable mechanism categories.
8. [#1055 — bench: add exposure-aware h500 reporting tables](https://github.com/ll7/robot_sf_ll7/issues/1055)
   Depends on #1049 and benefits from #1056; converts mechanisms and raw traces into safer
   reporting.
9. [#1058 — docs: write paper-facing h500 interpretation language](https://github.com/ll7/robot_sf_ll7/issues/1058)
   Comes after classification/reporting so the wording is evidence-backed.
