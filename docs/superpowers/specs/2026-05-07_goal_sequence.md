  1. #1052 docs: audit paper claim language against benchmark-set boundary
     First because it defines what the paper is allowed to claim. Everything else should serve that claim boundary.
  2. #1051 bench: audit camera-ready evidence provenance and SNQI trail
     Confirms whether the existing paper evidence is actually complete: PPO provenance, benchmark command, SNQI,
     seed/bootstrap, release inputs.
  3. #1053 docs: audit durable artifact references for paper evidence
     Follows #1051 because it checks whether the evidence trail is recoverable outside this worktree.
  4. #1054 bench: audit planner readiness and fallback modes for paper matrix
     Ensures baseline rows are valid evidence, not fallback/degraded caveats.
  5. #1057 bench: audit semantic blockers before paper failure attribution
     Checks route handoff, SVG geometry, SNQI semantics, metric drift, and fallback/degraded status before failures
     are interpreted as planner behavior.
  6. #1049 Run trace-backed h500 mechanism pilot
     This is the core benchmark-mechanism slice. It should run after the paper-risk guardrails above so its outputs
     are interpreted correctly.
  7. #1056 research: classify h500 failures by scenario, time-budget, and planner mechanism
     Depends on #1049; turns trace evidence into reusable mechanism categories.
  8. #1055 bench: add exposure-aware h500 reporting tables
     Depends on #1049 and benefits from #1056; converts mechanisms and raw traces into safer reporting.
  9. #1058 docs: write paper-facing h500 interpretation language
     Comes after classification/reporting so the wording is evidence-backed.