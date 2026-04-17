---
name: autoresearch
description: "Autonomous iterative experimentation loop for measurable Robot SF tasks; use when the user wants an improvement loop with baseline, experiments, and keep/discard decisions."
---

# Autoresearch

## Overview

Use this skill when the task has a measurable metric and the user wants an autonomous improvement
loop rather than a single pass fix.

This is the Agent-compatible version of the upstream `autoresearch` idea adapted to Robot SF's
repo-local workflow, validation gates, and benchmark conservatism.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `.specify/memory/constitution.md`
- `.github/copilot-instructions.md`
- `docs/ai/awesome_copilot_adaptation.md`

## Workflow

1. Define the experiment contract
   - Restate the goal, metric command, metric extraction method, scope, constraints, and stop
     condition.
   - If any of those are missing, ask for them or switch to a clarifying skill.

2. Establish a baseline
   - Run the metric command on the current branch before changing code.
   - Record the baseline in a short log under `output/ai/autoresearch/<slug>/results.tsv` or
     another user-approved scratch location.

3. Iterate on one hypothesis at a time
   - Make the smallest useful edit.
   - Commit the experiment before running it.
   - Run the metric command and any repo-required validation gate.
   - Keep the change only if it improves the metric without violating constraints.
   - Revert the change if it regresses, stalls, or becomes too complex for the gain.

4. Use repo-native validation
   - Prefer `scripts/dev/pr_ready_check.sh` for broad validation unless the user gave a tighter
     command.
   - Prefer `scripts/dev/` wrappers and config-first commands from the dev guide.
   - Treat fallback or degraded benchmark outcomes as caveats, not wins.

5. Report the result
   - Summarize the baseline, best result, discarded attempts, and remaining risks.
   - Call out any assumption that stayed unresolved.

## Guardrails

- Do not add dependencies or change public contracts unless explicitly allowed.
- Keep benchmark wording conservative.
- Use one concept per experiment.
- Stop when the user budget is reached or the metric stops improving.
