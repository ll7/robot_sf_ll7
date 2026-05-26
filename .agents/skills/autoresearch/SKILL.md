---
name: autoresearch
description: Autonomous iterative experimentation loop for measurable Robot SF tasks; use when the user
  wants an improvement loop with baseline, experiments, and keep/discard decisions.
category: research-iteration
kind: atomic
phase: analysis
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# Autoresearch

## When to use

Use this skill when the task has measurable goals and needs an iterative experiment loop (hypothesis,
baseline, test, keep/discard decisions), not a single-pass edit.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `.specify/memory/constitution.md`
- `.github/copilot-instructions.md`
- `docs/ai/awesome_copilot_adaptation.md`

## Workflow

1. Define the experiment contract
   - Restate the goal, metric command, extraction method, scope, constraints, and stop condition.
   - If any input is missing, ask for it or switch to a clarifying skill.

2. Establish a baseline
   - Run the metric command on the current branch before changing code.
   - Record baseline and metadata in `output/ai/autoresearch/<slug>/results.tsv` (or approved location).

3. Iterate on one hypothesis at a time
   - Make the smallest useful edit.
   - Commit the experiment snapshot before running it.
   - Run the metric command and any repo-required validation gate.
   - Keep only if metric improves and constraints are satisfied.
   - Revert the change if it regresses, stalls, or becomes too complex for the gain.

4. Use repo-native validation
   - Prefer `scripts/dev/pr_ready_check.sh` for broad validation unless the user gave a tighter
     command.
   - Prefer `scripts/dev/` wrappers and config-first commands from the dev guide.
   - Treat fallback or degraded benchmark outcomes as caveats, not wins.

5. Report the result
   - Summarize the baseline, best result, discarded attempts, and remaining risks.
   - Call out any assumption that stayed unresolved.

## Proof and Guardrails

- Do not add dependencies or change public contracts unless explicitly allowed.
- Keep benchmark wording conservative.
- Use one concept per experiment.
- Stop when the user budget is reached or the metric stops improving.
- If benchmark evidence is invalid or non-reproducible, stop and report that status as incomplete.

## Output

Summarize:

- experiment contract,
- baseline and best result,
- discarded attempts and reasons,
- unresolved assumptions and next risk.
## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.
