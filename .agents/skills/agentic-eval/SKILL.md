---
name: agentic-eval
description: Evaluate and improve AI workflow outputs with small goldens, rubrics, and repeatable checks;
  use when tuning skills, prompts, instructions, or agent behavior.
category: research-iteration
kind: atomic
phase: analysis
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# Agentic Eval

## When to use

Use this skill when the target is an AI workflow artifact (skills, prompts, rubrics, issue text,
review guidance, etc.) rather than core product code.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `docs/ai/awesome_copilot_adaptation.md`
- `docs/code_review.md`

## Workflow

1. Define the artifact and objective
   - State the artifact, expected behavior, and what a pass/fail result looks like.
   - Set a small representative evaluation set.

2. Build a rubric
   - Use 3–6 concrete criteria.
   - Mark critical failures separately from quality preferences.

3. Create a baseline
   - Score the current artifact against the rubric before editing.

4. Improve one dimension at a time
   - Make a targeted change.
   - Re-run the same evaluation.
   - Keep the change only if improvement is clear and reproducible.

5. Capture proof
   - Compare baseline vs post-change with the same rubric and commands.
   - Keep the result compact: what improved, what regressed, what remains unknown.

## Proof and Guardrails

- Always preserve baseline and post-change results.
- Reuse the same corpus for all iterations.
- If changes drift from the target behavior, revert and document the false regression.
- Prefer `autoresearch` when the task has repeated measurable benchmarks.
- If benchmark evidence is required, treat failed/invalid runs as inconclusive, not proof.

## Output

Report:

- artifact inspected,
- rubric and dataset,
- baseline and final score,
- keep/discard decision with rationale,
- open questions.
## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.
