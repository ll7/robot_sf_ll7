---
name: skill-picker
description: "Choose the most appropriate repo-local skill for an ambiguous task by consulting .agents/skills/README.md; use when the user asks which skill to use, does not know the right skill, or explicitly asks for skill routing."
---

# Skill Picker

## Overview

Use this skill when the user wants help choosing a repo-local skill before executing a task.

Good triggers:
- "which skill should I use?"
- "pick the right skill"
- "use the skill picker"
- "I do not know which skill applies"

Do not use this skill as a mandatory pre-step for ordinary tasks. If the user already named a
specific skill, use that skill directly unless the request asks you to reconsider the choice or the
named skill is clearly not the best fit for the task.

When the user explicitly asks for skill routing, prioritize the task over the user's tentative skill
choice: select the best matching skill, briefly state why it overrides the mentioned skill if
needed, and proceed with the task. Do not stop to ask for confirmation unless the task itself is
unsafe or missing required information.

## Workflow

1. Read the skill index
   - Open `.agents/skills/README.md`.
   - Use it as the routing overview before reading individual `SKILL.md` files.

2. Classify the task
   - Identify the task type: issue workflow, PR/review, implementation verification, benchmark
     review, docs sync, context discovery, experimentation, or cleanup.
   - Note whether the user requested execution now or only skill selection advice.

3. Select the smallest useful skill set
   - Prefer one specific skill over a broad stack.
   - Override a user-mentioned skill when another skill clearly fits better.
   - Combine skills only for distinct phases, for example:
     `context-map` -> `implementation-verification` -> `pr-ready-check`.
   - Avoid selecting skills that duplicate the same phase.

4. Report and proceed
   - State selected skill(s) and why.
   - Briefly state obvious alternatives skipped when the choice is non-obvious.
   - If the user asked you to proceed, load the selected skill's `SKILL.md` and continue.

## Output

Use a compact routing note:

```md
Selected: `skill-name`
Why: <one sentence>
Skipped: `<alternative>` because <one sentence>
Override: `<user-mentioned-skill>` because <one sentence>  # omit when not needed
Next: <what I will do>
```

If no skill fits, say that directly and continue with the normal repository workflow.
