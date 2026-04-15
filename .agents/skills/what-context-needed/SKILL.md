---
name: what-context-needed
description: "Ask for the minimum repository context needed to answer or implement a task safely; use when the task is underspecified or missing required files."
---

# What Context Needed

## Overview

Use this skill when the current task cannot be answered safely without additional repository
context.

The output should be a short, direct request for the missing files, commands, or decisions. Keep it
minimal and action-oriented.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `docs/code_review.md`
- `.specify/memory/constitution.md`

## Workflow

1. State what is already known
   - Summarize the task in one or two sentences.
   - Mention the files or signals already available.

2. Identify the blockers
   - Group blockers into:
     - missing files,
     - missing decision,
     - missing validation path,
     - ambiguous scope.

3. Ask for the minimum needed
   - Ask up to three short questions.
   - Prefer exact file paths, issue numbers, or commands over open-ended requests.

4. Offer the next step
   - Say which skill or workflow would apply once the missing context is provided.

## Output

Ask for the smallest missing set of files, commands, artifacts, or decisions, and explain what each
item will unblock.

## Guardrails

- Do not invent missing facts.
- Do not ask broad multi-part questions when one precise question is enough.
- If the task becomes clear after the clarification, proceed with the appropriate execution skill.
