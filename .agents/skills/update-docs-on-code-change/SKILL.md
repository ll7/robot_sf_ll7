---
name: update-docs-on-code-change
description: "Keep docs aligned with code changes that affect workflows, commands, contracts, or user-facing behavior."
---

# Update Docs On Code Change

## Overview

Use this skill whenever a code change would leave the repository documentation stale unless the docs
are updated at the same time.

The goal is to keep docs discoverable, concise, and accurate rather than to rewrite them broadly.

## Read First

- `AGENTS.md`
- `docs/dev_guide.md`
- `docs/README.md`
- `docs/ai/repo_overview.md`
- `.github/copilot-instructions.md`

## Workflow

1. Identify the doc surfaces
   - Decide which docs actually describe the changed behavior.
   - Prefer the smallest set of existing docs that users already read.

2. Sync terminology
   - Reuse the repo's existing names for commands, files, and workflows.
   - Avoid introducing parallel wording for the same concept.

3. Update the concrete instructions
   - Update commands, paths, caveats, examples, and links.
   - Keep the text short and executable.

4. Validate the docs
   - Check that referenced files and commands still exist.
   - If the change affects discoverability, add or update a test.

## Guardrails

- Do not add a new doc when an existing doc can be corrected.
- Do not describe code that no longer exists.
- Do not leave workflow docs stale after a code change.
- If the change is user-facing, make sure the docs reflect the new behavior in the same change.
