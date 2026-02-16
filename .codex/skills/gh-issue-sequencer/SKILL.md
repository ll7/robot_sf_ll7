---
name: gh-issue-sequencer
description: "Maintain a clear next-work queue in GitHub by normalizing issue metadata, project status/priority fields, and producing a deterministic execution order."
---

# GH Issue Sequencer

## Overview

Use this skill to create and maintain a sequential, high-signal backlog so the next work item is
always obvious.

## Goal State

- Exactly one issue is `In progress`.
- A small, explicit queue (for example top 3) is `Ready`.
- Remaining actionable issues are `Todo` or `Tracked`.
- Blocked/decision-heavy issues are labeled and not silently mixed into ready work.

## Workflow

1. Load project board context
   - `gh project view 5 --owner ll7 --format json`
   - `gh project field-list 5 --owner ll7 --format json`
   - `gh project item-list 5 --owner ll7 --limit 200 --format json`

2. Normalize metadata for open issues
   - Ensure issue has meaningful labels (`bug`, `enhancement`, domain labels like `validation`, `training`, `global planner`, `performance`).
   - Ensure milestone is set when roadmap context exists (for example `benchmark` milestone).
   - Ensure project `Priority` is set (`Very High`..`Very Low`) for actionable items.
   - Move stale completed items to `Done`.

3. Classify issues into lanes
   - `In progress`: actively being implemented now.
   - `Ready`: clear scope + acceptance criteria + no missing decisions.
   - `Todo`: valid but not yet ready to start.
   - `Tracked`: dependent, blocked, or exploratory.

4. Build deterministic order
   - Rank by:
     1. `Status` lane (`Ready` first)
     2. `Priority` (`Very High` to `Very Low`)
     3. impact label preference (`bug` before `enhancement`, then others)
     4. age (older issue number first)
   - Select and mark:
     - next one as `In progress` (if none already active)
     - next 2-3 as `Ready`
     - demote rest to `Todo` / `Tracked` as needed.

5. Publish the sequence
   - Add/update a project-level summary comment (or issue comment on the selected next issue):
     - `Now`: `#<issue>`
     - `Next`: `#<issue>, #<issue>`
     - `Blocked/Decision`: `#<issue>`
   - Use `scripts/dev/gh_comment.sh` for multiline formatting.

6. Keep sequence healthy
   - Remove `Ready` from issues missing acceptance criteria.
   - Add `decision-required` label for unresolved decisions and move to `Tracked`.
   - Split oversized issues into follow-ups and link them.

## Practical Commands

- Add issue to project:
  - `gh project item-add 5 --owner ll7 --url https://github.com/ll7/robot_sf_ll7/issues/<n>`
- Edit labels/milestone:
  - `gh issue edit <n> --add-label "<label1>,<label2>" --milestone "<milestone>"`
- Update project single-select fields:
  - `gh project item-edit --id <item_id> --project-id <project_id> --field-id <field_id> --single-select-option-id <option_id>`

## Output Requirements

- Explicitly list:
  - current `In progress` issue
  - ordered `Ready` queue
  - `Tracked` items requiring decisions
  - any metadata fixes applied (labels, milestone, priority, status)
