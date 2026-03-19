# Context Packing Decision

Date: 2026-03-19

## Decision

Recommend **Repomix** as the default context packer for `robot_sf_ll7`.

`Code2Prompt` remains a viable secondary option, but it is not the default recommendation for this
repository.

## Why Repomix Wins For This Repo

This repository needs focused, regenerable context bundles for benchmark, planner, and training
subsystems. Repomix is the better default because it combines the features that matter most for
that workflow:

- token-count tree inspection for finding context-heavy files,
- include/ignore controls for subsystem-focused bundles,
- optional compression for structure-first packing,
- security scanning aimed at avoiding accidental secret inclusion,
- multiple output styles and split-output support for large repositories,
- optional git diff/log inclusion when change context matters.

That feature set is a closer fit for recurring repo-local context bundles than a simple
"flatten the repo into one prompt" workflow.

## Comparison

### Repomix

Strengths:

- better support for targeted repository packs,
- built-in token-distribution analysis,
- security-aware packaging defaults,
- compression and split-output support,
- good fit for repeatable "pack this subsystem" workflows.

Weaknesses:

- larger feature surface than is needed for one-off prompt generation,
- compression mode still needs human judgment because code extraction can hide implementation detail.

### Code2Prompt

Strengths:

- excellent direct codebase-to-prompt workflow,
- strong templating and token-count support,
- useful when the goal is a single prompt artifact quickly.

Weaknesses:

- more prompt-generation oriented than bundle/regeneration oriented,
- less compelling than Repomix for this repo's likely need to create multiple subsystem-focused
  packs over time.

## Recommended Usage In This Repository

Start with a documented Repomix workflow for focused bundles such as:

- benchmark core (`robot_sf/benchmark`, `configs/benchmarks`, key docs),
- planner integration (`robot_sf/nav`, planner docs, planner quality audit docs),
- training/eval context (`scripts/training`, `configs/training`, relevant runbooks).

Only add automation around this after a few manual bundle workflows prove useful.

## Source Note

This decision was checked against the current upstream project pages on 2026-03-19.
Repomix currently advertises security checks, compression, token-count trees, split output, and
multiple output styles; Code2Prompt currently advertises prompt templating, token counting, CLI/TUI
usage, Python SDK support, and MCP support.
