# Understand-Anything Knowledge Graph

[Back to Documentation Index](../README.md)

This repository tracks an Understand-Anything graph as a shared orientation artifact for coding
agents and contributors. It complements the Markdown context stack; it does not replace source
inspection, benchmark validation, or issue-specific context notes.

## What Is Tracked

Tracked under `.understand-anything/`:

- `knowledge-graph.json`: graph used by the interactive dashboard.
- `fingerprints.json`: structural baseline used for incremental update decisions.
- `meta.json`: graph timestamp, source commit hash, version, and analyzed file count.
- `config.json`: project-level Understand-Anything settings.
- `.understandignore`: graph-specific ignore rules.

Ignored under `.understand-anything/`:

- `intermediate/`
- `tmp/`
- `diff-overlay.json`

`knowledge-graph.json` and `fingerprints.json` are Git LFS files. Do not stage them without Git LFS
installed and initialized.

## Codex Setup

Install the upstream skills for Codex:

```bash
curl -fsSL https://raw.githubusercontent.com/Lum1104/Understand-Anything/main/install.sh | bash -s codex
```

Then restart Codex so it sees the installed skills. The upstream installer links skills into
`~/.agents/skills` and creates the stable plugin-root symlink at `~/.understand-anything-plugin`.

Install Git LFS before pulling or updating shared graph artifacts:

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install git-lfs

git lfs install
git lfs pull
```

For this repository, LFS tracking is declared in `.gitattributes`.

## How Agents Should Use It

Use the graph for orientation before broad repo reads:

- Launch the dashboard with `/understand-dashboard` from the repository root.
- Inspect layers and tour steps to identify likely files.
- Use `/understand-chat` for graph-grounded questions when available.
- Use `/understand-diff` when a diff overlay is useful for local review.

Good graph questions:

- Where does benchmark execution start?
- Which files define planner interfaces and adapters?
- What classes or functions are central to environment creation?
- Which tests appear connected to a metric or planner module?
- Which configs point at a benchmark or training workflow?

Do not treat graph output as proof for benchmark, metric, schema, model-provenance, or paper-facing
claims. For those, follow the linked source files, run the relevant repo command, and record the
validation evidence in the PR, issue, or context note.

## Updating The Graph

The shared config enables Understand-Anything auto-update:

```json
{
  "autoUpdate": true,
  "outputLanguage": "en"
}
```

Prefer post-commit or explicit graph refreshes. Do not put full graph generation in a pre-commit
hook: it is slow, can mutate large files during commit preparation, and can make ordinary commits
hard to reason about.

Use these update paths:

- For ordinary source changes, run `/understand` from the repository root after the code/docs commit
  or when Codex reports the graph is stale.
- For large structural changes, run `/understand --full`.
- For a local dashboard-only refresh, do the same commands but do not commit the result unless the
  graph improvement is meant to become shared repo state.

Before committing graph updates, verify:

```bash
git lfs status
git lfs ls-files
git status --short -uall -- .understand-anything .gitattributes
```

The large JSON files should appear as LFS-tracked files. In the Git index they should be pointer
files, not raw multi-megabyte JSON blobs.

## Artifact Policy

The graph is intentionally shared because it improves repo navigation for large-agent tasks. Keep
the scope disciplined:

- Commit graph updates when they support shared onboarding, agent navigation, or review workflows.
- Avoid graph-only churn after small changes unless the graph is stale enough to mislead users.
- Do not commit `intermediate/`, `tmp/`, or `diff-overlay.json`.
- Do not use the graph as the durable record for experiments, benchmark outputs, or paper evidence.
  Use `docs/context/evidence/`, `memory/`, or the existing benchmark artifact policy for those.
