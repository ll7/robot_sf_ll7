# Adoption path: install, run, inspect, and explore

This page gives a newcomer one short path from a fresh install to a visible Robot SF run and the
next useful exploration step. It is the user-facing index for the adoption/UX product layer; the
[User Guide](user-guide.md) remains the broader task-oriented reference.

## Claim boundary

This is a discoverability guide. The commands below provide local smoke or inspection evidence,
not benchmark evidence. Demo summaries, recipe outputs, notebook plots, and gallery metadata are
worktree-local convenience artifacts unless they are separately promoted through the benchmark
provenance workflow. A single run or an estimated runtime must not be used to rank planners or
support a paper-facing claim.

## 1. Install and check the host

From the repository root, install the standard development dependencies and ask the readiness
checker for actionable local diagnostics:

```bash
uv sync --all-extras
uv run robot-sf doctor
```

For a quick host-only check that does not execute the environment or manifest quickstarts, use:

```bash
uv run robot-sf doctor --skip-env-smoke --skip-quickstart-smoke
```

The doctor report is the first fail-closed boundary: fix reported missing tools, imports, model
artifacts, or quickstart failures before interpreting later output.

## 2. Run one visible episode

Run the deterministic CPU demo and open the generated viewer in a browser:

```bash
uv run robot-sf demo --output-root output/demo/latest --seed 270
```

The disposable output directory contains:

| Artifact | Use |
| --- | --- |
| `episode.jsonl` | Inspect the recorded per-step trace. |
| `summary.json` | Read the plain-English outcome and claim boundary. |
| `metrics.json` | Inspect machine-readable local outcome metrics. |
| `viewer/index.html` | Play back the episode in a browser. |
| `thumbnail.png` | See the top-down map and route at a glance. |

These artifacts answer “does the install run and produce something visible?” They do not answer
“which planner is better?”

## 3. Discover examples and blessed workflows

Use the manifest-backed example catalog when you want source-level examples, and the curated recipe
catalog when you want a copy-pasteable workflow without learning repository paths first:

```bash
uv run robot-sf examples list
uv run robot-sf examples run quickstart/01_basic_robot --fast

uv run robot-sf recipe list
uv run robot-sf recipe explain first-demo
uv run robot-sf recipe run first-demo
```

`examples` is the source-of-truth inventory for example scripts. Recipes are thin mappings to
existing scripts and configs; they do not add simulation or training logic. Use
`uv run robot-sf recipe run <id> --dry-run` to inspect a command before executing it. The
`ppo-smoke` recipe exercises config loading in dry-run mode and is not a training result.

## 4. Inspect scenarios and try a controlled comparison

Build the self-contained scenario gallery to browse maps, scenario metadata, estimated runtime,
and runnable commands:

```bash
uv run robot-sf gallery build \
  --matrix configs/baselines/example_matrix.yaml \
  --out-dir output/gallery
```

Open `output/gallery/index.html`. The gallery's thumbnails, supported-planner labels, and runtime
estimates are inspection aids. They are not measured per-scenario capabilities or benchmark
results. For a rigorous comparison, follow the [Research & Benchmark Guide](research-guide.md)
and its declared matrix, seeds, metrics, and artifact-provenance requirements.

## 5. Continue by task

- Compare or visualize teaching runs: [beginner notebooks](../notebooks/README.md).
- Check local model availability: `uv run robot-sf models list` and
  `uv run robot-sf models verify`.
- Check external-data layout without downloading: `uv run robot-sf datasets list` and
  `uv run robot-sf datasets prepare <id>`.
- Discover the supported environment catalog: `uv run robot-sf envs list` and
  `uv run robot-sf envs describe <env-id>`.
- Move to reproducible evaluation: [Research & Benchmark Guide](research-guide.md).

All generated files in this path belong under the git-ignored `output/` directory. Keep raw runs,
videos, checkpoints, and large datasets out of the repository; promote only durable, provenance-
complete evidence through the established evidence workflow.
