# Question-First Experiment Registry

This directory records planned or active exploratory experiments before they run. It complements
GitHub issues, W&B artifacts, model registries, and publication bundles; it does not make local
`output/` files durable by itself.

Each record must state:

- `experiment_id`
- `issue` and `issue_url`
- `question`
- `hypothesis`
- `config`
- `command`
- `inputs`
- `outputs`
- `expected_artifacts`
- `evidence_grade`
- `paper_relevance`
- `status`

## Run-Ready Launch-Packet Manifest

`experiments/run_ready_manifest.yaml` is generated from launch packets under `configs/`:

```bash
uv run python scripts/dev/build_run_ready_manifest.py --output experiments/run_ready_manifest.yaml
```

The manifest is a preflight and discovery surface for queue tooling. `ready: true` only means the
packet passed local path, checksum, seed-budget, command, and claim-boundary checks; it does not
assert benchmark results, paper evidence, or permission to submit a job.

## Experiment Record v2

`experiment-record.v2` is the authoritative research-control-plane schema for new sprint studies.
It replaces free-form `status` with one explicit `state` field. GitHub labels, status comments,
dashboards, and release checklists should be derived from this state instead of maintained as
independent runtime databases.

Active states:

```text
idea -> protocol_frozen -> implementation_ready -> preflight_passed -> submitted -> running
  -> finalized -> evidence_promoted -> claim_reviewed -> released
```

Terminal alternatives:

```text
blocked_external
invalid_execution
negative_result
null_result
superseded
stopped_by_gate
```

Derived GitHub state labels:

| Record state | Derived issue label |
| --- | --- |
| `idea`, `protocol_frozen`, `implementation_ready`, `preflight_passed` | `state:ready` |
| `submitted`, `running`, `finalized`, `evidence_promoted`, `claim_reviewed` | `state:running` |
| `blocked_external`, `invalid_execution`, `negative_result`, `null_result`, `superseded`, `stopped_by_gate` | `state:blocked` |
| `released` | remove `state:*` labels |

Closed GitHub issues are historical truth unless a new follow-up issue is opened. Stale cards
should be corrected or superseded through a PR rather than silently treated as active work.
Local `output/` paths and `:pending` artifact aliases are not durable evidence.
For legacy `experiment-record.v1` cards, use `status: proposal` for non-actionable proposals that
still contain placeholder commands or expected artifacts. `planned` is treated as actionable by the
validator, so required durable artifacts must have concrete `durable_reference` values and command
or artifact paths must not contain placeholder tokens such as `<campaign-id>`.
For `experiment-record.v2` cards, `state: idea` and `state: proposal` are the equivalent
non-actionable proposal states; move the card to an implementation or execution state only after
placeholder commands and durable artifact references are resolved.

### Control-plane dry-run report

Use the validator to emit a compact drift report before new empirical campaigns start:

```bash
uv run python scripts/tools/validate_experiment_registry.py \
  experiments/registry.yaml \
  --issue-state-json output/experiments/issue_state_snapshot.json \
  --control-plane-report-json output/experiments/control_plane_report.json
```

The issue-state snapshot is a compact JSON list or object with `number`, `state`, and `labels`.
The report detects closed issues with nonterminal cards, closed blockers with blocked cards,
dry-run derived `state:*` label updates, missing config/input paths, pending artifact aliases, and
expected artifacts that still need durable references. It reports `labels_to_add` and
`labels_to_remove` without mutating GitHub by default.

To explicitly project derived registry state labels back to GitHub issues, rerun the same command
with `--apply-labels` and a write cap:

```bash
uv run python scripts/tools/validate_experiment_registry.py \
  experiments/registry.yaml \
  --issue-state-json output/experiments/issue_state_snapshot.json \
  --control-plane-report-json output/experiments/control_plane_report.json \
  --apply-labels \
  --max-writes 5
```

The apply path only touches `state:*` labels from existing `derived_issue_label_update` findings,
checks `gh api rate_limit` before writing, honors `--max-writes`, and writes a disposable audit log
under `output/issue_state_sync/`.
This v2 control-plane guidance is part of the #3057 research-control-plane work and is exercised by
`tests/tools/test_validate_experiment_registry.py`.

Use `paper_relevance: exploratory` for local pilots and early research runs. Use
`paper_relevance: paper_facing` only when every local `output/` artifact listed in `outputs` or
`expected_artifacts` has a durable `durable_reference`, such as a W&B artifact, model registry
entry, release asset, or tracked evidence manifest.

## Create / Review / Validate / Update Flow

### Create a draft experiment card

```bash
uv run python scripts/tools/create_experiment_card.py \
  --issue 2103 \
  --experiment-id issue_2103_example \
  --template benchmark-analysis \
  --output-root output/experiments/issue_2103_example
```

Available templates: `benchmark-analysis`, `planner-ablation`, `figure-table-pack`.

This writes:

- `output/experiments/<experiment-id>/<experiment-id>.yaml` - the experiment record,
- `output/experiments/<experiment-id>/CHECKLIST.md` - validation and promotion checklist.

Generated records contain `TODO` placeholders that must be filled before the card is actionable.

### Review

Edit the generated YAML to replace all `TODO` placeholders with concrete config paths,
commands, and hypothesis details.

### Validate

```bash
# Validate the full registry after registering the card.
uv run python scripts/tools/validate_experiment_registry.py experiments/registry.yaml
```

### Update

1. Edit the record YAML when values change.
2. Re-validate with `validate_experiment_registry.py`.
3. Register the card in `experiments/registry.yaml` (add the filename to the `records` list).
