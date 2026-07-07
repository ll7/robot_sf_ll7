# Agent Run Manifest

[Back to Documentation Index](./README.md)

**Status**: Canonical convention for making substantial agent-assisted runs auditable.

**Related issue**: [#4756](https://github.com/ll7/robot_sf_ll7/issues/4756)

## What this manifest is for

An `agent_run_manifest.yaml` is a lightweight, copyable record that captures what a substantial
agent-assisted run did, under which tool/model/version and permission mode, which commands and tests
it ran, what evidence it produced, and what trace/log hygiene and human validation were performed.
It makes large Codex, Claude Code, Copilot, or other agent-assisted runs auditable without
introducing a workflow system, telemetry collector, or runtime dependency.

It is a documentation/template/checklist convention only. This first slice does not add mandatory CI
enforcement, YAML schema validation libraries, or GitHub Actions. Follow-up issues may add a schema
checker if validation is desired.

## When the manifest is required

Required for major agent-assisted work that creates or changes any of:

- durable evidence (promoted into `docs/context/evidence/`),
- benchmark or reporting gates,
- generated artifacts that become durable dependencies,
- CI or release policy,
- substantial code paths.

Recommended for multi-agent or multi-run tasks even when each individual task is small, so the
combined run stays reviewable.

## When the manifest is optional

Optional for small comment-only edits, trivial docs tweaks, or single-line fixes that do not touch
durable evidence, gates, generated artifacts, CI/release policy, or substantial code paths. When in
doubt, attach one; a cheap manifest is better than an unrecorded run.

## Where to store it

- For a **PR evidence bundle**: attach or link the manifest in the PR body, or place it next to the
  evidence under `docs/context/evidence/<issue-or-topic>/agent_run_manifest.yaml`. Do not store
  manifests under worktree-local `output/`.
- For an **issue evidence packet**: attach or link the manifest in an issue comment, or place it next
  to the issue evidence under `docs/context/evidence/`.
- Keep manifests repository-relative and tracked when they support durable evidence; keep them as
  linked, redacted attachments when they only support an exploratory run.

Start from [`docs/templates/agent_run_manifest.yaml`](./templates/agent_run_manifest.yaml). A scrubbed
illustrative example is at
[`docs/templates/agent_run_manifest.example.yaml`](./templates/agent_run_manifest.example.yaml).

## What the manifest captures

The template captures:

- `schema_version` (`agent-run-manifest.v1`) so future validation can evolve cleanly;
- tool, model, version, and permission mode;
- worktree and optional workflow run id/name;
- declared and actual agent counts;
- commands run and tests run;
- external pages accessed;
- trace/log references (redacted or stored-privately);
- trace redaction checked flag;
- coarse cost or credit cap;
- stop reason and retry count;
- partial failures;
- evidence produced (with durable locations);
- human validity check note;
- remaining risk.

## Trace and log hygiene

- Do not paste raw traces, private prompts, secrets, or precise token/credit details into a manifest.
- Prefer a `trace_or_log_refs` entry that points to a stored-privately path or a redacted excerpt.
- Set `trace_redaction_checked: true` only after reviewing the referenced traces/logs and confirming
  they do not leak secrets, private prompts, or non-public user data.
- Keep `cost_or_credit_cap` coarse (for example `~10 USD` or `<5% of weekly cap`); do not record
  exact token counts or private billing details.

See the [Artifact Evidence Vocabulary](./context/artifact_evidence_vocabulary.md) for the
durable-vs-local distinction. Manifest `evidence_produced` entries should name durable locations,
not worktree-local `output/` paths.

## Human validation expectations

- A human (reviewer or maintainer) should confirm the manifest fields match the actual run before
  treating the run as established evidence.
- `human_validity_check` should record what was checked (for example, "diff reviewed, validation
  command rerun, template fields checked against issue acceptance criteria") or `pending`.
- Fallback, degraded, failed, or not-available execution modes must be named in `partial_failures` or
  `remaining_risk`; they are caveats, not success evidence, per the
  [Issue #691 Benchmark Fallback Policy](./context/issue_691_benchmark_fallback_policy.md).

## Examples

### PR evidence bundle

A PR that used a nontrivial agent run to add a durable evidence bundle should attach or link a
manifest in the PR body, for example:

```text
Agent run manifest: docs/context/evidence/issue_4756_agent_run_manifest/agent_run_manifest.yaml
- [x] If this PR used a nontrivial agent run, attach or link an agent_run_manifest.yaml and confirm
      trace/log redaction was checked.
```

### Issue evidence packet

An issue comment that posts agent-produced evidence should attach or link a manifest, for example:

```text
Agent run manifest: attached (scrubbed). Evidence produced: docs/context/evidence/issue_NNNN_*/.
Trace redaction checked: true. Human validity check: pending maintainer review.
```

## Verification

A minimal YAML-parse smoke for the template:

```bash
python - <<'PY'
from pathlib import Path
import yaml
for path in [Path('docs/templates/agent_run_manifest.yaml'), Path('docs/templates/agent_run_manifest.example.yaml')]:
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict)
    assert data['agent_run']['schema_version'] == 'agent-run-manifest.v1'
PY
```

If the repo has docs lint checks, run them as well. Otherwise use:

```bash
git diff --check
```

## Non-goals for this slice

- No telemetry collectors, agent integrations, or mandatory CI enforcement.
- No YAML validation libraries or GitHub Actions.
- No change to existing evidence, benchmark, or release gates.

Follow-up issues may add a schema checker or CI enforcement once the convention has been used enough
to justify it.
