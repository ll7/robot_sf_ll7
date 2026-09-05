# Open-Issue Contract Preparation

## Purpose

`scripts/dev/prepare_open_issue_contracts.py` turns the report-only
`open_issue_contract_audit.v1` output (from
`scripts/dev/audit_open_issue_contracts.py`, merged via #7909) into per-issue
`goal-autopilot` preparation packets. It is the apply successor for #7929: it
makes every open issue actionable through the correct authority and makes true
implementation leaves directly claimable by `goal-autopilot`, without falsely
promoting everything to `state:ready`.

## Contract boundaries

The helper is a bounded-github-mutation tool:

- **Report-only by default.** Plan, render, and verify modes never write.
  Apply mode requires an explicit `--apply` flag and a reviewed plan digest.
- **Never reimplements canonical owners.** Classification, claim ownership,
  dependency resolution, blocker transitions, terminal-label policy, and
  scientific admission stay in their canonical `scripts/dev/` modules.
- **Does not create arbitrary labels**, never adds `runner:luna`/`runner:max`
  to issues, and never mutates issue state, assignments, milestones, projects, comments,
  parent relations, PRs, or merges. A reviewed readiness transition is the
  exception: it invokes the canonical `issue_readiness_gate.gate_issue()`
  function, and the generic label writer is never allowed to add `state:ready`.
- **One marker per issue.** At most one `goal-autopilot-preparation:v1` block;
  bytes outside the marker region are preserved.

## Workflow

### 1. Audit (report-only)

```bash
uv run python -m scripts.dev.audit_open_issue_contracts \
  --repo ll7/robot_sf_ll7 \
  --format json \
  --output /tmp/open_issue_audit.json
```

The audit must be `complete: true` with `errors: []` before planning.

### 2. Plan (report-only)

```bash
uv run python -m scripts.dev.prepare_open_issue_contracts \
  --audit-json /tmp/open_issue_audit.json \
  --plan-json /tmp/open_issue_preparation_plan.json \
  --plan-markdown /tmp/open_issue_preparation_plan.md \
  --batch-id <stable-batch-id>
```

The plan JSON (`open_issue_preparation_plan.v1`) contains per-issue packets
with classification before/after, execution mode, worker route, next action,
authority, admission reason, proposed label plan, and skip reasons. Its summary
retains the admission-reason histogram and `not_admitted` projection. `mutation_authorized` is
always `false` in plan mode.

An ordinary complete, local, leaf issue with no blocking authority and no
execution-state label remains `state_conflict` in the canonical audit (the
admission contract is not weakened), but the preparation planner emits a
`gate_readiness` operation instead of a generic `add state:ready` operation:

```yaml
action: gate_readiness
issue: 1234
expected_body_sha256: "<issue-body-sha256>"
expected_labels: ["type:workflow"]
```

Incomplete contracts remain formalization work. Parent, decision, external
input, compute, blocker, and active-handoff classifications retain their own
authority and are never promoted by preparation.

### 3. Render (report-only)

```bash
uv run python -m scripts.dev.prepare_open_issue_contracts \
  --audit-json /tmp/open_issue_audit.json \
  --mode render --issue <number> --batch-id <stable-batch-id>
```

Prints the exact `goal-autopilot-preparation:v1` marker block for one issue.

### 4. Review

Every `ready`, `needs_ready_label`, `needs_spec`, parent, decision, compute,
blocker, active, and error group is reviewed separately. Candidate executable
leaves run current-main stale-evidence and duplicate-PR searches. The approved
plan digest and deterministic batch membership are frozen; any issue/thread
change invalidates that item and may invalidate its batch.

### 5. Verify (report-only)

```bash
uv run python -m scripts.dev.prepare_open_issue_contracts \
  --audit-json /tmp/open_issue_audit.json \
  --mode verify \
  --bodies-json /tmp/prepared_bodies.json
```

Checks exactly one marker per prepared issue and byte preservation outside the
marker region (with the documented boundary-newline normalization).

### 6. Apply (bounded, after review)

```bash
uv run python -m scripts.dev.prepare_open_issue_contracts \
  --audit-json /tmp/open_issue_audit.json \
  --mode apply --apply \
  --batch-id <stable-batch-id> \
  --mutation-ceiling 10 \
  --issues <number>... \
  --reviewed-plan-digest <selected-plan-content-sha256> \
  --source-ref origin/main
```

Real apply requires an explicit issue list, a digest of that exact selected
plan, and a complete error-free audit. Apply is exact-item and label-set
compare-and-swap guarded:

- default maximum 10 issue-body mutations per batch; hard ceiling 25 body
  mutations or 50 label operations, whichever occurs first;
- the reviewed `label_plan` is validated before any write: generic operations
  name their entry issue, use only `add`/`remove`, refer to an existing
  repository label, contain no duplicate operation, and never target
  `runner:luna`/`runner:max`; readiness operations instead carry the exact
  expected body digest and label set and invoke the canonical gate;
- every actionable issue is re-read before mutation; any drift in state,
  body, labels, or assignees aborts the batch before its first write;
- readiness gates run before their marker body write, then generic label
  cleanup follows the body, because GitHub REST writes are not transactional;
  each operation is recorded and a partial receipt identifies the exact
  completed and skipped operations;
- this issue does not add a body compare-and-swap protocol: body writes retain
  the existing marker-only composition and immediate body readback contract;
- the batch aborts on unknown identity, active-owner drift, audit mismatch,
  rate limit, or REST error;
- body and label writes are immediately read back and verified;
- the receipt records the selected plan and audit digests, expected/observed
  body and label state, readiness-gate outcomes, operation status, API-helper
  status, safe order, partial failure, and proof that no unauthorized mutation
  occurred.

Use `--dry-run` first to render the exact operations without writing.

## Fail-closed rules

The helper stops before implementation or apply when:

- the audit is incomplete, mutation-authorized, or has unresolved errors;
- `main` is red for work other than the exact CI-repair issue;
- pagination, exact reads, claim state, dependency state, or digests are
  incomplete;
- an active owner or covering PR exists for the same mutation boundary;
- a label transition lacks an exact canonical authority;
- the mutation ceiling is reached;
- any write changes content outside the approved marker or labels outside the
  approved set;
- GitHub returns a permission, rate-limit, conflict, or verification error.

A failed batch leaves all unprocessed issues unchanged and preserves a compact
failed-batch receipt.

## Validation

```bash
uv run pytest tests/dev/test_prepare_open_issue_contracts.py -q
uv run pytest tests/dev/test_audit_open_issue_contracts.py -q
uv run pytest tests/dev/test_issue_implementability.py -q
uv run ruff check scripts/dev/prepare_open_issue_contracts.py
uv run ruff format --check scripts/dev/prepare_open_issue_contracts.py
```

## See also

- Issue #7929 (this work's contract)
- `scripts/dev/audit_open_issue_contracts.py` — report-only audit
- `docs/ai/issue_implementation_admission.md` — generic admission contract
- `.agents/skills/goal-autopilot/SKILL.md` — worker loop contract
