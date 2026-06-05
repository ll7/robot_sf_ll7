# Issue #2263 Mechanism Activation Report Fields

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2263>
Date: 2026-06-05
Status: current trace-mechanism reporting contract update.

## Goal

Recent mechanism analyses could classify outcomes more quickly if every trace-mechanism report
stated whether the proposed mechanism activated, changed command arbitration, changed the observed
outcome, and what likely failure reason remained. This note records the narrow reporting contract
added for future reports; it does not backfill historical reports or rerun benchmarks.

## Canonical Surface

The canonical schema surface is:

- `.agents/skills/schemas/trace_mechanism_summary.v1.yaml`
- `.agents/skills/trace-mechanism-review/SKILL.md`
- [issue_2236_trace_mechanism_evidence_rubric.md](issue_2236_trace_mechanism_evidence_rubric.md)

The compatibility skill mirrors under `.codex/` and `.opencode/` consume the same tracked
`.agents` source in this checkout, so the tracked change is made in `.agents`.

## Field Contract

Future trace-mechanism summaries should include:

```yaml
mechanism_activation:
  activated: true | false | unknown
  activation_count: integer | unknown
  changed_command_source: true | false | unknown
  changed_outcome: true | false | unknown
  likely_failure_reason: string
```

Use `unknown` when the durable trace or compact summary does not preserve the needed signal. Do not
infer activation count, command-source change, or outcome change from aggregate terminal metrics
alone.

## Interpretation Boundary

The block is an interpretation aid, not a claim upgrade. Activation evidence remains diagnostic
unless the same report ties it to benchmark-valid row status, controlled baseline/intervention
identity, durable artifacts, and a measured outcome change.

This distinction matters for recent results:

- Issue #2259 showed command clipping improved without success movement, so the recommended status
  remains `keep_diagnostic_only` until route-progress and command-feasibility traces exist.
- Issue #2261 found terminal parity plus missing activation evidence, so static recentering remains
  a `slice_local_boundary` diagnostic rather than a transferable mechanism claim.

## Example

The compact example summary is tracked at:

- `docs/context/evidence/issue_2263_mechanism_activation_fields_2026-06-05/summary.json`

It demonstrates how the block can represent both known activation and missing telemetry without
promoting either case to benchmark or paper-facing evidence.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_2263_mechanism_activation_fields_2026-06-05/summary.json
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
