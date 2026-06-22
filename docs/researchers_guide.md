# Robot SF Researcher's Guide

This guide is the first stop for using Robot SF as a continuous social-navigation
**research engine** (epic [#3057](https://github.com/ll7/robot_sf_ll7/issues/3057)).
It walks from a research question to a published, correctly-graded result, and
links to the canonical configs, scripts, and evidence policy rather than copying
volatile details. A new researcher or agent should be able to start here without
re-reading the whole issue tree.

The single most important rule: **do not present diagnostic, smoke, fallback, or
degraded output as benchmark or paper-facing evidence.** Match the claim to the
proof. The rest of this guide is how to do that in practice.

## Audience and scope

- **In scope:** defining a question, choosing an evidence tier, authoring a
  campaign manifest, running validation, interpreting evidence grades, and
  publishing artifacts without overclaiming.
- **Out of scope:** paper-draft generation and any claim of completed
  experiments without tracked evidence. This guide is `evidence_tier: synthesis`
  documentation — it teaches the workflow; it is not itself a benchmark result.

## Step 1 — Define a research question

State the question as a falsifiable hypothesis with a discriminating outcome and
a stop rule *before* running anything. Research children of the epic follow this
shape (hypothesis, discriminating evidence, stop rule, artifact policy).

- Pick an issue archetype and required metadata:
  [`context/issue_1512_issue_archetypes.md`](context/issue_1512_issue_archetypes.md).
- Ground the question in prior work and open lanes:
  [`context/INDEX.md`](context/INDEX.md) and
  [`context/research_lane_states.md`](context/research_lane_states.md).
- Avoid duplicating a closed lane — verify live state first (see
  [`ai/ai-workflow.md`](ai/ai-workflow.md)).

## Step 2 — Choose an evidence tier

Declare the strongest claim you intend to support, then hold the run to it. The
repository uses a fixed evidence vocabulary and ladder:

- Vocabulary and fail-closed rules:
  [`context/artifact_evidence_vocabulary.md`](context/artifact_evidence_vocabulary.md).
- The grading ladder (`diagnostic-only` → `smoke` → `nominal benchmark` →
  `paper-grade`): [`maintainer_values.md`](maintainer_values.md).
- Fallback/degraded handling policy:
  [`context/issue_691_benchmark_fallback_policy.md`](context/issue_691_benchmark_fallback_policy.md).

| Tier | Supports the claim that… | Typical use |
|---|---|---|
| `diagnostic-only` | a contract/path works; no semantic performance claim | debugging, probes |
| `smoke` | a narrow path runs end-to-end | "does it run" guards |
| `nominal benchmark` | predeclared benchmark-matrix results hold | comparison runs |
| `paper-grade` | a fully reproducible, manuscript-facing claim holds | publication |

## Step 3 — Author a campaign manifest and artifact flow

Campaigns are described by a standardized manifest so runs are reproducible and
results are routed to a durable store rather than local-only files.

- Manifest and artifact-flow contract:
  [`context/issue_3062_campaign_manifest_flow.md`](context/issue_3062_campaign_manifest_flow.md).
- Result-store contract (row status + artifact provenance fields):
  [`context/issue_3076_campaign_result_store_contract.md`](context/issue_3076_campaign_result_store_contract.md)
  and the helper [`../scripts/tools/campaign_result_store.py`](../scripts/tools/campaign_result_store.py).
- Frozen scenario suite to target:
  [`context/issue_3059_research_engine_suite_v0.md`](context/issue_3059_research_engine_suite_v0.md);
  baseline planner readiness and the social-compliance metric contract are in the
  benchmark-evidence row of [`context/INDEX.md`](context/INDEX.md).

**`output/` is disposable.** Treat everything under `output/` as local scratch.
Promote only small, durable summaries to tracked evidence
([`context/evidence/README.md`](context/evidence/README.md)) or to an external
artifact store with a tracked pointer. Never cite a local-only `output/` path as
a dependency or as benchmark evidence.

## Step 4 — Run validation proportional to risk

Match the validation effort to what the change touches:

```bash
# Benchmark / campaign execution (console entry point: robot_sf_bench)
uv run robot_sf_bench --help

# Focused + parallel test suite
scripts/dev/run_tests_parallel.sh

# Pre-PR gate (use the cheaper path for low-risk branches; see ai/ai-workflow.md)
BASE_REF=origin/main scripts/dev/pr_ready_check.sh

# Docs / evidence catalog consistency (for docs and evidence changes)
uv run python scripts/validation/check_docs_proof_consistency.py --check-evidence-catalog
```

- Docs/instruction changes: diff inspection plus link/path checks.
- Runtime code: focused tests plus lint/format.
- Shared workflows, metrics, schemas, benchmark, model provenance, or
  paper-facing claims: executable proof and an explicit claim boundary.

## Step 5 — Interpret the result honestly

Classify every row before reporting. The fail-closed reading is:

| Result class | Meaning | What you may claim |
|---|---|---|
| `diagnostic` | a probe/contract ran | the mechanism is wired; no performance claim |
| `smoke` | a narrow path executed | it runs end-to-end; not a benchmark |
| `benchmark` | predeclared matrix executed and graded | the predeclared comparison, within stated seeds/CIs |
| `blocked` | a dependency/resource is missing | nothing yet; state the precise blocker |
| `non-claim` | fallback/degraded/unavailable row | explicitly *not* evidence; do not aggregate as success |

A deterministic tie-break, a fallback row, or an all-zero/all-one denominator is
**not** empirical evidence — report it as non-identifiable or non-claim. See the
benchmark-evidence-policy row in [`context/INDEX.md`](context/INDEX.md).

## Step 6 — Publish artifacts without overclaiming

- Report generation and claim boundaries: [`research_reporting.md`](research_reporting.md).
- Keep durable proof small and tracked under
  [`context/evidence/`](context/evidence/README.md); link the issue/PR that
  produced it, not a transient local file.
- Update [`../CHANGELOG.md`](../CHANGELOG.md) for user-facing changes and add a
  context note + catalog entry for any new durable surface.

## Month-one synthesis

For the current landed-vs-pending state of the research-engine roadmap (epic
[#3057](https://github.com/ll7/robot_sf_ll7/issues/3057)), see
[`context/research_month_one_synthesis_2026-06.md`](context/research_month_one_synthesis_2026-06.md).
That surface indexes completed and blocked children with their declared evidence
tiers; it makes no new benchmark claim.

## See also

- Contributor workflow and testing strategy: [`dev_guide.md`](dev_guide.md).
- AI-assistant workflow and cheaper validation paths: [`ai/ai-workflow.md`](ai/ai-workflow.md).
- Repository values, hard rules, and validation hierarchy: [`maintainer_values.md`](maintainer_values.md).
- Retrieval-first context routing: [`context/INDEX.md`](context/INDEX.md).
