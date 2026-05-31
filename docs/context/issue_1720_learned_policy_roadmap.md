# Issue #1720 Learned-Policy Roadmap And Issue Routing

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1720>

Date: 2026-05-30

## Goal

Consolidate learned-policy work into one routing surface so future Robot SF agents distinguish
adapter/readiness work, registry work, deterministic fixtures, candidate-policy research, smoke
candidates, and training candidates before touching individual learned-policy issues.

This roadmap is synthesis only. It does not train a policy, import external assets, promote a
checkpoint, create a benchmark row, or claim learned-policy benchmark success.

## Foundation Baseline

| Issue | Current status | Baseline contribution | Current boundary |
|---|---|---|---|
| Issue #1618 | closed | `docs/context/issue_1618_learned_policy_adapter_interface.md` defines the `LearnedLocalPolicyAdapter` contract, required metadata, status semantics, and fail-closed behavior. | Adapter metadata is a prerequisite, not benchmark evidence. |
| Issue #1657 | closed | `docs/context/policy_search/learned_policy_registry.md` defines the learned-policy registry schema, implemented/staged/monitor/rejected statuses, and candidate intake rules. | Registry entries are planning metadata, not benchmark evidence. |
| Issue #1675 | closed | `docs/context/issue_1675_learned_risk_surface_interface.md` defines a deterministic local risk-surface producer/consumer smoke path. | The fixture proves an interface shape, not learned risk performance. |
| Issue #1685 | closed | `docs/context/issue_1685_dummy_learned_policy_adapter.md` adds a deterministic dummy adapter fixture for the learned-policy boundary. | Fixture-only evidence; not a checkpoint, registry promotion, or benchmark claim. |

## Routing Stack

Use this order when evaluating learned-policy work:

1. **Adapter contract:** does the candidate satisfy Issue #1618 observation, action, checkpoint,
   determinism, diagnostics, and fail-closed metadata?
2. **Registry status:** is the candidate represented in
   `docs/context/policy_search/learned_policy_registry.md` or the runnable
   `docs/context/policy_search/candidate_registry.yaml` with clear provenance?
3. **Fixture or interface proof:** if no real model is ready, can a deterministic fixture prove the
   adapter or surface shape without benchmark claims?
4. **Artifact manifest:** is any checkpoint, dataset, normalizer, or raw campaign archive durable
   enough to rerun or review?
5. **Candidate-policy research:** if source/checkpoint/dataset fit is unclear, keep the work in an
   analysis-only issue.
6. **Smoke candidate:** open only after a concrete policy, source or local config, adapter
   contract, and smallest falsification command are known.
7. **Training candidate:** open only after durable data/provenance, stop conditions, evaluation
   config, and benchmark non-claim boundaries are explicit.

Fallback, degraded, unavailable, or guard-dominated learned-policy execution remains caveated under
`docs/context/issue_691_benchmark_fallback_policy.md`.

## Classification Vocabulary

- `analysis_only`: synthesis, feasibility, readiness, ranking, or architecture work; no execution
  target yet.
- `adapter_needed`: a concrete policy or family is chosen, but Robot SF observation/action/checkpoint
  adapter work is the next blocker.
- `dataset_needed`: a plausible policy path exists, but durable dataset or trace provenance blocks
  execution.
- `smoke_candidate`: a bounded runnable smoke exists or can be created without training.
- `training_candidate`: training is the next justified action and has durable data, config, stop
  gates, and evaluation boundaries.
- `not_recommended`: the path is rejected or monitor-only for the current Robot SF contract.

## Issue Routing

| Issue | Title | Category | Priority/readiness recommendation | Reason |
|---:|---|---|---|---|
| Issue #1620 | Rank external learned local-navigation policy candidates | `analysis_only` | Keep low priority until adapter/registry consolidation is consumed. | Broad external shortlist work needs source/license/checkpoint checks before any implementation issue. |
| Issue #1621 | Assess diffusion-policy local navigation feasibility | `analysis_only` | Keep low priority / monitor-first. | Visual, topological, or iterative diffusion assumptions may not map fairly to local state/action planning. |
| Issue #1622 | Assess Decision Transformer local-navigation baseline | `analysis_only` | Keep medium priority as a dataset-provenance preflight. | It may become `dataset_needed`, but only after durable offline trajectory sources are inventoried. |
| Issue #1623 | Assess world-model navigation feasibility | `analysis_only` | Keep low-to-medium priority / feasibility only. | Training and compute risk are high; prior Dreamer-style work needs reconciliation before follow-up. |
| Issue #1624 | Propose unified hybrid-learning navigation architecture | `analysis_only` | Keep medium priority as synthesis after this roadmap. | Architecture work should map components without implying performance evidence. |
| Issue #1625 | Assess learned planner arbitration policy | `analysis_only` | Keep medium priority as preflight. | Arbitration may be promising, but it first needs inference-available features and leakage-free labels. |
| Issue #1626 | Assess foundation-model readiness for local navigation | `analysis_only` | Keep low priority / monitor-only unless split into interface gaps. | VLA and multimodal policies require missing observation/task abstractions and should not become model-integration work by default. |
| Issue #1627 | Define learned-policy transfer benchmark | `analysis_only` | Keep medium priority after Issue #1620 identifies concrete candidates. | Transfer criteria are useful, but execution should wait for a candidate with source/checkpoint proof. |
| Issue #1628 | Assess actuation-aware learned navigation for AMVs | `analysis_only` | Keep highest among Issue #1620-Issue #1629 for local synthesis. | AMV actuation constraints are the most Robot SF-specific learned-policy hypothesis, but the issue still asks for analysis and scenario design only. |
| Issue #1629 | Assess latency-aware learned navigation safety | `analysis_only` | Keep high for local synthesis after timing hooks are inventoried. | Latency is an AMV-relevant safety stressor, but the current issue is feasibility/design, not benchmark execution. |

No issue in Issue #1620-Issue #1629 currently qualifies as `adapter_needed`, `dataset_needed`,
`smoke_candidate`, or `training_candidate` as written. Promote one only by opening a follow-up that
names the exact policy or interface, durable inputs, first command, expected output, and stop
condition.

## Recommended Queue Policy

1. Treat Issue #1720 as the retrieval entrypoint before selecting any learned-policy issue.
2. Keep Issue #1628 and Issue #1629 as the strongest near-term local synthesis topics because they are
   AMV-specific.
3. Route Issue #1622 to dataset provenance before training discussion.
4. Route Issue #1627 after Issue #1620 or the registry names at least one imported-policy candidate with source
   and checkpoint proof.
5. Keep Issue #1621, Issue #1623, and Issue #1626 monitor-first until a fair Robot SF observation/action contract is
   evident.
6. Do not open new learned-policy idea issues until a roadmap row identifies a concrete follow-up
   with a proof path.

## Follow-Up Templates

Use these shapes when splitting a roadmap row into implementation work.

### Adapter Follow-Up

- Policy/family:
- Source or local implementation:
- Checkpoint/artifact status:
- Observation contract:
- Action contract:
- Fail-closed conditions:
- First smoke command:
- Expected compact evidence:

### Dataset Follow-Up

- Dataset or trace source:
- Durable artifact pointer:
- License/provenance boundary:
- Required fields:
- Missing fields:
- Minimal validation command:
- Stop condition:

### Training Follow-Up

- Training config path:
- Dataset/artifact pointer:
- Hardware/runtime expectation:
- Evaluation config path:
- Success and failure gates:
- Non-claim boundary:
- Artifact promotion plan:

## Validation

Validation for this roadmap:

```bash
for issue in 1618 1657 1675 1685 1620 1621 1622 1623 1624 1625 1626 1627 1628 1629; do
  gh issue view "$issue" --json number,title,state,labels,url
done
rg -n "Issue #1618|Issue #1657|Issue #1675|learned-policy registry|risk-surface" docs/context
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
