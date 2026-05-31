# Open Issues Training Split Audit 2026-05-30

Related workflow: [issue_713_batch_first_issue_workflow.md](issue_713_batch_first_issue_workflow.md)
Related policy: [issue_1512_issue_archetypes.md](issue_1512_issue_archetypes.md)

Status note: the issue-state tables below are a 2026-05-30 routing snapshot. Later state-label
cleanup superseded some `state:ready` queue rows, so do not use this historical audit as the
current implementation queue. For current queue routing, use
[issue_1776_state_label_routing.md](issue_1776_state_label_routing.md) and live GitHub issue
labels.

## Goal

Audit the 55 open non-PR GitHub issues as of 2026-05-30 and keep training-related
work clearly separated from non-training work. This note is a routing and issue-hygiene
handoff, not an implementation result.

User execution rule for training work:

- Training jobs expected to finish within 1 hour may run locally.
- Longer training jobs should be queued on the SLURM cluster.

Local machine caveat: `local.machine.md` on `imech156-u` says GPU jobs are allowed in tmux, but
SLURM submission is not available from this host. SLURM issues therefore remain classified as
SLURM-needed, but were not submitted in this pass.

## Evidence Sources

- Live REST issue inventory from `gh api repos/ll7/robot_sf_ll7/issues?state=open&per_page=100`.
- Live REST issue comments for open issues with comments.
- Local issue-template audit through `scripts/tools/issue_template_audit.py`.
- Local-only delegated worker artifacts were inspected during drafting, but they live under the
  gitignored `.git/codex-agent-runs/` cache and are not durable review evidence. The durable
  evidence for this note is the live issue inventory, issue comments, and local template audit
  listed above.
- One Copilot worker route failed because of the GitHub Copilot weekly rate limit; that failure was
  not used as an issue-data finding.

## Applied Issue-Hygiene Updates

The following additive label updates were applied through REST. No issue bodies, Project fields,
PRs, or remote branches were changed.

| Issue | Added labels | Reason |
|---|---|---|
| Issue #1676 | `type:benchmark`, `state:ready`, `evidence:proposal`, `resource:local` | Local benchmark/research scenario slice with no external blocker. |
| Issue #1675 | `type:workflow`, `state:ready`, `evidence:proposal`, `resource:local` | Local planner-interface design/prototype issue. |
| Issue #1674 | `type:analysis`, `state:ready`, `evidence:proposal`, `resource:local` | Local diagnostic analysis issue. |
| Issue #1653 | `type:workflow`, `state:ready`, `evidence:proposal`, `resource:local` | CI/test-runtime workflow issue with measurement-first comment. |

## Training-Involved Issues

These issues involve learned policies, training data, model artifacts, or training-campaign
decisions. Keep them separate from pure benchmark, CARLA, visualization, and data-staging work.

### Learned-Policy Manifest Pointers

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1686>

This manifest-specific overlay does not run training, move checkpoints, upload artifacts, or make
benchmark claims. It identifies current training lanes whose future checkpoints, normalizers,
datasets, or adapter configs should cite the learned-policy manifest fields in
[artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md).

| Lane | Current source | Split/provenance implication |
| --- | --- | --- |
| learned risk model v1 | `configs/training/learned_risk_model_issue_1395_launch_packet.yaml` and [issue_1395_learned_risk_launch_packet.md](issue_1395_learned_risk_launch_packet.md) | Manifest must keep hard guards authoritative, preserve trace-contract checksums, and replace `pending` durable aliases before benchmark eligibility. |
| shielded PPO repair | `configs/training/shielded_ppo_issue_1396_launch_packet.yaml` and [issue_1396_shielded_ppo_launch_packet.md](issue_1396_shielded_ppo_launch_packet.md) | Manifest must state the PPO checkpoint lineage, repair config, comparison-freeze checksum, and guard/fallback policy. |
| oracle imitation dataset | `configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml` and [issue_1397_oracle_imitation_launch_packet.md](issue_1397_oracle_imitation_launch_packet.md) | Manifest must cite the dataset split contract, durable dataset artifacts, and whether the resulting policy is training-only/oracle or deployable. |
| ORCA residual BC | `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml` and [issue_1428_orca_residual_lineage.md](issue_1428_orca_residual_lineage.md) | Manifest must separate base ORCA behavior, residual bounds, candidate config, diagnostic checksums, and guarded fallback behavior. |
| LiDAR PPO MLP | `configs/training/lidar/lidar_learned_policy_launch_packet_issue_1615.yaml` and [issue_1615_lidar_learned_policy_plan.md](issue_1615_lidar_learned_policy_plan.md) | Manifest must state the LiDAR observation schema, learned normalizer fit split if any, and cross-observation benchmark eligibility boundary. |

If a lane lacks a durable artifact URI, checksum, training config, training commit, observation
schema, action schema, normalizer URI or `not_required` declaration, license/access note, split
contract, or benchmark eligibility verdict, it may remain a launch packet or research-only record
but must not be promoted as a learned-policy benchmark row. Worktree-local `output/` paths may
describe exploratory runs or hydration targets only. They are not durable learned-policy artifacts.

### Local <=1h Or Analysis-Only

These should not submit long training from this host. They are local desk research, preflight,
analysis, or synthesis unless a later issue body adds a concrete longer run.

| Issue | Current status | Improvement |
|---|---|---|
| Issue #1620 | External learned-policy candidate ranking, `resource:external-data`. | Keep as analysis-only; record candidate acceptance/rejection criteria before opening implementation children. |
| Issue #1621 | Diffusion-policy feasibility, `resource:external-data`. | Keep external-code/data provenance explicit; do not import code without a follow-up issue. |
| Issue #1622 | Decision Transformer baseline assessment, labeled `training`. | Clarify that this is analysis-first, not a training job, unless a bounded run is later specified. |
| Issue #1623 | World-model feasibility, `resource:external-data`. | Keep external-data dependency visible and separate from local trainability. |
| Issue #1624 | Hybrid-learning architecture synthesis. | Use this only after component assessments; avoid treating it as execution evidence. |
| Issue #1625 | Learned arbitration analysis. | Define whether outputs are a design recommendation or an implementation issue. |
| Issue #1626 | Foundation-model readiness analysis. | Keep as low-priority analysis unless a concrete adapter/preflight is split out. |
| Issue #1627 | Learned-policy transfer benchmark definition. | Keep benchmark contract separate from any future training campaign. |
| Issue #1628 | Actuation-aware learned navigation analysis. | Require AMV actuation caveats before any training scope. |
| Issue #1629 | Latency-aware learned navigation safety analysis. | Add an explicit validation target before selecting implementation. |
| Issue #1505 | Predictive-v2 preflight, blocked by #1490. | Do not run until #1490 is revised or closed after #1543 negative evidence. |
| Issue #1507 | Predictive-v2 analysis, blocked by #1490. | Keep local-only analysis after any accepted parent rerun. |
| Issue #1489 | Hybrid-learning synthesis, blocked on component campaigns. | Keep local synthesis only; do not claim new evidence. |

### SLURM Or Long Training Needed

These are not appropriate for direct local execution under the user's 1-hour rule unless narrowed to
a smoke/preflight issue.

| Issue | Current status | Improvement |
|---|---|---|
| Issue #1470 | Oracle imitation dataset collection, `resource:slurm`, blocked. | Revalidate packet and submit from a SLURM-capable host; this gates #1496. |
| Issue #1472 | Learned risk model v1 campaign, `resource:slurm`, launch-packet evidence. | Refresh exact config/commit/artifact URI before submission. |
| Issue #1474 | Shielded PPO repair SLURM campaign, launch-packet evidence. | Keep latest comment's not-submitted state authoritative until queued. |
| Issue #1475 | ORCA-residual BC smoke/nominal lineage job, launch-packet evidence. | Confirm artifact URIs before SLURM submission. |
| Issue #1496 | Oracle imitation warm-start benchmark, blocked by #1470. | Do not start until dataset/checksum/split evidence from #1470 exists. |
| Issue #1490 | Predictive-v2 same-seed comparison parent, `resource:slurm`, blocked. | Maintainer must revise, narrow, or close after #1543 negative audit. |
| Issue #1506 | Predictive-v2 four-way training matrix, blocked by #1490. | Keep blocked until parent decision. |
| Issue #1358 | ORCA-residual learned local policy parent, `resource:slurm`, blocked. | Keep as parent coordination; child #1475 should provide next evidence. |
| Issue #1108 | BC warm-start PPO artifact rescue, `state:needs-artifact-promotion`. | Add a stale-review deadline or decide rerun vs inconclusive close. |

## Non-Training Issues

These do not involve model training, even when they are benchmark or SLURM/resource-heavy.

### Ready Or Local Next

| Issue | Current status | Improvement |
|---|---|---|
| Issue #1676 | Ready local benchmark scenario slice. | Keep comfort metrics and interpretation limits explicit. |
| Issue #1675 | Ready local planner-interface prototype. | Start with a deterministic fixture before benchmark claims. |
| Issue #1674 | Ready local diagnostic analysis. | Define expected topology diagnostics before running experiments. |
| Issue #1653 | Ready CI/test-runtime workflow. | First PR should produce baseline timing report before optimizations. |
| Issue #1646 | Epic for analysis/visualization workbench. | Do not implement directly; split child issues. |
| Issue #1638 | Baseline config migration, needs artifact promotion. | Identify durable model registry IDs before changing configs. |
| Issue #1612 | Observation-track architecture, `decision-required`. | Resolve metadata/schema placement before LiDAR work expands. |
| Issue #1611 | LiDAR benchmark epic. | Keep as epic; route concrete work through child issues. |
| Issue #1610 | Planner criticality analysis, ready. | Use additive artifacts and avoid changing scenario semantics. |
| Issue #1609 | Seed-sensitive mechanism synthesis. | Keep blocked on/after #1608 evidence. |
| Issue #1608 | Seed sensitivity analysis, ready. | Add `resource:local` in a future label-sync pass if desired. |

### Decision Required

| Issue | Blocker | Improvement |
|---|---|---|
| Issue #1582 | AMV metadata contract decision. | Accept or override the conservative contract proposed in the issue comment. |
| Issue #1604 | Ego-up renderer scope decision. | Choose ego-up-only first versus a broader camera-mode refactor. |
| Issue #1606 | Placeholder metric/artifact retirement policy. | Prefer compatibility aliases unless a maintainer wants a breaking cleanup. |
| Issue #1612 | Observation-track architecture. | Resolve track metadata and aggregation boundaries before implementation. |

### Blocked By External Data, CARLA, Or SLURM Benchmark Execution

| Issue group | Issues | Improvement |
|---|---|---|
| AMV actuation/provenance | #1585, #1559, #1586, #1577, #1570 | Issue #1585 and #1582 are the useful unblockers; avoid running paper-facing AMV work before provenance and metadata decisions. |
| S20/S30 archive | Issue #1554 | Needs actual S20/S30 runs or durable artifact recovery, likely SLURM. |
| CARLA parity | #1508, #1509, #1510, #1511, #1491 | Needs a CARLA-capable host; keep non-CARLA machines in audit-only mode. |
| Adversarial comparison | #1488, #1502, #1503 | Issue #1502 needs SLURM execution; #1503 is analysis only after #1502. |
| SocNavBench ETH | #1498, #1134 | Issue #1498 stages the source asset; #1134 should wait for it. Cross-linking these in issue bodies would reduce duplicate work. |
| Stanford Drone Dataset | #1497, #1126 | Issue #1497 stages licensed annotations; #1126 should wait for it. Cross-linking these in issue bodies would reduce duplicate work. |
| SocNavBench pipeline assets | Issue #1456 | Remains blocked on external assets and should not be treated as locally implementable. |

## Recommended Batch Order

1. Resolve decision-required blockers: #1582, #1604, #1606, #1612, then #1490.
2. Run local-ready issue work: #1653, #1674, #1675, #1676, #1608, #1610, #1638.
3. Keep training-analysis issues local and bounded: #1620-#1629, #1505, #1507, #1489.
4. Queue long training only from a SLURM-capable host: #1470, #1472, #1474, #1475, #1496, #1506, #1358, #1108.
5. Keep external-data and CARLA blockers explicit: #1498/#1134, #1497/#1126, #1508-#1511/#1491, #1456.

## Template And Metadata Notes

The local template audit reports many missing sections because older issues often place the YAML
metadata block under `## Goal / Problem` rather than a dedicated `## Archetype Metadata` heading.
Do not blindly append placeholder sections to all issues. Prefer targeted issue-body repair only
when the issue's intent is already clear and the edit reduces routing ambiguity.

Safe future hygiene candidates:

- Cross-link #1498 with #1134 and #1497 with #1126.
- Add a stale-review decision date to #1108.
- Add concise blocker-summary comments to #1505, #1506, and #1507 after #1490 is decided.
- Consider a narrow label-sync pass for obvious `resource:local` omissions, keeping Project #5
  updates separate.
