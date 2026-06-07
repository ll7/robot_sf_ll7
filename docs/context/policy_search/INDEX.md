# Policy Search Retrieval Index

Status: current retrieval surface for policy-search context.

Use this file before reading the full policy-search README or the complete candidate registry.
It points to the smallest authority surface for common agent questions.

## First Stops

| Need | Read first | Why |
| --- | --- | --- |
| Current runnable candidate list | `candidate_registry.yaml` | Canonical machine-readable list of concrete Robot SF candidates with config pointers. |
| Candidate lifecycle buckets | `candidate_registry_summary.md` | Compact routing summary for active, diagnostic, learned, SLURM, monitor, and historical lanes. |
| Local execution workflow | `contracts/agent_runbook.md` | How to pick and run candidate stages without overclaiming evidence. |
| Promotion gates | [promotion_gates.md](contracts/promotion_gates.md), [promotion_gates.yaml](../../../configs/policy_search/promotion_gates.yaml) | Human and structured gate definitions. |
| Learned-policy intake | `learned_policy_registry.md`, `contracts/learned_local_policy_eligibility.md` | Learned-policy status, adapter eligibility, and leakage checks. |
| External learned-policy rejects or monitor lanes | `reject_monitor_registry.md` | Negative and source-side-first decisions that should not re-enter runnable work. |
| SLURM handoffs | `SLURM/todo.md`, `../slurm_issue_batch_status_2026-05-21.md` | Local handoff index plus execution-status ledger for training or long-run work. |
| Current portfolio interpretation | `portfolio_overview_2026-05-05.md` | Broad snapshot of leaders, gaps, and reproduction commands from the May 2026 sweep. |
| H500/latest policy-search analysis | `reports/2026-05-05_full_matrix_h500_analysis.md` | Current h500 leader analysis and caveats. |
| Component ablation pilot | `../issue_2104_component_ablation_pilot.md` | Retrospective grouped-component pilot for leading hybrid candidates; diagnostic only, not one-factor proof. |
| One-factor component manifest | `../issue_2170_one_factor_hybrid_component_manifest.md` | Pre-execution contract for the next one-factor hybrid component ablation slice. |
| Worker-scaling diagnostics | `../issue_2172_benchmark_worker_scaling.md`, `../issue_2302_benchmark_worker_scaling.md`, `../issue_2304_benchmark_worker_scaling_stress.md` | Local runtime profile helper and compact evidence for policy-search worker scaling, including nominal-sanity and stress-slice 1/2/4/6-worker continuations. |
| One-factor ablation pilot | `../issue_2174_one_factor_ablation_pilot.md` | First executable one-factor comparison and runner for the #2170 manifest. |
| Remaining one-factor h80 comparisons | `../issue_2176_remaining_one_factor_h80.md` | Remaining selected h80 component comparisons; diagnostic-only with a partial selector row. |
| Selector ORCA-extra h80 rerun | `../issue_2178_selector_orca_extra_h80.md` | Corrected selector-only comparison after proving `rvo2`; diagnostic-only h80 evidence. |
| One-factor h500 component run | `../issue_2180_one_factor_h500.md` | Complete h500 execution of the #2170 manifest; recentering is the clearest positive component signal. |
| Component effect synthesis | `../issue_2182_component_effect_synthesis.md` | Closeout synthesis for #2104 component classifications and acceptance mapping. |
| Planner mechanism cards | `../issue_2453_planner_mechanism_cards.md` | Active planner mechanism claims, activation signals, positive/negative evidence, transfer status, and next-proof boundaries. |

## Lifecycle Routing

Use these routing labels when summarizing or adding candidates. They are descriptive metadata for
agent retrieval; they do not change benchmark claims by themselves.

| Lifecycle | Meaning | Primary source |
| --- | --- | --- |
| `active_runnable` | Candidate has a local Robot SF candidate config and can be routed through policy-search stages. | `candidate_registry.yaml` rows with `candidate_config_path`. |
| `diagnostic_only` | Candidate exists to probe a mechanism or sensitivity; do not treat it as a promoted planner without a separate evidence decision. | Proxemic diagnostic rows and diagnostic reports. |
| `learned_policy_intake` | Learned or learned-style policy needs eligibility, adapter, artifact, and fail-closed checks before benchmark claims. | `learned_policy_registry.md` and eligibility contract. |
| `slurm_handoff_only` | Candidate or campaign is documented locally but requires SLURM/Auxme or durable artifacts before execution. | `SLURM/` notes and launch packets. |
| `monitor_or_source_first` | External family is useful for research monitoring but not runnable as a Robot SF candidate yet. | `reject_monitor_registry.md` and source-assessment notes. |
| `historical_report` | Report or candidate is useful provenance but should not be read as the current best route without checking newer summaries. | `reports/`, `validation/`, and `experiment_ledger.md`. |

## Claim Eligibility

- `benchmark_candidate`: executable candidate with a concrete config and matching stage evidence.
- `diagnostic_only`: mechanism probe or sensitivity profile; useful for hypotheses, not headline
  ranking.
- `smoke_only`: local smoke or launch-packet proof only; not full benchmark evidence.
- `not_benchmark_evidence`: planning, source-harness, or metadata proof that cannot support a
  benchmark claim.
- `blocked_or_monitor`: source-side, missing-artifact, unsupported-runtime, or rejected adapter
  lane.

When claim eligibility is uncertain, use the weaker category and point to the note that would
change the status.

## Reading Rules

- Do not treat all rows in `candidate_registry.yaml` as equally active. Start with
  `candidate_registry_summary.md` and only then open the specific candidate config or report.
- Do not promote a learned policy from `learned_policy_registry.md` or `reject_monitor_registry.md`
  into `candidate_registry.yaml` without a runnable config, adapter contract, and proof path.
- Do not treat fallback, degraded, or missing-runtime execution as benchmark success.
- Treat local `output/` files as scratch unless a tracked evidence note or durable artifact pointer
  explicitly promotes the result.
