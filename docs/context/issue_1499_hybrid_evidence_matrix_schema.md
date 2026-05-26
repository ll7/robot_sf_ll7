# Issue 1499 Hard-Guarded Hybrid-Learning Evidence Matrix Schema

Status: Canonical schema for the hard-guarded hybrid-learning evidence matrix.

Motivating issues:

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/1499> (this schema)
- Parent: <https://github.com/ll7/robot_sf_ll7/issues/1489> (synthesis consumer)

Component issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1470> (oracle imitation dataset)
- <https://github.com/ll7/robot_sf_ll7/issues/1472> (learned-risk model v1 Slurm campaign)
- <https://github.com/ll7/robot_sf_ll7/issues/1474> (shielded PPO repair Slurm campaign)
- <https://github.com/ll7/robot_sf_ll7/issues/1475> (bounded ORCA-residual BC smoke/nominal)
- <https://github.com/ll7/robot_sf_ll7/issues/1358> (bounded ORCA-residual learned local policy)
- <https://github.com/ll7/robot_sf_ll7/issues/1496> (additional learned component, if included)

Related policies:

- `docs/context/issue_691_benchmark_fallback_policy.md` — fail-closed fallback contract
- `docs/context/artifact_evidence_vocabulary.md` — evidence category vocabulary
- `docs/context/evidence/README.md` — evidence bundle policy
- `docs/context/issue_1054_planner_readiness_fallback_audit.md` — readiness/fallback audit template

## Purpose

Define the canonical evidence matrix schema that issue #1489 will consume after component campaigns
produce comparable outputs. The schema standardises comparison boundaries across learned components
while enforcing that hard guards remain authoritative and fallback/degraded status is never hidden.

## Scope Boundaries

- **In scope**: Schema definition, field contracts, evidence-tier vocabulary, placeholder example
  rows, non-evidence/failure-mode enumeration, and consumer rules for #1489.
- **Out of scope**: Synthesising campaign results, claiming component campaigns are complete,
  changing benchmark semantics/metrics/code/seeds/configs, paper claims, and training or Slurm
  execution.
- **Evidence authority**: This schema is an enabling workflow/docs artefact. It is not itself
  benchmark evidence or a paper-facing claim.

## Guard Authority (Hard Constraint)

Every row in the hybrid-learning evidence matrix must obey the following hard-guard contract:

1. **Hard guards are always authoritative.** No learned component may bypass, override, or silently
   disable a hard guard.
2. **Learned components are auxiliary.** A learned-risk scorer, residual policy, imitation warm
   start, or repair network may inform or adjust decisions only through declared, bounded interfaces
   that respect the hard-guard boundary.
3. **Guard veto is visible.** Every row must report whether the hard guard intervened (vetoed a
   learned proposal) and at what rate.
4. **Rejection rule.** A row that allows a learned component to bypass hard guards without an
   explicit, measurable guard-veto record is invalid for synthesis. A row that reports zero guard
   interventions without proving the guard was active is not synthesis-grade evidence.

## Evidence Matrix Schema

### Required Fields

| # | Field | Type | Description |
|---|---|---|---|
| 1 | `component` | string | Learned component identifier matching the source component issue (e.g. `learned_risk_model_v1`, `orca_residual_bc_v1`, `shielded_ppo_repair_v1`, `oracle_imitation_v1`). |
| 2 | `source_issue` | string | GitHub issue number that produced the component evidence (e.g. `#1472`, `#1474`). |
| 3 | `commit_artifact` | string | Git commit SHA plus the best available provenance pointer for the row. For synthesis-eligible rows, this must identify the exact campaign commit and durable output. For non-execution rows such as `launch_packet` or dry-run placeholders, it may instead point to the launch packet, validator output, or tracked planning artefact, but that provenance is still non-evidence. |
| 4 | `evaluation_slice` | enum | One of `not_run`, `smoke`, `nominal_sanity`, `stress_slice`, `full_matrix`. Defines the evaluation breadth of the reported row. Use `not_run` only for pre-execution placeholders, dry-runs, and launch-packet rows; `not_run` is never synthesis-eligible. |
| 5 | `guard_authority` | object | `{mechanism: string, active: boolean, veto_rate: number\|null}`. Describes which hard guard is active (e.g. `risk_guarded_ppo`, `static_margin`), whether it was enforced during evaluation, and the fraction of learned-action decisions vetoed by the guard. `veto_rate` must be in `[0.0, 1.0]` when present, and must be `null` only when `active` is `false` (invalid row). |
| 6 | `learned_component_contribution` | object | `{contribution_type: string, bound: string, active_rate: number\|null}`. Describes how the learned component contributed (e.g. `auxiliary_cost`, `bounded_residual`, `warm_start_initialisation`, `repair_decision`), the declared bound or interface constraint, and the fraction of policy decisions where the component produced a non-trivial change. `active_rate` must be in `[0.0, 1.0]` when present; `null` allowed only when the contribution mechanism was not active. |
| 7 | `intervention_fallback_rates` | object | `{guard_veto_rate: number\|null, fallback_rate: number\|null, degraded_rate: number\|null}`. Guard veto rate (same decision denominator as `guard_authority.veto_rate`), fallback rate (fraction of policy decisions routed through fallback mode), and degraded rate (fraction of planned evaluation rows/episodes skipped, partially failed, or otherwise degraded). All numeric rates must be in `[0.0, 1.0]`. All three fields must be present. `null` means unavailable, not zero, and is allowed for non-execution rows or when that rate could not be observed/collected; synthesis consumers must never coerce `null` to `0.0`. |
| 8 | `outcomes` | object | `{success_rate: number\|null, collision_rate: number\|null, near_miss_rate: number\|null, low_progress_rate: number\|null, timeout_rate: number\|null}`. Aggregate per-episode outcome fractions over the evaluation slice. Numeric rates must be in `[0.0, 1.0]`. `null` means unavailable, not zero, and is allowed for non-execution rows or when a metric was not collected. |
| 9 | `evidence_tier` | enum | One of `launch_packet`, `smoke_only`, `nominal_only`, `stress`, `full_matrix`, `degraded`, `fallback`, `failed`, `not_available`. Defines the evidence strength of the row. |
| 10 | `verdict` | enum | Consumer-facing classification: `continue` (evidence supports further investment), `revise` (evidence reveals a fixable issue), `stop` (evidence shows the approach is not viable), `insufficient_evidence` (not enough evidence to classify), `pending` (campaign has not yet produced evidence for this slice). |

### Additional Diagnostic Fields (Optional)

These fields are not required for every row but are strongly recommended when the evaluation
slice is `stress_slice` or `full_matrix`:

| # | Field | Type | Description |
|---|---|---|---|
| D1 | `comfort_exposure` | number\|null | Aggregate comfort-exposure metric over the evaluation slice. |
| D2 | `min_pedestrian_distance` | number\|null | Minimum robot-pedestrian distance observed. |
| D3 | `force_exposure_rate` | number\|null | Per-step or per-second force-exposure rate. |
| D4 | `path_efficiency` | number\|null | Ratio of straight-line distance to actual path length. |
| D5 | `mean_time_to_goal` | number\|null | Mean episode steps or simulated seconds to goal (successful episodes only). |
| D6 | `baseline_comparator` | string\|null | Identifier of the non-learning baseline used for comparison (e.g. `hybrid_rule_v3_static_margin0_waypoint2`). |
| D7 | `seed_schedule` | string\|null | Seed set identifier or seed list used for this slice. |
| D8 | `scenario_manifest` | string\|null | Reference to the scenario manifest or scenario list used. |

## Evidence Tier Definitions

| Tier | Meaning | Allowed for synthesis? | Typical source |
|---|---|---|---|
| `launch_packet` | Pre-SLURM config, validator, and stub artefacts; no campaign has run. Rows in this tier should normally use `evaluation_slice = not_run`. | **No.** Not evidence. | Launch-packet validator output. |
| `smoke_only` | A small number of episodes passed a smoke gate; no broader evaluation exists. | **No.** Diagnostic only. | Smoke gate e.g. success=1.0, collision=0.0 on a tiny slice. |
| `nominal_only` | The nominal-sanity slice passed its gate, but no stress or full-matrix evaluation was run. | **Limited.** May indicate plausibility but cannot support comparative claims. | Nominal-sanity gate output. |
| `stress` | A stress slice was evaluated; broader matrix coverage is incomplete. | **Partial.** Supports bounded claims within the stress slice only. | Stress-slice benchmark or policy-analysis run. |
| `full_matrix` | The full evaluation matrix was run with durable, reproducible artefacts. | **Yes.** Required for comparative synthesis claims. | Full-matrix benchmark run with provenance. |
| `degraded` | Some episodes or scenarios failed or were skipped; the run is incomplete. | **No.** Caveat only. | Partial campaign output with explicit degraded status. |
| `fallback` | The planner entered a fallback path instead of its intended contract. | **No.** Non-success outcome. | Fallback-triggered benchmark row. |
| `failed` | The campaign or planner run failed. | **No.** Exclusion reason. | Failed campaign output with explicit failure reason. |
| `not_available` | The planner, dependency, or runtime contract could not be satisfied. | **No.** Exclusion reason. | Availability check or fail-closed guard output. |

## Non-Evidence And Failure Modes

The following artefact classes are **not** benchmark evidence and must not appear in synthesis rows:

- **Dry-runs.** A validator or preflight that reports `status=valid` without any training or
  evaluation execution.
- **Local-only `output/` files.** Any file under `output/` without a corresponding durable artifact
  pointer (release, W&B artifact, or tracked evidence copy with checksum).
- **Launch packets.** Configs, validators, and stub files that stage a campaign but have not
  produced evaluation output.
- **Fallback rows.** Any row where `execution_mode=fallback` or `readiness_status=fallback`. These
  are diagnostic only.
- **Degraded rows.** Any row where `readiness_status=degraded` or `availability_status` is not
  `available`. These represent incomplete or partial execution.
- **Unsupported rows.** Any row that references a missing artefact, an unavailable dependency, an
  import-guarded module that could not be loaded, or a planner that could not satisfy its contract.

When a component campaign cannot produce a synthesis-eligible row, the matrix must record the
exclusion reason explicitly (failure mode, missing artefact, unavailable dependency) rather than
omitting the row or counting a non-evidence artefact as success.

## Consumer Rules For Issue #1489

1. Accept a row for synthesis only when `evidence_tier` is `full_matrix` or `stress` and `verdict`
   is `continue` or `revise`.
2. `nominal_only` rows may inform qualitative status but must not be used for comparative claims.
3. Rows with `evidence_tier` of `launch_packet`, `smoke_only`, `degraded`, `fallback`, `failed`, or
   `not_available` are excluded from synthesis.
4. Every synthesis-accepted row must have `guard_authority.active = true` and a non-null
   `guard_authority.veto_rate`.
5. If a learned component contributed non-trivially (`learned_component_contribution.active_rate > 0`)
   but the guard veto rate is zero, the synthesis must note that the guard was not exercised and
   mark the row with a caveat.
6. `outcomes` fields that are `null` must be treated as unavailable, not as zero.
7. Synthesis must not imply component campaigns are complete when rows are missing or have
   `verdict = pending` or `insufficient_evidence`.

## Placeholder Example Rows

The rows below use placeholder issue IDs and demonstrate the schema shape. They do not represent
actual campaign results and must not be cited as paper claims.

| component | source_issue | commit_artifact | evaluation_slice | guard_authority | learned_component_contribution | intervention_fallback_rates | outcomes | evidence_tier | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `learned_risk_model_v1` | #9991 | `abc123def`, `wandb://...` | `stress_slice` | `{mechanism: risk_guarded_ppo, active: true, veto_rate: 0.12}` | `{contribution_type: auxiliary_cost, bound: "cost_weight in [0,1]", active_rate: 0.85}` | `{guard_veto_rate: 0.12, fallback_rate: 0.03, degraded_rate: null}` | `{success_rate: 0.72, collision_rate: 0.04, near_miss_rate: 0.08, low_progress_rate: 0.05, timeout_rate: 0.11}` | `stress` | `pending` |
| `orca_residual_bc_v1` | #9992 | `def456abc`, `wandb://...` | `nominal_sanity` | `{mechanism: static_margin, active: true, veto_rate: 0.08}` | `{contribution_type: bounded_residual, bound: "clip(residual, -0.5, 0.5)", active_rate: 0.60}` | `{guard_veto_rate: 0.08, fallback_rate: 0.01, degraded_rate: null}` | `{success_rate: 0.85, collision_rate: 0.02, near_miss_rate: 0.03, low_progress_rate: null, timeout_rate: 0.10}` | `nominal_only` | `pending` |
| `shielded_ppo_repair_v1` | #9993 | `ghi789jkl`, `wandb://...` | `smoke` | `{mechanism: risk_guarded_ppo, active: true, veto_rate: 0.25}` | `{contribution_type: repair_decision, bound: "collision_weight=-20.0 only", active_rate: 0.40}` | `{guard_veto_rate: 0.25, fallback_rate: 0.05, degraded_rate: 0.02}` | `{success_rate: 0.45, collision_rate: 0.15, near_miss_rate: null, low_progress_rate: 0.20, timeout_rate: 0.20}` | `smoke_only` | `pending` |
| `oracle_imitation_v1` | #9994 | `mno012pqr`, `docs/context/issue_9994_example_launch_packet.md` | `not_run` | `{mechanism: static_margin, active: false, veto_rate: null}` | `{contribution_type: warm_start_initialisation, bound: "oracle dataset only", active_rate: null}` | `{guard_veto_rate: null, fallback_rate: null, degraded_rate: null}` | `{success_rate: null, collision_rate: null, near_miss_rate: null, low_progress_rate: null, timeout_rate: null}` | `launch_packet` | `pending` |
| `example_fallback_row` | #9999 | `N/A` | `full_matrix` | `{mechanism: static_margin, active: true, veto_rate: 1.0}` | `{contribution_type: bounded_residual, bound: "clip(residual, -0.5, 0.5)", active_rate: null}` | `{guard_veto_rate: 1.0, fallback_rate: 1.0, degraded_rate: null}` | `{success_rate: 0.0, collision_rate: 0.0, near_miss_rate: null, low_progress_rate: null, timeout_rate: 1.0}` | `fallback` | `stop` |

### Example Row Interpretation Notes

- **Row 1 (`learned_risk_model_v1`)**: Guard active with 12% veto rate; auxiliary cost contributed
  in 85% of steps; stress-slice evidence. Not yet synthesised (`verdict=pending`).
- **Row 2 (`orca_residual_bc_v1`)**: Guard active; bounded residual active in 60% of steps;
  nominal-sanity evidence. Stress or full-matrix evidence is missing; this row alone is
  insufficient for comparative claims.
- **Row 3 (`shielded_ppo_repair_v1`)**: Guard active with high veto rate (25%); repair contribution
  active in 40% of steps; smoke-only evidence. Not synthesis-eligible.
- **Row 4 (`oracle_imitation_v1`)**: Guard not active; `evaluation_slice=not_run`; no evaluation
  output produced; launch-packet tier. **Invalid for synthesis.** Illustrates a row where
  `guard_authority.active = false` and all outcome rates are null — the consumer must reject this
  row.
- **Row 5 (`example_fallback_row`)**: Guard vetoed 100% of decisions; fallback took over entirely;
  zero success. **Must be excluded from synthesis.** Illustrates the fail-closed fallback contract
  from `docs/context/issue_691_benchmark_fallback_policy.md`.

## Machine-Readable Validation

Issue [#1515](https://github.com/ll7/robot_sf_ll7/issues/1515) adds a read-only validator for
future matrix inputs:

```bash
uv run python scripts/validation/validate_hybrid_evidence_matrix.py \
  --input tests/fixtures/hybrid_evidence_matrix/v1/valid_rows.yaml
```

Validation surface:

- accepts a single row, a list of rows, or a mapping with `rows: [...]` from YAML or JSON input,
- validates required fields, enums, nullability, and rate bounds,
- checks `commit_artifact` as a comma- or newline-separated string containing a git SHA token plus
  one or more provenance tokens,
- defaults to **format-only** `commit_artifact` validation so local row checks stay lightweight and
  deterministic even when a SHA-looking token has not yet been proven against repository history,
- supports an opt-in **history-backed** proof path via
  `--check-git-history`, which additionally requires each git SHA token to resolve in the selected
  local repository history before a row can pass,
- requires repository-local provenance tokens to be repository-root-relative, present in the
  checkout, and outside `output/`,
- enforces fail-closed fallback/degraded semantics for success-like evidence tiers,
- verifies `guard_authority.veto_rate` matches
  `intervention_fallback_rates.guard_veto_rate` for synthesis-candidate rows,
- preserves launch-packet / `not_run` rows as valid non-synthesis inputs instead of forcing them to
  look like execution evidence.

The validator is intentionally conservative: it does not synthesize results, resolve remote artifact
availability, or upgrade local `output/` paths into durable evidence.

Use the default validator path while drafting rows, launch packets, and local schema fixes. Before
issue [#1489](https://github.com/ll7/robot_sf_ll7/issues/1489) treats a `stress` or `full_matrix`
row as synthesis input, rerun the same file with `--check-git-history` from a checkout whose git
history includes the cited commit(s). That stricter pass separates shape validation from local
repository-history proof without downgrading launch-packet rows or durable remote artifact pointers.

## Known Gaps

1. Seed-schedule and scenario-manifest references remain optional diagnostic fields; stricter
   provenance requirements may be added after component campaigns demonstrate what is feasible.
2. The validator proves row-level contract compliance, but it does not interpret benchmark outcomes
   or replace the synthesis consumer planned in #1489.

## Validation

Validation of the schema itself is through review against the Issue #1499 acceptance criteria plus
the #1515 validator/tests:

- [x] All required fields present: `component`, `source_issue`, `commit_artifact`,
  `evaluation_slice`, `guard_authority`, `learned_component_contribution`,
  `intervention_fallback_rates`, `outcomes`, `evidence_tier`, `verdict`.
- [x] Guard authority is enforced: `guard_authority.active` and `guard_authority.veto_rate` are
  required; rows with `active=false` are invalid for synthesis.
- [x] Fallback/degraded status is explicitly represented in `evidence_tier` and
  `intervention_fallback_rates`.
- [x] Non-evidence/failure modes are enumerated: dry-runs, local-only `output/` files, launch
  packets, fallback, degraded, failed, not-available, and unsupported rows.
- [x] Placeholder example rows use only placeholder issue IDs (no paper claims).
- [x] Consumer rules define when a row is synthesis-eligible.
- [x] Links to canonical policies: fallback policy (#691), evidence vocabulary, readiness/fallback
  audit (#1054).
- [x] Repository-root-relative paths only.
- [x] Machine-readable row validator rejects invalid enums, nullability, provenance, and guard-veto
  divergence before synthesis.

Run the targeted validator proof and the standard readiness gate:

```bash
uv run pytest tests/benchmark/test_hybrid_evidence_matrix.py
uv run python scripts/validation/validate_hybrid_evidence_matrix.py \
  --input tests/fixtures/hybrid_evidence_matrix/v1/valid_rows.yaml
uv run python scripts/validation/validate_hybrid_evidence_matrix.py \
  --input path/to/hybrid_evidence_rows.yaml \
  --check-git-history
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```
