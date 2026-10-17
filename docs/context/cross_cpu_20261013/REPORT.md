# Cross-CPU diagnostic reconstruction, 2026-10-13

**Diagnostic-only, development seeds 1001–1030.** This study compares two CPU/software contexts on ten fixed scenarios. It is a reconstruction on current-main source, not an authenticated replay of the missing historical producer. CPU, kernel and glibc remain confounded. The fixed scenario set includes the declared infeasible narrow doorway; its coordinates are retained. There is no release, claim, manuscript, calibration, universal-portability or safety admission.

## Original numbers versus fresh measurement

The original column is reported by [issue #10211, comment 6056176084](https://github.com/ll7/robot_sf_ll7/issues/10211#issuecomment-6056176084); its original script and raw archive were not independently recovered. The later script-name comment supplies no producing-script proof. The rerun column is derived record by record from preserved native JSONL, with a complete coordinate ledger.

| Measure | Original comment | This rerun |
|---|---:|---:|
| Default learned scientific records different | 548/600 | 537/600 |
| Default outcome differences, all arms | 17/1,500 | 12/1,500 |
| Default control scientific records identical | 900/900 | 900/900 |
| Pinned scientific records identical, all arms | 1,500/1,500 | 1500/1,500 |
| Pinned outcome differences | 0/1,500 | 0/1,500 |
| Float64 scientific records identical (supplemental) | 1,500/1,500 | 1500/1,500 |

| Default learned arm | Original identical | This rerun identical | Original changed outcomes | This rerun changed outcomes |
|---|---:|---:|---:|---:|
| ppo | 7/300 | 7/300 | 7/300 | 7/300 |
| guarded_ppo | 45/300 | 56/300 | 10/300 | 5/300 |


## Design and comparisons

Ten named scenarios × thirty dev seeds × five arms = 1,500 paired coordinates per comparison. Each CPU ran two default repeats, one kernel-only pinned campaign and one separately classified float64 campaign: eight campaigns and 12,000 study episodes. See [study.json](../../../scripts/validation/cross_cpu_20261013/study.json) and [PLAN](../../../scripts/validation/cross_cpu_20261013/PLAN.md) for exact scenarios, arms, thread/BLAS environment and every reconstruction choice. Main source at clone: `e251a2f94e9478ab7fd5252e88d0296666876ab1`; actual frozen acquisition producer: `83b16e720e348ed794062b35dfd0772ad8d5763a`. Historical source was `66f402ba176b13e45210d0da0b2cf20fcdc0cc02`. A hash audit verifies that the authored campaign template, release scenario entrypoint, horizon schedule and all arm configs are byte-identical to that historical source; generated configs and the lost call graph remain reconstruction choices. Of 54 transitive scenario/include/map files, 52 match; only unselected join-group/leave-group definitions differ. All ten selected SVG maps match.

Default explicitly removes inherited dispatch pins. Kernel-only pinned sets the comment's four dispatch variables and disables mkldnn/enables deterministic Torch algorithms in interpreter startup, arm subprocesses and fork workers, while retaining the normal float32 actor. Float64 opts into main's tracked actor through both campaign and learned-arm configs. All runs use four episode workers, one-thread numerical libraries, explicit CPU devices and empty CUDA visibility; no GPUs were requested.

Scientific identity canonicalizes JSON object order, with no tolerance, across `metrics`, `metric_values`, steps/status, outcome, termination and failure-mechanism fields. Signed zeros and adjacent float values remain different. Outcome identity uses explicit native event outcomes, termination and status, separately from numeric collision counts. Timestamps, timing, paths, machine identity and other administrative provenance are preserved raw but excluded from the scientific projection. Intermediate tensors and full trajectories are not compared.

`a` below means AMD and `b` means Intel. The cross-CPU default headline uses repeat 1. Noise rows compare default repeat 1/2 on the same node; all 30 seeds and all ten scenarios are retained.

| Comparison | Scientific identical | Learned identical | Controls identical | Outcome differences |
|---|---:|---:|---:|---:|
| cpu-default-r1 | 963/1500 | 63/600 | 900/900 | 12/1500 |
| cpu-default-r2 | 963/1500 | 63/600 | 900/900 | 12/1500 |
| cpu-float64-r1 | 1500/1500 | 600/600 | 900/900 | 0/1500 |
| cpu-pinned-r1 | 1500/1500 | 600/600 | 900/900 | 0/1500 |
| default-vs-float64-a | 786/1500 | 57/600 | 729/900 | 14/1500 |
| default-vs-float64-b | 786/1500 | 57/600 | 729/900 | 14/1500 |
| default-vs-pinned-a | 789/1500 | 60/600 | 729/900 | 17/1500 |
| default-vs-pinned-b | 786/1500 | 57/600 | 729/900 | 15/1500 |
| noise-a | 1500/1500 | 600/600 | 900/900 | 0/1500 |
| noise-b | 1500/1500 | 600/600 | 900/900 | 0/1500 |


The pinned bundle changes whole-benchmark numerical semantics against default; consult the within-CPU comparison rows before interpreting it as behavior-preserving. Observed equality in these coordinates does not establish equivalence elsewhere.

## Environment and execution

| Context | Node | CPU model | Kernel | glibc |
|---|---|---|---|---|
| amd | auxme-imech091 | AMD EPYC 9354P 32-Core Processor | 6.8.0-106-generic | glibc 2.39 |
| intel | auxme-imech172 | Intel(R) Xeon(R) Silver 4310 CPU @ 2.10GHz | 5.15.0-191-generic | glibc 2.35 |


Both requested CPU models were available, so no substitute was used. Python was 3.13.14 on both contexts; NumPy `2.4.6`, Torch distribution `2.13.0` (runtime CUDA build is captured in process context JSON), SB3 `2.9.0`. Complete installed inventories and CPU identity are retained. Both CPUs have identical lock and installed-inventory identities: lock SHA256 `da16ae855eb53d76fddf8a351c881f9b178dd8157367b30679bfd459a6040f28`, inventory SHA256 `76ca6970990981429c6bd41237072a8f105c6546a72c0024e2736de79050eeb9`. This lock differs from the original comment's lock; historical environment equality is not assumed.

Live default process contexts show AVX512/mkldnn enabled. Kernel-only pinned contexts show DEFAULT/mkldnn disabled/deterministic algorithms enabled, with single-threaded Haswell OpenBLAS on both CPUs, not merely requested config strings. The Torch compatibility helper preserves the effective deterministic flag while avoiding the version's public-wrapper import crash. Supplemental float64 uses the canonical early bootstrap and its recorded per-row numerical context.

Slurm jobs: AMD default-r1 31591, default-r2 31592, pinned 31593, float64 31600; Intel default-r1 31595, default-r2 31596, pinned 31597, float64 31601. Campaigns overlap subject to normal Slurm resources/QoS. Shared-host load limits timing interpretation. Allocated CPU-hours for all task jobs, including probes, canaries and failed attempts: **40.421**, below 150. Scheduler receipts/accounting bind each job to its actual node and four-CPU allocation. The eight acquisition jobs had a combined worst-case allocation of 128 CPU-hours, with a separate ten-hour setup reserve.

## Integrity and excluded setup attempts

The audited study census contains `12000` records. Schema failures `0`, forbidden execution/evidence markers `0`, integrity contradictions `0`, invalid or unmeasured spawns `0`. Every campaign has exactly the expected 1,500 unique coordinates. No study coordinate was dropped to improve identity rates. Guarded PPO's declared shield decisions remain method telemetry under the canonical identity-bound fallback policy.

Two first canaries (31587/31588, 100 episodes total) are excluded from scientific comparisons because changing the producer Git head while they ran caused the canonical mixed-commit integrity gate to reject them. Frozen replacements 31589/31590 passed (100 native episodes). These setup episodes are preserved separately and are outside the 12,000 study census.

Initial supplemental job 31594 failed before episodes because float64 was not also opted in at the learned-arm configs. Pending 31598 was cancelled before execution. Corrected generation copies each original arm config and adds only its numerical-mode flag. Dedicated staged preflight 31599 passed before jobs 31600/31601 were submitted. The failed attempt, input versions, corrected generator overlay and differential retry receipt are preserved. Default/pinned acquisition was not rerun. Final checked-in generation includes the repair and a pre-launch development-seed/CPU-input guard. The acquisition checkout stayed frozen at the producer above; those tooling repairs are distinguished from its execution bytes.

## Preservation and reproduction

External data hub retrieval key: `benchmark-results/robot_sf_ll7/cross_cpu_20261013/run-1d1b1b1007`, accessed through `imech192`. This namespace was chosen during setup; its suffix is not the full acquisition source identity. See [evidence pointer](evidence/hub_pointer.json) for the exact retrieval path, SHA256 manifest digest and readback receipt. [Comparison summary](evidence/comparison.json) and [native row audit](evidence/row_audit.json) are tracked compact evidence. The hub preserves raw episodes, campaign/preflight manifests, exact generated configs, checkpoint bytes, source archives and Git objects/bundles, original comment JSON, complete environment/inventory/CPU contexts, all logs/receipts, pair-by-pair hashes and scheduler accounting. A separately retrieved mirror was byte-verified against every manifest entry after producers stopped.

To rerun, hydrate the archived producer source and dependency lock, stage the public checkpoints through the canonical preflight, then use the checked-in driver to `prepare`, `submit`, `collect`, `compare`, and `manifest`. Supply both verified node/partition/CPU-QoS tuples explicitly; the login host is not an episode worker. For byte-exact input audit, use the preserved corrected configs and generator overlay. All final generated inputs must pass the explicit development-coordinate/CPU guard. See PLAN for stop/recovery rules. Offline comparison needs no episode compute.

Validation: 29 focused tests (nine new comparison/input controls and twenty existing numerical-mode/float64 actor controls) pass in a fresh frozen-lock environment; lint, format, type checks and documentation/evidence integrity pass. Three pre-fix negative controls demonstrate the corrected outcome-count distinction, float64 config binding and seed guard. The full unrelated repository test suite was not run: shared runtime is untouched, and canonical staged preflights plus real native acquisition supply the task's executable evidence. Hosted CI and independent review remain separate from this diagnostic verification.

The reconstruction guesses are explicit in PLAN: current authored-config/runtime rather than historical bytes; unknown original call graph, arm order/affinity; current lock/build inventory; compatibility setter; current float64 actor; reconstructed scientific/outcome projection; omitted auxiliary empty-world/inference gates and aggregate intervals. No inference is made about the missing original producer.

## Independent-review correction

The P2 review of PR #10371 at `d81f0195d227b398cdd5ddb8e545c356d4d1542b` found that the startup hook was missing from the repository broad-exception ratchet. Register its unchanged fatal startup boundary with an explicit justification: Python ignores ordinary `sitecustomize` exceptions and can continue unpinned after an unexpected dependency error, so a known-type list cannot preserve this startup contract. Two subprocess controls inject an unfamiliar dependency exception in default and pinned modes and require exit 70 before user code. The complete canonical lint phase passes, including 207 registered broad handlers; the focused study, numerical-mode and ratchet tests pass 41 cases. See [correction validation](evidence/review_fix_validation.json) and its adjacent logs. Hook behavior and the sealed acquisition bundle are unchanged.
