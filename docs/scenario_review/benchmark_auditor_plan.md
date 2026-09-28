# RobotSF Benchmark Auditor — complete product and delivery plan

Status: maintainer-selected requirements and implementation plan, not implemented functionality or benchmark evidence.

Plan version: 1.0, 2026-09-17. Coordination issue: [#9483](https://github.com/ll7/robot_sf_ll7/issues/9483).
Source snapshot inspected during preparation: `main` at `a74984e2376e55728d792787705f015b94b38e99`. Refresh current code, issue state, claims and pull requests before implementation.

### Additive BA-05 contract amendment (2026-09-21)

The accepted BA-05 service slice uses `github-publication.v1`. It creates one
immutable issue snapshot containing the stable finding marker, then publishes
later finding revisions as auditor-owned comments carrying an exact semantic
publication marker. The durable local outbox is the authority for publication
identity, revision, digest, request operation, remote observation and
reconciliation state. Replays therefore deduplicate by semantic publication
key rather than by a caller's operation ID. Existing body-CAS helpers remain
only as an explicitly selected compatibility path for test fakes; the REST
adapter never performs a read-then-`PATCH` issue-body update.

This protocol does **not** claim a GitHub compare-and-swap primitive,
atomicity between the local journal and GitHub, or exactly-once remote
delivery. A complete marker search is required before create or comment
retry; incomplete pagination, a timeout that cannot be reconciled, a remote
duplicate, or a local link-CAS conflict remains visible as ambiguous or
conflict and blocks automatic retry. These limitations are acceptance gates,
not reasons to promote a plausible issue body to canonical evidence. The
acceptance report must retain the local outbox record, remote marker, exact
head and hosted-CI receipt for every claimed publication.

The service's Codex route is likewise explicit about accounting. `offline`
is provider-free and read-only. `local_accounting` reserves finite local
token/compute/issue-write budgets, records observed usage and overspend, and
does not imply a provider-enforced ceiling. The opt-in
`strict_provider_ceiling` mode requires an exact route/provider/model and a
verified finite provider compute ceiling; it refuses admission when that
ceiling is absent or a reservation exceeds it. A provider capability that
does not expose a verified physical ceiling cannot be represented as strict
mode. Missing usage, cancellation, retries, reconnects and nested sessions
remain ledger outcomes, never silent budget resets. Hosted/live evidence must
report the selected mode and whether a physical provider cap was actually
verified.

### Additive implementation and acceptance amendment (2026-09-22)

The original plan publication was documentation-only. The current implementation
is deliberately recorded as independently reviewed slices; this amendment does
not replace the requirements register and does not close the epic.

| Slice | Exact reviewed head | Hosted evidence | Result |
| --- | --- | --- | --- |
| BA-05 D1 append-only publication | PR #9559, `ec8061c4517fb3926fefd1b1a2cb6cb3cdcf0088` | run `35655668527`, accepted retry | accepted bounded slice |
| BA-05 D2 route/accounting | PR #9558, `78fb3c818fafe1ae0ea1522c96092e55b8460e3b` | run `35658827492` | accepted bounded slice |
| BA-06 server-held publication | PR #9561, `e2c84dd8ea75bd63a62005ecbc55ae7847d06328` | run `35667772524` | diagnostic-only slice |
| BA-06 external MCP context | PR #9565, `2d486ed94bfc1d785e1fe576ca8232ab3bc7d880` | run `35676387775`, attempt 2 | diagnostic-only slice |
| BA-06 runtime friction repair | PR #9569, `4833a1eb06b60f04aca78003789bdf992361c8be` | run `35683653944`; merge gate `35685480912` | merged; 64.88 s → 17.55 s |

BA-05 acceptance is therefore bounded to the offline/fake-provider and route
contracts plus the hosted evidence above. The REST transport's admissible V1
path is append-only: one immutable issue snapshot followed by marker-bearing
auditor comments. GitHub's ordinary issue API does not supply a proven
cross-system compare-and-swap or exactly-once delivery boundary. Complete
marker pagination, timeout reconciliation, duplicate-marker detection, local
outbox recovery, and the canonical finding-link CAS remain explicit gates. A
read-then-`PATCH` body update is not CAS and must never be promoted as live
evidence; body-CAS helpers are compatibility fakes only.

Codex accounting is likewise bounded. `offline` is provider-free and
read-only; `local_accounting` records finite local reservations and any
provider overspend without claiming physical enforcement; and
`strict_provider_ceiling` refuses admission unless a finite provider ceiling
is verified. The tested App Server exposes token telemetry but no measured
provider compute cap, so no strict live route or premium-provider fallback is
claimed. A future live receipt must include the selected accounting mode,
provider/model route, verified-cap status, and before/after usage.

BA-06 remains open for the full real-data/browser workflow, recovery and
cancellation, native/source-bound diagnostics, and a live Codex/MCP receipt.
SREV #9285/#9287/#9288 are contract/test integrated; #9296/#9299 still depend
on admitted-source/live evidence and prerequisite #9417 enforcement. The full
decision matrix and machine receipt are in
[`benchmark_auditor_v1_acceptance_2026-09-22.md`](./benchmark_auditor_v1_acceptance_2026-09-22.md).

Merging this plan amendment does not close #9483, #9488, or #9489.

## 1. Goal and product boundary

Build a local Benchmark Auditor that answers: **Which case should be investigated next to reduce uncertainty about benchmark correctness, what happened, what did we conclude, and what remains unchecked?** Correctness auditing, understanding planner behaviour and discovering scientifically useful examples are all goals. Their review-queue priority is deliberately different from scientific example selection.

The complete loop is:

```text
current-schema campaign -> complete input accounting -> automatic checks
 -> transparent signals -> fixed or active review queue -> ReviewPacket
 -> synchronized scene/video/metrics/events -> quick or rich annotation
 -> shared finding -> related cases -> bounded diagnostic investigation
 -> finding-level GitHub issue -> updated knowledge and coverage -> Next
```

The existing SREV workbench is the viewer subsystem. Do not create another renderer, trace format, annotation database, experiment runner, hosted service or general-purpose agent fleet. The new product-level responsibilities are the audit engine, review policy, durable findings, coverage protocol and autonomous control interface.

## 2. Accepted requirements register

The numbers below preserve the entire requirements discussion, including the corrected original numbering for questions 25–32. These are accepted product choices; further product interviews are not a prerequisite.

| Decision | Accepted requirement |
| --- | --- |
| 1 | Correctness, behaviour understanding and interesting examples are all product goals. |
| 2 | Choose episode, matched comparison, cell or group review scope dynamically. |
| 3 | Cheap checks across the entire campaign, expensive analysis selectively. |
| 4 | All proposed anomaly dimensions remain in scope, subject to actual data availability. |
| 5 | Prioritize benchmark/config defect, release/manuscript impact, unexplained behaviour, statistical unusualness, planner disagreement, then safety severity. |
| 6 | Initially flag and explain suspected problems; automatic grading/regrading is an explicit later extension. |
| 7 | Small top-level annotation taxonomy with optional detailed tags. |
| 8 | Separate observations, hypotheses, measured evidence and confidence. |
| 9 | Rich interval/actor/evidence annotations plus faster, simpler forms. |
| 10 | Scene/video, summary, metrics and annotations together, with reference points and contextual help. |
| 11 | Core metric/event options are available but toggleable, not all visible at once. |
| 12 | Cross-planner comparison is important; convenient direct Codex/MCP discussion is also required. |
| 13 | Include suspicious successes and unusual metric combinations, not failures alone. |
| 14 | Explicit formal audit-coverage rules. |
| 15 | Deterministic rules and statistical anomaly discovery. |
| 16 | Bounded diagnostic reruns are part of the selected product, implemented after basic inspection in the delivery sequence. Later decision 29 includes autonomous execution in V1. |
| 17 | Structured portable records plus a fast local SQLite index. Decision 36 fixes which is canonical. |
| 18 | Queue, annotation database, health report, bugs and examples; durable audit knowledge is central. |
| 19 | Keep score components and evidence channels visible. |
| 20 | Human-approved detector proposals first, adaptive ranking later, learned classifiers after that. |
| 21 | One-click, quick and full annotation modes. |
| 22 | Timestamp plus spatial point is the minimum; actor/object/goal/waypoint references where available. |
| 23 | Toggleable scene overlays, numbered references by default. |
| 24 | Findings group multiple episode annotations. |
| 25 | Named similarity modes plus explained agent suggestions, not opaque similarity alone. |
| 26 | Already-reviewed redundancy lowers priority while useful new evidence remains eligible. |
| 27 | Dynamically coverage-aware ordinary/control sampling. |
| 28 | Release-specific protocol completion with explicit remaining exceptions. |
| 29 | Autonomous analysis, annotations, GitHub issues and bounded diagnostics from V1; no per-action confirmations. |
| 30 | Agent access to complete scoped underlying data, code/configuration, peers and annotations; UI selection focuses the question. |
| 31 | Agent proactively suggests evidence, related cases and hypotheses. |
| 32 | Findings, not individual strange episodes, drive persistent GitHub linkage. |
| 33 | Any campaign conforming to current RobotSF result contracts; historical incompatible adapters are later work. |
| 34 | Lazily obtain or regenerate missing review material on demand. |
| 35 | Broad twelve-family detector suite in V1, plus advisory statistical analysis. |
| 36 | Versioned JSON/NDJSON is canonical; SQLite is a rebuildable projection. |
| 37 | Full agreed audit UI in V1; no polished 3D, publication editor, dashboard builder, mobile or hosted service. |
| 38 | Agent-centric API/MCP interface is first-class V1 functionality. |
| 39 | Both fixed-score and knowledge/coverage-aware Next Best Review, active policy default. |

## 3. Issue ownership and implementation inventory

| ID | Owning issue | Responsibility |
| --- | --- | --- |
| BA-01 | [#9485](https://github.com/ll7/robot_sf_ll7/issues/9485) | Campaign accounting, detector registry, deterministic and statistical signals. |
| BA-02 | [#9486](https://github.com/ll7/robot_sf_ll7/issues/9486) | ReviewPacket selection, transparent ranking, redundancy and control sampling. |
| BA-03 | [#9484](https://github.com/ll7/robot_sf_ll7/issues/9484) | Minimal shared audit records, canonical persistence, annotations, findings and similarity. |
| BA-04 | [#9487](https://github.com/ll7/robot_sf_ll7/issues/9487) | Protocol, coverage accounting, deficits and health report. |
| BA-05 | [#9488](https://github.com/ll7/robot_sf_ll7/issues/9488) | Shared domain service, Codex/MCP, materialization, bounded diagnostics and GitHub sync. |
| BA-06 | [#9489](https://github.com/ll7/robot_sf_ll7/issues/9489) | Workbench integration, early real-case vertical slice and final end-to-end acceptance. |

The integration issue requires BA-01 through BA-05 plus SREV-16/17/18 acceptance. BA-03 first supplies a **small shared schema and executable fixture slice**, not a complete knowledge platform. Sibling development can then proceed against those fixtures; do not turn every interface handoff into a dependency on another entire issue. Record any genuinely required smaller dependency with an exact deliverable and unblock predicate.

### Reuse and extend the existing SREV owners

The [SREV portfolio #7398](https://github.com/ll7/robot_sf_ll7/issues/7398) remains separate. Its current modules, implementation receipts and tests outrank historical task descriptions.

| Existing work | Auditor use |
| --- | --- |
| #9270 contracts; #9271 import; #9273 packaging | Reuse source references, capabilities, versioned requests/results and portable artifact handling. |
| #9274 events; #9275 campaign context; #9276 storyboards | Reuse event intervals, cohort denominators and explanatory clips without equating audit priority with scientific selection. |
| #9277 alignment | Verify comparison compatibility; equal seeds alone are insufficient. |
| #9272 sync; #9278 scene; #9279 encoding; #9280 cameras; #9281 overlays; #9282 media QA | Reuse measured time mapping and canonical rendering, with truthful missing-stream behaviour. |
| #9284 shell | Host the auditor in the existing offline browser application. |
| #9285 SREV-16 | Synchronized scene/video/metric/event panels and selection context. |
| #9287 SREV-17 | Three annotation speeds, numbered spatial references and source-bound storyboard editing. |
| #9288 SREV-18 | Optional recorded planner/pedestrian diagnostics and commanded/executed controls. |
| #9283 reports | Reuse traceable numerical references and readable report generation. |
| #9290 registry; #9291 jobs | Reuse component discovery, job cache and resumable progress rather than another orchestration system. |
| #9292 recipes; #9293 executor; #9295 experiment report; #9296 loop; #9299 session UI | Reuse bounded control/intervention semantics with their actual supported source/planner scope. |
| #9297 narrative adapter | Keep this tool-free explanation component separate from the autonomous audit control service. |

Historical plans refer to a separate `docs/scenario_review/review_workbench.md` and render-folder tests. Source inspection instead found the SREV-15 documentation in [review_contracts.md](./review_contracts.md), section `Review workbench (SREV-15)`, and tests in `tests/test_review_workbench.py`. Locate actual symbols and tests; do not create duplicate files to satisfy obsolete paths.

### State corrections made during plan publication

SREV-16 #9285, SREV-17 #9287 and SREV-18 #9288 had stale dependency and `needs-triage` labels despite #9284's completed shell. Their bodies now reflect accepted scope, the historical prerequisite and actual source paths; those stale labels were removed. **No live `state:ready` result is asserted by that metadata cleanup.**

Closed executor issue #9293 does not mean all runtime integrity work is complete. [#9380](https://github.com/ll7/robot_sf_ll7/issues/9380) records post-merge defects. Its resume, planner-parity and deadline slices have reported successor fixes, which must be verified rather than reimplemented blindly. The residual [#9417](https://github.com/ll7/robot_sf_ll7/issues/9417) source-admission enforcement is still implementation work. Its design hold is resolved in Section 12; #9296 now names #9417 as its actual runtime prerequisite. Keep that evidence dependency until proof exists.

Native GitHub parent/dependency links, Project #5 fields and canonical live admission could not be executed by the connector-only publication environment. Body and comment context may support relationship decisions, but native GitHub links are authoritative and do not need body mirrors. The implementation coordinator must perform scoped native reconciliation and live admission through the repository helpers before dispatch; prose references are not completed native mutations.

## 4. Campaign identity and evidence accounting

Use immutable campaign/source digest and execution identity. Never identify an episode solely by planner, scenario and seed: checkpoints, configuration, implementation revision, environment, attempts and reruns can differ. A campaign's file location is not its identity. Support the current schema without hard-coding release, planner count, horizon, seed count or campaign size. The discussed 20,160 episodes are an example source population, not a constant.

Every expected input row is accounted for as readable, missing, duplicate, invalid or unsupported. Separate:

- indexed campaign coverage;
- per-detector evaluability and unavailable/error counts;
- human review, agent review and automated flags;
- reviewed interval versus full-episode decision;
- reviewed finding representatives versus confirmed affected members;
- original recorded evidence versus new diagnostic executions.

Visiting all aggregate rows does not imply that every trajectory or planner-internal detector ran. Unavailable optional telemetry is not itself proof of corruption. Missing telemetry promised by a source contract is a separate integrity signal. Repeated excerpts, repeated annotations and related-case membership do not increase original-episode denominators.

## 5. Automatic audit engine

Each detector declares its version, input capabilities, cohort definition, parameters and units. Return `flagged`, `clear`, `unavailable` or `error`, with evidence references, measured values, interval, thresholds and reason codes. Keep heuristic suspicion, factual contract violation and causal interpretation separate. Derived data identifies its producing method and source; detectors do not rewrite released metrics or secretly run simulations.

V1 covers the agreed twelve families:

| Family | Examples and decisive controls |
| --- | --- |
| 1. Completion and goal geometry | Goal-adjacent timeout, wall clearance, active/final waypoint and visible-goal versus actual completion boundary. A nearby wall is not automatically an impossible task. |
| 2. Initial/reset anomalies | Immediate collision, spawn overlap, invalid initial state, reset mismatch; distinguish legitimate initial conditions under the actual scenario contract. |
| 3. No progress | Stuck or stalled movement relative to route/goal progress, with expected waiting/yielding controls. |
| 4. Oscillation and loops | Repeated turning, reversals, trajectory loops and terminal limit cycles; distinguish deliberate avoidance and turns. |
| 5. Outcome/metric inconsistency | Completion, collision, timeout, duration and event contradictions under the actual termination definitions, not a guessed universal rule. |
| 6. Extreme measurements | Clearance, collision, force and TTC/PET anomalies with units, finite-data rules and legitimate undefined/non-applicable values. |
| 7. Command/execution mismatch | Saturation, clipping, discontinuity and desired-versus-applied command differences, respecting action/dynamics semantics. |
| 8. Telemetry integrity | Missing promised fields, nonfinite values, corrupt/truncated streams and gaps, separated from optional unavailable measurements. |
| 9. Configuration/provenance | Unexpected planner, adapter, checkpoint, observation contract, parameter drift, seed/reset and source identity. |
| 10. Common-mode anomalies | Shared failures or suspicious outcomes across comparable planners/configurations. |
| 11. Planner disagreement | One planner differs strongly from compatible peers; disagreement is suspicion, not proof of a defect. |
| 12. Seed/cohort outliers | Unusual realization within an explicitly comparable configuration; seed-number adjacency has no scientific meaning. |

Add cheap robust statistical/multivariate and trajectory-shape outliers as advisory discovery. Report features, scaling, cohort, method version, missingness and too-small-cohort cases. Avoid introducing a heavy embedding/vector-database or training system in V1.

Reuse goal-timeout work [#9429](https://github.com/ll7/robot_sf_ll7/issues/9429), canonical failure predicates, event phases, portfolio and metric owners. The Social Force case [#9428](https://github.com/ll7/robot_sf_ll7/issues/9428) is a required diagnostic example: point-based completion, robot footprint/goal radius, nearby wall, force response and dynamics can combine into a limit cycle. Do not call it geometrical impossibility without a separate reachability check. Disabling all obstacle forces is a diagnostic intervention, not evidence that wall avoidance is safe.

Detector evolution is staged: annotations can suggest candidate rules; human-approved rules may later run campaign-wide. V1 records these proposals without waiting for approval to continue other investigations. Adaptive ranking and supervised grading/classifiers are later explicit extensions; no silent online policy change.

## 6. ReviewPacket, priorities and selection

A ReviewPacket includes a primary episode, compatible peer candidates, other realizations/cell context, signals, related findings, selection reasons, available assets and missingness. It may represent one episode, a comparison, a cell or an anomaly group. Always retain an exact primary source and a way back to full-episode context.

Offer fixed ranking and active Next Best Review. The fixed policy is transparent and reproducible. Preserve the accepted ordering with priority bands or lexicographic precedence before within-band weights: a collection of low-impact novelty points must not silently dominate a strong benchmark/configuration defect. Display component contributions and reason codes. A score such as 0.91 is an **uncalibrated priority**, not a 91% probability of a bug.

The active policy adds coverage gain, new evidence about competing hypotheses, unresolved finding requests, novelty and redundancy. It is a documented heuristic for useful review, not a claim of mathematically optimal information gain. Store policy/config version, input/index revision, RNG state and selection rationale.

Reviewing another manifestation of a known symptom lowers its priority, but distinct geometry, planner, outcome, severity, contradictory evidence and explicit requests for more evidence can raise it. Similarity never means already reviewed. Prevent starvation with recorded age/defer rules and a finite schedule for the control stream.

Ordinary/control selection is dynamically coverage-aware. Prefer under-reviewed observed strata and select randomly within the chosen stratum independently of anomaly flags. There is no permanent fixed 70/20/10 split. Preserve a separate record of why each case entered the control stream. Adaptive/anomaly-selected review is not an unbiased estimate of defect prevalence.

Support manual pin, skip/defer, return to previous packet, resume and explicit more-evidence requests. Recompute against new knowledge without losing earlier selection provenance. Missing dissertation-impact metadata remains unknown; core use cannot require a private dissertation checkout.

## 7. Review interface and synchronization

The default screen shows the current scene or video, compact source/outcome/anomaly context, a shared scrubber, a small chosen set of metric traces and annotations. Queue, findings, related cases, coverage and agent conversation can expand or collapse. The product must feel like one review session rather than a directory of CLI outputs.

Core interaction: play/pause, previous/next step, speed, selected interval, event jumps, graph scrubbing, reference selection, quick annotation, undo/redo and Next. Keyboard shortcuts must not fire while typing notes. Reconfigure panels and save view preferences separately from canonical evidence.

Metrics are toggleable: goal distance, speed/angular velocity, pedestrian and obstacle clearance, collision/near-miss events, meaningful TTC/PET, commanded/executed control and completion state. Optional planner candidates/costs/constraints and pedestrian force/interaction diagnostics appear only when actually recorded or explicitly derived. Keep planner-visible input, simulator ground truth and post-hoc values distinguishable.

Simulation time is the authority. Synchronize video using its explicit presentation-timestamp mapping, not inferred frame/fps arithmetic. Declare nearest-sample/interpolation rules, units and resolution. Preserve nonzero origins, gaps, unequal rates/durations and unavailable tails. Cropping, pauses and speed changes retain source-time mapping; do not normalize every comparison to the same apparent duration.

Show the sampled final goal point, active waypoint and actual completion boundary when available, not merely a decorative goal rectangle. Comparison uses canonical initial-state/configuration checks: same scenario and seed are lookup keys, not proof of matched starts or identical pedestrian histories. Robot-dependent pedestrian responses may diverge even under matched initial conditions. Offer full context around selected excerpts.

## 8. Spatial references and annotation workflow

Minimum spatial reference: source execution, source time, coordinate frame and clicked point. Prefer world coordinates derived from known scene geometry/transforms. A video without a verified transform supports image-space coordinates bound to media/frame/crop identity, clearly not metre measurements. Actor, goal, waypoint, map object, event and metric references are optional typed links, never invented identifiers.

Draw numbered references by default, with toggleable highlights, arrows, circles, short labels and measured distance lines when geometry permits. References survive pan/zoom/crop and remain tied to source time. A changed source hash makes the attachment stale rather than rebinding it silently. UI hints show the selected actor/object, goal/completion state and relevant metric units so annotations are precise without lengthy spatial prose.

Three annotation levels share one schema:

1. One-click Normal/Suspicious/Bug/Unsure triage; `Bug` initially means a suspected defect classification, not canonical episode invalidation.
2. Quick category/tag plus current timestamp or interval.
3. Full observation, interval, actors, references, measured evidence, hypothesis, confidence and notes.

Use `normal`, `interesting_valid`, `planner_defect`, `benchmark_defect`, `scenario_defect`, `instrumentation_defect`, `unclear` plus tags. Do not force a cause or confidence into a quick observation. Scope records distinguish a flag, a reviewed interval and an explicit full-episode review. A quick whole-episode decision is possible, but not inferred merely from opening a clip.

Author identity is human, detector or agent. Autonomous agents can save their own records immediately; they cannot impersonate a human review. Preserve history when humans or agents correct/refute a record. SREV-17 owns editor behaviour; BA-03/05 own the common storage/service, not another UI-local database.

## 9. Canonical storage, findings and similarity

Use versioned JSON/NDJSON as the canonical durable record and SQLite as an index/projection. Suggested logical records: CampaignAudit, EpisodeRef, ReviewPacket, Signal, Reference, Annotation, Finding, ReviewRecord and ActionRecord. Reuse SREV source/time schemas; avoid another raw trace format.

One local writer serializes committed transactions. Every write carries operation ID, expected revision, actor and source context. Acknowledge save only after the canonical transaction is durable; update SQLite with a projection checkpoint so a crash between journal and index loses no acknowledged edit. Incomplete transactions cannot appear committed. A UI and an agent must not append independently to the same NDJSON file. Test two simultaneous clients, conflicts and replayed requests.

Version edits, deletions/tombstones and undo. Cache validity depends on source, tool, config and schema versions. Rebuilding SQLite, moving the artifact root or restoring a backup must recover the exact committed history and references. Raw traces/videos remain referenced in preserved storage outside Git; annotations/manifests are small, and an audit export must not accidentally include credentials or unrelated files.

A Finding groups a recurring symptom or supported defect. Store title/status, representative cases, candidate and confirmed members separately, evidence/hypotheses/negative controls, diagnostic results and GitHub identity. Suggested lifecycle: proposed, under_investigation, supported, refuted, resolved. A group of similar cases is not automatically one proven cause; interventions and confirmations keep their separate evidence status.

Named similarity modes cover same case across planners, same planner across realizations, symptom, anomaly signature, geometry, outcome pattern, metric behaviour and existing finding. Agent combinations must report why each result is related and its compatibility/missingness. Display candidate counts separately from confirmed manifestations. No opaque similarity score replaces explanatory features.

## 10. Lazy artifact materialization

The product must work without pre-rendering an entire campaign. Resolve an asset in this order:

1. Locate and hash-verify the original recording/trace.
2. Render retained source states through existing plotting/rendering primitives without constructing or advancing the simulator.
3. Only when states are absent and exact inputs are recoverable, perform a new diagnostic reexecution using the intended canonical planner/configuration and pinned source identity.

A new execution receives its own ID and links to the historical episode. Bind source commit/configuration/checkpoint/initial state/environment; compare recorded outcomes, available aggregates and state fingerprints through existing comparators. Report fidelity as verified for the explicitly checked equivalence scope, diverged or unverifiable. An outcome match alone does not prove full trajectory equivalence. Never label reexecuted data as an original recording or fill absent original metrics with replay values without explicit separation.

Missing original artifacts do not mean the whole audit stops. Continue aggregate inspection and other cases, show the missing capability and enqueue permitted materialization. However, fixture-only fallback cannot satisfy every positive native-data acceptance test. Unsupported stateful/checkpoint planners remain explicitly unsupported until the correct adapter is validated; no quiet substitution of a simple policy.

The service manages source/config/tool-keyed cache, job status, cancellation, progress and durable preservation. Display the available-data path immediately while a requested asset is prepared. Large media is not committed. An initial full scan does not launch a full replacement benchmark campaign.

## 11. Shared API, MCP and Codex

One local domain service is the authority for browser, CLI and MCP. Representative proposed API groups are campaign inspection/scanning; episode data/geometry/metrics/events/comparison; queue/coverage; annotation/reference/finding writes; related-case retrieval; materialization; diagnostic recipe/run/status; and finding-level GitHub synchronization. These are proposed public operations, not claims that such commands already exist.

UI context is a versioned selection snapshot: campaign, source execution, interval/cursor, actor/reference and selection revision. A long investigation must not attach results to the next episode after the user changes selection. Pass exact context to every write and handle stale revision explicitly.

Codex can inspect all scoped underlying evidence and pinned source code/configuration on demand. Do not paste all campaign traces into every prompt. The selected frame/interval/reference focuses questions such as “Why is the robot turning away from R3?” but the agent can inspect precursors, peers, source and findings.

Use an installed supported Codex client/App Server integration for the embedded conversation, alongside the same audit MCP API available to an external Codex session. Inspect the actual installed version and official integration contract rather than inventing methods, model IDs or provider interoperability. Official reference starting points are https://developers.openai.com/codex/app-server/ and https://developers.openai.com/codex/mcp/; these are background documentation, not live proof. Pin the chosen adapter version and test reconnect, cancellation and evidence links.

Keep #9297's tool-free narrative adapter as a reusable explanation component. The autonomous controller is BA-05; do not silently widen the old adapter's authority. Fake-provider tests run without credentials/network. A real available Codex smoke is a separate acceptance receipt, identifying actual route and usage rather than treating mocked success as live integration.

## 12. Autonomous authority and source-admission decision

Autonomous mode is part of V1, not a preview requiring approval for each action. Starting a configured autonomous session grants its scoped service capabilities: read evidence/source, save agent-authored annotations/findings, look for related cases, run supported bounded diagnostics, and create/update relevant GitHub issues. Read-only mode remains usable offline and launches no execution.

Use one explicit session policy with allowed source/output roots, repositories, recipe interventions, finite aggregate compute/token/issue-write budgets and cancellation. Trusted server/launcher configuration supplies hard ceilings independently of artifact content; every session must declare finite token and issue-write maxima and may only narrow those ceilings. Missing, malformed or over-limit budgets fail closed. Enforce permissions server-side; do not rely on prompt text. Keep tokens/credentials out of the browser, artifacts, logs and published reports. Treat source files, annotations and GitHub discussion as data, not new authorization or shell instructions. Protect local endpoints with loopback binding, origin/session-token checks and scoped path resolution.

Every action records operation ID, actor/actual model, source/config/context revisions, reason, requested operation, state transition, result, budget consumption and external mutation identity. Source inputs are immutable. Local record corrections are reversible through history, but a published issue is an external action that cannot be made unseen; avoid unsupported public claims and accidental private-data disclosure.

### Resolved #9417 engineering choice

The admitted-source trust root must come from validated executor/launcher configuration outside the artifact. Reuse `AdmittedSourceReceipt` and `resolve_admitted_source`; resolve receipts within the configured root, rehash source bytes at use, and bind request/recipe/config and preservation identity. The recipe cannot authorize its own source or nominate a new trusted root.

Prefer an additive versioned executor admission configuration with documented migration. If actual code proves a request-schema change unavoidable, introduce a narrow explicit v2 envelope while preserving review-bundle v1 semantics and a legacy path that cannot claim admitted completion. The coordinator can make that engineering choice without a human question. It may not mint scientific admission, trust a string-only receipt or waive source verification. `scientific_claim_allowed: false` remains intact for this diagnostic executor.

The #9417 design label was cleared; the enforcement code and tests are still required. Resolve that slice before accepting real source-bound #9296/BA-05 execution. Verify the other reported #9380 fixes rather than broadening the work into another hardening rewrite.

## 13. Bounded diagnostic investigation

A recipe states hypothesis, unchanged control, finite interventions, expected measurements/direction, activation test, fidelity checks, evaluation rule and preservation destination. Reuse SREV recipes/executor/report/loop and canonical runner adapters. The existing fixture executor does not automatically provide native campaign replay; BA-05 owns the smallest validated native-source integration needed by the auditor.

Default per-session SREV limits remain three candidate interventions, six total simulator executions, 600 elapsed seconds and one concurrent local CPU process. Count controls, failed attempts, retries and required fidelity repetitions. Reserve a full pair; do not weaken fidelity tests to fit. Persist total consumed budget across crash/resume and nested sessions. The outer audit session also has an aggregate cap so starting a new child cannot reset spending.

For V1, keep the existing SREV child hard envelope unchanged and add an outer hard envelope of at most ten diagnostic sessions per audit session, 60 simulator executions and 6,000 active execution seconds in aggregate, with one concurrent simulator. Trusted server/launcher configuration may narrow these limits at session startup but may not raise them; child sessions cannot reset either envelope. These are safety/accounting limits for application operation, not a wall-time estimate or limit for implementing the package. Keep the service useful when its execution budget is exhausted.

A successful treatment with a failed or divergent control is inconclusive, not proof. Measure intervention activation. Retain negative, refuted and inconclusive results. Stop only the affected investigation on invalid sources, unsupported planner state, exhausted budget or cancellation. No automatic source-code repair, training, remote/Slurm scheduling, full campaign rerun or benchmark release from the runtime auditor.

## 14. GitHub findings and downstream use

A finding may link to an existing issue or create one after deduplication. Include finding ID, exact campaign/source identities, representative cases, candidate versus confirmed counts, observations, competing explanations, supported diagnostics and reproduction commands. Local artifact URLs are not remotely accessible evidence links; publish only preserved authorized artifact references or identify an artifact as local-only.

Use an operation journal/outbox. After a network-ambiguous create, search/reconcile by stable finding marker before retrying. Update an auditor-owned block; preserve human text, comments and unrelated labels. Keep conflicts and failed sync visible. Do not automatically close an issue as fixed just because a hypothesis has an explanation or a new rerun looks better.

Core product implementation belongs in RobotSF. An optional locally supplied release/manuscript-impact mapping may link findings to claims and figures in `ll7/diss`, but no private manuscript or credentials go into the public repository. The auditor may export evidence for the existing dissertation admission/rebase process; it cannot silently change pins, numbers, rankings, claims or release identity. The selected audit queue is not the publication case-selection protocol.

## 15. Audit protocol and health report

Ship `audit-protocol.v1` as a configurable operational specification, not a statistical confidence certificate. Defaults remove the need for another product decision:

- Account for all expected source rows and every scheduled detector attempt, including unavailable/error results.
- Obtain one explicit full human review in each observed planner × declared scenario-group × outcome stratum.
- Include a detector-independent ordinary/control review in each observed planner × group stratum where a valid ordinary candidate exists. A control may satisfy another requirement when both rules explicitly permit it.
- Review every high-priority anomaly cluster through a representative and a different context/control when available.
- Give every high-priority integrity finding an explicit disposition; unresolved critical findings remain visible and cannot produce an unqualified complete result.

These are initial engineering coverage targets, not statistically calibrated sample-size claims. Do not require watching every failure. Empty strata, unavailable assets, unreviewable cases and declared exceptions retain reasons and original denominators. Record protocol version/digest before review; later changes create a new receipt, not retroactive completion.

Report indexed/expected rows; detector evaluability; human and agent coverage; planner/group/outcome/control gaps; finding candidates/confirmations; open integrity findings; materialization fidelity; diagnostics and GitHub links. Output machine-readable JSON and readable Markdown/HTML. Coverage deficit records feed BA-02. Suggested status vocabulary: `incomplete`, `complete_with_declared_exceptions`, `complete_under_protocol`.

The legitimate completion statement is: “Human review coverage satisfies Audit Protocol v1 for campaign <digest>, subject to these declared exceptions.” Never output an unexplained “benchmark valid” badge or a percentage probability that no unknown bugs remain. Implementation completion can be achieved autonomously while actual human audit coverage correctly remains incomplete.

## 16. Proposed code ownership and maintenance

Prefer existing canonical owners. New modules are justified only for missing audit responsibilities; names below are proposed, not an obligation to create redundant files.

| Owner | Proposed modules |
| --- | --- |
| BA-03 | `audit_contracts`, `audit_store`, `audit_findings`, `audit_similarity` in `robot_sf/analysis_workbench/`. |
| BA-01 | `audit_scan`, `audit_detectors` and narrow detector implementations. |
| BA-02 | `audit_queue`. |
| BA-04 | `audit_coverage` and protocol configuration/report adapter. |
| BA-05 | `audit_service`, `audit_mcp`, `audit_codex`, `audit_materialize`, `audit_github`. |
| BA-06 | Extend `robot_sf/render/review_workbench.py` with scoped web components/integration tests; do not create a second renderer entry point. |
| SREV-16/17/18/24/28 issue deliverables | Extend their canonical issue-owned paths as they land. At this snapshot those issue modules are prerequisites, not already-existing panels/editor/diagnostics/loop/session implementations, and must not be duplicated under BA names. |

One coordinator serializes shared `pyproject.toml`, lockfiles, entrypoints, shell/bootstrap, schemas and `tests/conftest.py` registrations. Workers own non-overlapping implementation/test/docs paths. Use current dependencies where adequate; do not add a hosted database, vector store, heavyweight frontend framework or separate authentication platform by default. Maintain offline core operation and test optional provider dependencies separately.

Documentation must include one launch route, demo fixture, source/capability inspection, missing-artifact handling, annotation backup/restore, autonomous budgets/cancellation, Codex setup using existing local authentication, and exact unsupported cases. Proposed commands must not be documented as working until exercised.

## 17. Delivery sequence and non-interactive recovery

### Phase A — review and merge this plan PR first

Read current root/scoped guidance and the plan PR's complete diff, comments, exact head and CI state. A separate reviewer checks all 39 requirements, existing-owner reuse, evidence boundaries and scope. Fix concrete mistakes in this one file; keep the PR documentation-only. Use cheap documentation diff/link checks and the repository's appropriate instruction/config checks, not simulation campaigns. Reconcile PR metadata, then follow exact-head review and guarded CI/merge procedures. Do not close #9483 when the plan merges.

### Phase B — reconcile the bounded graph and establish actual source capabilities

Read the six BA issues plus #9285/#9287/#9288/#9296/#9299/#9417 and the relevant #9380 comments. Search active PRs/claims for these IDs; reuse or finish valid existing work instead of competing with it. Run body preflight and live admission using repository scripts, refresh drift and set readiness only from their real results. Set intended native parent/dependency links from the owning writable worktree and verify; preserve unrelated links and existing parent ownership. Batch Project #5 updates through existing routing. Do not scan or rewrite the entire repository issue graph.

The connector publication environment did not run canonical preflight/live admission, native relationship writes, Project #5 updates or runtime tests. These are explicit bootstrap actions for the implementation agent, not missing user decisions. Read actual helper `--help` before using an assumed option.

### Phase C — foundations and an early useful product slice

BA-03 first delivers minimal shared record contracts, commit-safe writes and fixtures. In parallel, complete the non-overlapping #9417 source-admission slice. Then route detectors, queue/coverage, editor/panels and service work against the same fixtures. Integrate one real retained failure and a normal control early: queue → synchronized inspection → quick/spatial annotation → durable finding → report. Do not postpone discovering interface incompatibility until every leaf is large.

### Phase D — full V1 coverage

Complete all detector families, active/control sampling, rich references, similarities, optional diagnostic panels, native-source materialization, actual Codex/MCP integration, bounded loop/session UI and idempotent GitHub sync. Preserve the selected broad V1 boundary; do not silently downgrade it to a CLI-only demo, mocked agent or a viewer that only handles synthetic data.

### Phase E — independent verification, repair and merge

Review current intended behaviour against actual implementation, not only green tests. Workers repair concrete findings. Run focused tests first, integrated exact-head readiness after integration, required hosted CI and guarded merges. Keep evidence/benchmark-sensitive review independent of the author where required. Re-read merge receipts and post-merge checks on the exact merge commit. Close only issues whose actual acceptance is met. Do not mark an unexecuted live integration complete through a mocked test.

### Recovery without user attention

Make engineering choices from these defaults and current canonical code; record consequential deviations. No requests for product answers, manual file copying, routine approvals or “which option?” pauses. Discover existing authenticated tools/checkouts and declared dependencies automatically, preserving user work. Repair bounded environment errors and retry transient transport failures using idempotent operations. For active ownership, work on another non-overlapping slice and recheck through the existing lease/claim mechanism; never delete another worker's work.

On a missing external credential, exhausted provider quota, inaccessible source or required protected merge that cannot be repaired with authorized tools, continue all independent work. Keep the exact blocked item, attempts, preserved progress and next machine-executable action in a durable ledger. Do not claim success or silently weaken security/evidence rules to satisfy unattended delivery. A real external blocker is not a reason to discard completed work or ask the user to repeat the requirements.

Persist compact progress after each accepted slice: task/issue, base/head, owned paths, commits/PRs, tests, unresolved findings, actual routing/usage and next action. Restore from it after context compaction or process interruption. A context-limit exit must not be presented as completed implementation.

## 18. Low-cost implementation-agent strategy

The requested coordinator tier is **sol/medium** and the default implementation/review worker tier is **luna/max**. These are task-request aliases, not hard-coded provider API IDs or a replacement global routing policy. Resolve them through the repository's existing local routing system and record actual selected provider/model/effort.

Read [agent workflow entrypoints](../ai/agent_workflow_entrypoints.md), including `CODEX_ROUTING_REPO`, the flat `handoff.v2` example and resolver output. Discover an existing shared-routing checkout through known configuration/worktrees; do not ask the user for a path already locally discoverable. Inspect installed CLI capabilities and help. Do not assume native Codex subagent controls can launch arbitrary external-provider models. Use the resolved supported adapter; never silently run all workers on an expensive default.

Cost allocation:

- Luna/max performs the bulk of bounded code, tests, documentation, migration adapters and first-pass reviews.
- Sol/medium coordinates shared interfaces, dependency/ownership decisions, integration, difficult failed cases and final exact-head review/merge disposition. It should not rewrite every worker patch or repeatedly read the entire repository.
- A reviewer must be a separate task/context from the author. Cheap independent review is useful; source identity, persistence, execution/security and evidence-boundary changes also need the coordinator's domain-focused review.
- No premium escalation above the requested tier without an already authorized local routing policy. Prefer a confirmed available equally cheap route after startup failure; disclose actual fallback and do not invent usage savings.

Start with at most three concurrent non-overlapping implementation workers, or fewer under local machine guidance. Review/merge shared files serially. Workers do not launch grandchildren. This bounds context duplication and merge conflicts; it is an engineering default, not a blanket restriction on unrelated compute.

Each worker gets the relevant issue, accepted interface fixture, exact base/head, small owned-path set, non-goals and test command. Target a compact initial packet of roughly eight files/8,000 tokens, expanding only for a named missing symbol. Do not attach the whole conversation, every SREV issue or raw CI logs. Reuse stable context and retrieve bounded error excerpts.

Use one initial worker attempt plus one focused repair attempt before coordinator diagnosis changes the approach; do not bounce the same failure indefinitely between agents. Startup failure before a worker begins is distinct from a code/test failure; use the existing routed-worker recovery manifest. Continue accepted work rather than restart a whole leaf. Record tokens/usage when exposed and call counts otherwise; monetary costs remain unknown unless actually measured.

Run deterministic tools for formatting, imports, schema validation, test collection, paths and diffs. Use fake providers for routine tests, one minimal real agent smoke, targeted tests per patch and the full appropriate suite at integration boundaries. Do not pay for repeated full-repository LLM audits or redundant full-suite runs without a changed dependency/risk reason. Correctness and source integrity are not traded away for token savings.

## 19. Acceptance and validation matrix

| Area | Required positive and negative evidence |
| --- | --- |
| Campaign/detectors | A second current-schema campaign; all expected rows accounted for; all twelve families; legitimate controls; unavailable/undefined telemetry; reproducible outputs. |
| Selection | Accepted priority precedence; reproducible fixed/active order; non-starved ordinary controls; redundancy without false review credit; manual pin/resume. |
| Storage | Crash at each canonical/index boundary; simultaneous clients; repeated operation IDs; stale edits; SQLite deletion/rebuild; backup/relocation. |
| UI/references | Unequal rates/origins/gaps; wrong PTS maps; cropped video; world/image references; keyboard typing; autosave failure; quick/full annotation. |
| Findings | Candidate versus confirmed counts; negative evidence; explained similarity; stale source; no causal promotion from similarity. |
| Agent/service | Exact UI context revision; shared CLI/browser/MCP semantics; artifact prompt injection; wrong origin/path; reconnect/cancel; fake and real provider receipts separated. |
| Materialization | Original recording, rendering saved states, supported native rerun; changed checkpoint/config, divergent/unverifiable replay and missing source; no policy fallback. |
| Diagnostics | Admitted source, matched unchanged control, measured activation, negative/inconclusive outcome, aggregate budget, deadline/cancel and resume. |
| GitHub | Existing finding link; duplicate detection; timeout after successful create; preservation of human edits; local-only media and private-data filtering. |
| Coverage | Hand-computed tiny strata; agent/clip/replay non-inflation; new protocol/release invalidation; explicit exceptions and unresolved critical findings. |
| End to end | Real retained failure plus normal control; Next → annotate → finding → related → diagnostic → issue → updated coverage; all required features accessible through one launch route. |

Use actual tests discovered in the current tree. Typical existing wrappers are:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest <focused-tests> -q
scripts/dev/run_worktree_shared_venv.sh -- uv run ruff check <changed-paths>
scripts/dev/run_worktree_shared_venv.sh -- uv run ruff format --check <changed-paths>
git diff --check
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

The angle-bracket arguments are placeholders resolved by the implementing agent, not commands to run literally. Match proof to risk: documentation-only changes use the cheap route; runtime/schema/source-integrity changes require their applicable tests and exact-head evidence. Nothing in this plan waives current repository invariants or claims that these validations already ran.

## 20. Definition of package completion

Completion means the selected V1 product is implemented, independently reviewed, tested at its declared capability boundary, documented and merged, with every requirement mapped to a source/test/receipt or an explicitly later extension. It does not mean all historical campaign assets are recoverable or the human has already reviewed the benchmark.

Final delivery must report exact merge revisions; implemented issue/PR map; one working launch/demo command; supported source/planner capabilities; test and real-data/agent smoke receipts; annotation preservation/restore instructions; autonomous policy and cancellation; actual routing/usage; unresolved external limitations; and remaining genuine audit findings. Do not substitute plans, synthetic-only demonstrations or “green CI” for required real behaviour.

The success criterion is practical: open a conforming campaign, receive an explained next case, inspect synchronized evidence, annotate precisely or quickly, discuss it with Codex, retain related findings and bounded investigations, synchronize the appropriate issue, and see an honest update of what still needs checking.
