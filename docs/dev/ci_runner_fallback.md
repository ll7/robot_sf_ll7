# Hosted-first CI routing and security review

CI uses `ubuntu-latest` by default. Self-hosted runners are optional disposable
accelerators; removing every registered runner requires no code change. All
existing jobs, six test shards, path selections, result gates and timeouts remain.
The availability/recovery owner is `scripts/ci/runner_fallback.cjs`; trust and
commit-provenance enforcement remain in the workflow and job-started hook.

## Availability and kill switch

`ROBOT_SF_SELF_HOSTED_CI_ENABLED` must be exactly `true` to allow acceleration.
Unset, `false`, or any other value selects hosted on the next routing decision.
This switch does not migrate jobs already queued; the watchdog covers that race.
It remains off until reviewed infrastructure is ready. Changing repository
variables or secrets is a separate authorized settings action.

The hosted `runner-availability` job runs after admission, before workload jobs.
It reads main's base source for PRs targeting main, probes the repository runners
API with a five-second abort signal and decision bound and no retries, and treats online, idle Linux x64 registered
runners carrying the reviewed ephemeral label as available. That is a GitHub
heartbeat, not proof of application health. Busy/offline/missing runners,
incomplete pagination, malformed inventory, duplicate IDs, API errors and
timeouts select hosted. Each singleton costs one idle slot; the six test shards
cost six slots. Inventory is a snapshot, not a reservation across workflow runs.

The repository runner-list endpoint requires Administration **read**, which the
normal `GITHUB_TOKEN` cannot request. The optional `CI_RUNNER_READ_TOKEN` must be
a fine-grained token limited to this repository with Administration read only,
or an equivalently scoped installation token. It is consumed only by the hosted
probe step. With no token, the normal job token is tried; a denied response
selects hosted. This PR neither creates credentials nor enables acceleration.
The probe step runs only for the author as both actor and rerun initiator, on
push or an author-owned same-repository PR targeting main, and only on attempt one. Forks,
bots, manual dispatch, privileged events and every retry stay hosted.

## Queue recovery

The separate default-branch `CI runner queue watchdog` watches first-attempt CI
runs. It never checks out a triggering PR, restores its cache, or downloads its
artifacts. Every minute it reads the exact run attempt and all job pages. After
ten minutes of continuously **observed** self-hosted queue time (normally a
10–11 minute bound from first observation), it rechecks the current head and
queue, cancels that CI run, waits up to six minutes for confirmed cancellation,
and reruns **all** jobs once. The new attempt is forced hosted, including manual
retries. The watchdog reads back the new attempt before reporting recovery.
The jobs API has no documented queue-entry timestamp; dependency waiting is
not counted. Running jobs are not treated as queued jobs.

Recovery preserves the run's commit and check identity. Cancelling the whole
run can discard successful work and repeat uploads; it is necessary because
GitHub provides no job-level cancel or supported relabel operation. Superseded
main commits, closed/draft/updated/fork PRs, other workflows, moved attempts and
completed runs are declined. API errors, incomplete job inventory, failed
cancellation, or an unconfirmed rerun fail the watchdog with no blind retry.
It exits when every workload job has started, finished or been routed hosted.
It observes for at most 110 minutes of wall-clock time and reserves seven minutes
for cancellation/rerun before any write; its hosted job timeout is 120 minutes.

GitHub control-plane outages, hosted queue delays and watchdog failure can
exceed the normal bound. Inspect the watchdog's `ci-runner-watchdog` decision
logs before repair. A cancelled first attempt can be rerun manually; attempt
two is always hosted. Do not automatically retry an ambiguous POST. The
watchdog is active only after its workflow is present on the default branch;
pre-merge PR CI stays hosted if its base lacks the routing module.

## Threat note for independent review

- **Tokens:** workload jobs retain their read-only job permissions and no
  inventory token. Checkout credential persistence is disabled. The hosted
  router reads trusted PR base code; it never imports the writable PR head.
  Inventory/error responses and runner names are never logged. The watchdog
  has repository `actions: write` only for cancel/rerun, plus `contents: read`;
  no registration, settings, secret, deployment or repository write authority.
  A repository secret remains accessible to write collaborators who can edit
  workflows: the main-base checkout is defense in depth, not protection from
  existing repository write authority. Scope the optional token to this single
  repository and read only. Stronger protection would require a separately
  authorized environment or default-branch inventory service.
- **Who may execute self-hosted:** the existing author/event/repository
  expression and immutable runner hook remain necessary admission boundaries.
  Availability never expands trust. A contributor can edit a workflow, so the
  execution hook must reject direct-label attempts. Commit provenance work in
  issue #10210 remains independently owned and must be integrated separately.
  The inventory token must never be added to workload jobs or the runner image.
- **Fail closed:** missing inputs, credentials, routing output, trust fields,
  complete inventory or heartbeat select hosted. A broken hosted probe job
  does not skip admitted workload jobs. Recovery refuses stale/ambiguous state
  and never promotes an untrusted run or loops on its own retry.
- **Residual races:** inventory may change before scheduling; the watchdog
  repairs that liveness failure. GitHub has no compare-and-swap cancel endpoint;
  a job can start after the final read and be cancelled. Head updates or human
  reruns between API reads/writes may also race. No security acceptance may
  claim stronger atomicity than the API supports.
- **Activation:** independent security review must cover the exact final SHA,
  both hosted authority jobs, all five expressions, fixtures and this note.
  Settings/credential activation remains a separate owner action. No live
  cancellation or rerun is exercised by local tests.

## Heavy jobs and hosted capacity proposal

No current CI job is declared self-hosted-only. No job is silently dropped.
Public Linux `ubuntu-latest` currently supplies 4 CPUs, 16 GB RAM and 14 GB
documented SSD capacity. The repository's all-extras environment has previously
downloaded about 9 GB; checkout, build scratch and cache restore can exceed
that storage budget. This is a configuration risk, not a measured per-job
peak in this PR. Preserve the existing cleanup and uv caches while measuring
peak disk/RAM and elapsed time on hosted runs.

| Job family | Existing hosted configuration | If hosted capacity is exceeded |
| --- | --- | --- |
| `fast-feedback` | Six complete-admission shards, 45m each, uv/model/duration caches | Split slow shards further while preserving coverage union; use a larger Linux runner with at least 32 GB RAM and 75 GB SSD if any single test exceeds 16 GB. |
| `smoke-artifacts`, `wheel-smoke-install`, `notebooks-smoke` | Linux, 30m, shared setup/cache and disk cleanup | Cache locked dependency downloads; narrow extras to actual imports after dependency proof; larger runner for measured disk/RAM overflow. |
| `examples-smoke` | Linux, 30m, examples-only dependency group | Retain the narrower environment; split the manifest with an aggregate result if elapsed time exceeds 30m. |
| `compat-matrix`, `fast-pysf-compat` | Existing OS/Python matrices | Keep platform coverage; split setup/build from test caches rather than deleting an OS. |
| `scenario-validation`, `xdist-scratch-isolation` | Existing Linux validation jobs | Reduce process fanout only after measuring peak RAM; split scenarios with a complete result gate if needed. |
| `determinism-gate`, `exact-repeat-model-preflight` | Existing path-triggered 30m jobs with digest-pinned model caches | Separate hydration from timed offline repeats, retain model assets and every repeat; larger disk runner if model hydration exceeds standard storage. |
| Coverage gates, dispatch ownership and aggregate `ci` | Existing hosted control/aggregation jobs | Keep hosted; shard artifact merging if measurement shows a memory limit. |

Larger-runner access/pricing and per-job peak measurements require an owner
decision if limits are actually reached; this change requests no settings.

## Cheap validation and upstream contracts

Run `node --test tests/ci/runner_fallback.test.cjs` for mocked API decisions,
capacity, queue timing, recovery order, trust/staleness and failed cancellation.
Run the focused Python workflow tests for actual expression evaluation and
credential boundaries. Mocked transport tests are implementation evidence;
they do not establish live GitHub cancellation or inventory-token access.

- [Runner inventory API and permissions](https://docs.github.com/en/rest/actions/self-hosted-runners#list-self-hosted-runners-for-a-repository)
- [Run cancel and rerun API](https://docs.github.com/en/rest/actions/workflow-runs)
- [Job API](https://docs.github.com/en/rest/actions/workflow-jobs)
- [Self-hosted queue behavior](https://docs.github.com/en/actions/reference/runners/self-hosted-runners)
- [Default-branch workflow_run contract](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run)
- [Hosted runner capacities](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
