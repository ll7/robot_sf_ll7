# Autonomous issue implementation admission

Autonomous implementation must use the repository label state and a complete issue contract. A Project field, title, priority, open state, or absence of a blocker is not sufficient.

## Admission rule

An implementation worker may acquire `agent-claims/issue-<number>` only when the issue:

- is open, has exactly one `state:*` lifecycle label, and is labeled `state:ready`;
- is unassigned and has no existing atomic claim;
- is not a parent, epic, decision, review, active-work, compute, campaign, external-input, or blocked issue;
- contains non-empty sections for the objective, scope, inputs or affected surfaces, acceptance criteria, and verification.

## Repository execution contract

An explicit execution declaration keeps implementation ownership in one repository and makes
cross-repository routing visible before a claim is acquired. Existing issue bodies without this
section retain the historical single-repository-local default; new cross-repository work must
declare the contract below under an `Execution` or `Execution contract` heading:

```yaml
execution:
  owning_repo: ll7/robot_sf_ll7
  mutation_repos:
    - ll7/robot_sf_ll7
  route_required: local
  external_inputs: []
```

The live checker returns `wrong_owner_repo` when `owning_repo` or `mutation_repos` leaves the
current repository, or when a local route declaration is inconsistent. It may accept an explicit
`multi_repository` declaration only with a fresh route-plan artifact supplied through
`--route-preflight-json`; the artifact must include a selected route, a routing-config digest, and
a timezone-qualified timestamp no older than 30 minutes. The route configuration must be
re-probed when its digest changes, at the start of a new autopilot cycle, after a new user request,
or after the freshness window expires. A stale or missing route is not an implementation claim.

`external_inputs` must be empty for local autonomous implementation. A non-empty declaration is
reported as `external_input_missing` until its separate input contract is satisfied. A successful
`goal_issue_admission.py --check-only` result is the only claimable verdict; a label-filtered
snapshot is a candidate queue until that live check succeeds.

The checker reports each missing field separately. It never converts a numeric completeness score into a pass.

## Commands

Check an issue without a write:

```bash
uv run python scripts/dev/issue_implementability.py <issue-number>
uv run python scripts/dev/goal_issue_admission.py <issue-number> --check-only
```

Acquire the existing atomic claim only after the live preflight passes:

```bash
uv run python scripts/dev/goal_issue_admission.py <issue-number>
```

Offline contract fixtures can use a body file and explicit labels:

```bash
uv run python scripts/dev/issue_implementability.py 1 \
  --body-file /tmp/issue.md \
  --label state:ready \
  --title "fixture issue"
```

## Outcomes

- `ready`: the generic implementation contract permits claim admission.
- `needs_ready_label`: a maintainer or authorized preparation phase has not marked the issue ready.
- `needs_spec`: one or more required contract fields are missing.
- `parent`: dispatch a bounded child, not the tracker.
- `human_decision`: obtain the named ruling before implementation.
- `needs_compute`: route through the compute owner, not the local implementation lane.
- `blocked`, `working`, `review`, `assigned`, or `already_claimed`: do not start another implementation worker.
- `wrong_owner_repo`, `state_conflict`, or `stale_running`: reconcile repository ownership or
  lifecycle state before considering the issue again.
- `error`: re-read the live state; do not infer readiness.

## Authority boundary

This is a generic implementation gate. Research campaigns still require `research_answerability.v1`. Result and claim interpretation still require their specialized evidence and review contracts. A successful claim does not imply validation, review, merge readiness, evidence admission, compute authorization, release authorization, or issue completion.

## Queue snapshot fields

`snapshot_issue_batch.py` and the compact autopilot issue queue expose an `admission` object for
route evidence. Its `classification`, `reasons`, `ready`, `write_allowed`, `outcome`, and
`claim_outcome` fields are projections of the canonical wrapper; snapshot producers must not
reimplement the admission precedence. A `ready_check_only` result never writes a claim. An
`error` or `unavailable` result is a fail-closed hold.

For a write-enabled admission, the wrapper performs a complete second live preflight immediately
before the atomic claim attempt. A changed issue state, body digest, label, title, assignee, or
claim state fails closed with `not_admitted` and `write_attempted: false`.

The low-level `issue_claim.py acquire` command is rejected by default. Maintainer incident or
forensic recovery must explicitly provide `--manual-override`, `--override-actor`,
`--override-reason`, and `--no-scientific-claim`; this path is not ordinary autonomous dispatch.

## Blocker transition reconciliation

Queue snapshots and blocked receipts may also expose a read-only
`blocker_transition_plan.v1` projection from
[`blocker_transition.py`](../../scripts/dev/blocker_transition.py). It preserves distinct blocker
classes, exact owners, source and freshness keys, required child or PR links, and secondary
blockers; it is not a numeric priority or a second dependency resolver.

Operators and autonomous workers must keep the following boundary:

- `report-only` and `plan` never write. A ruling must carry an exact token and carrier, and a
  dependency link alone never proves satisfaction.
- A valid ruling without an executable bounded child remains `ruled_pending_child`. A satisfied
  dependency still requires a fresh `issue_implementability.v1` result before `state:ready` can be
  proposed.
- `apply` is exact-item only. It requires explicit authorization, an expected plan digest, an
  observed issue body, live state/label/body compare-and-swap, and current source revalidation for
  ruling, dependency, child, PR, or implementability observations. Concurrent drift aborts the
  operation; no repository-wide apply mode exists.
- Closed-item label cleanup remains delegated to `#7651`. No transition plan authorizes merge,
  issue closure, compute, publication, evidence admission, release, or scientific claims.

For an offline plan, use `uv run python scripts/dev/blocker_transition.py --issue <number>
--issue-json <fixture> --mode plan`. An apply caller must additionally provide the exact digest,
explicit authorization, and a current source projection through `--revalidated-sources-json`.
