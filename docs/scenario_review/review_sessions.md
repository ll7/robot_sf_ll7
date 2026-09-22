# SREV-28 review sessions

`robot_sf.render.review_sessions` is the standalone local workbench surface
for inspecting and explicitly starting a bounded experiment session. It is a
thin adapter over the SREV-24 experiment loop: SREV-24 remains the owner of
the executor, operation IDs, journal, lock, budget accounting, cancellation,
crash recovery, and retained outcomes.

Every result is diagnostic-only. A fixture session is not benchmark evidence,
scientific evidence, a causal claim, or permission to publish a result.

## Preview before start

The default CLI command creates a small `review-session-preview.v1.json` and
prints a component-result envelope. Preview is read-only with respect to input
sources and never dispatches an executor:

```bash
python -m robot_sf.render.review_sessions \
  --input tests/fixtures/scenario_review/review_sessions/request.json \
  --config tests/fixtures/scenario_review/review_sessions/config.json \
  --output output/scenario_review/srev-28-smoke
```

The preview shows the effective candidate, execution, retry, and elapsed-time
ceilings; consumed/reserved/remaining budget; deterministic candidate order;
source-admission state; preservation destination/receipt state; output
collision state; and the explicit-start requirement. A missing launcher-owned
admission configuration is shown as `required`; it is not guessed from the
recipe.

The checked-in fixture is synthetic and deliberately tiny. The preview is the
expected smoke artifact. It does not run a simulator. A real/native start
requires an explicit caller-owned admission companion:

```bash
python -m robot_sf.render.review_sessions \
  --input tests/fixtures/scenario_review/review_sessions/request.json \
  --config tests/fixtures/scenario_review/review_sessions/config.json \
  --admission-config <launcher-admission.json> \
  --output output/scenario_review/srev-28-run \
  --start --autonomous
```

The launcher admission document is passed separately so a recipe cannot
nominate its own source root or preservation authority. An injected executor
used by offline tests must receive a measured, request/recipe-bound
`source_admission` proof below the invocation base.

## API and lifecycle

```python
from robot_sf.render.review_sessions import ReviewSession, run

preview = ReviewSession(request, base=output_root).preview()
result = ReviewSession(request, base=output_root, admission_config=admission).start()
resumed = ReviewSession(request, base=output_root, admission_config=admission).resume()
progress = ReviewSession(request, base=output_root).progress()
current = ReviewSession(request, base=output_root).result(index=0)
```

`run(request)` without `autonomous=True` returns
`unavailable: autonomous_start_authorization_required` and does not call an
executor. `run(request, read_only=True)` returns
`unavailable: read_only_never_executes`. `start()` and `resume()` delegate to
SREV-24; they do not create a second journal or reset consumed attempts and
elapsed budget. `stop()` sends cancellation through the same SREV-24 journal
owner and waits for that owner to settle. `start()`, `resume()`, and `stop()`
on one `ReviewSession` are serialized, including the fresh-output stop race. A
different controller cannot cancel a live owner; it receives the delegated
`session_lock_owned` diagnostic and the original owner must issue `stop()`.
`progress()` and `result_navigation()` only read the durable journal or report
and never acquire a dispatch lock. Read APIs revalidate journal/report
identity against the selected request, recipe, source bytes, and session
context before exposing it. A mutable `running` journal is readable or
recoverable only while its token-keyed lifecycle lease and current
process-local lifecycle anchor are valid; a settled journal/report pair is
readable only with its token-keyed integrity seal. A durable running lease is
not an owner credential by itself. Stale or tampered state is reported as
diagnostic failure/unavailable rather than adopted as current progress.

Because the output directory is writable by its owner, semantic consistency or
an unkeyed SHA-256 cannot authenticate a rewritten result. Every settled state
therefore requires a caller-supplied `session_token`; the wrapper writes
`review-session-integrity.v1.json`, an HMAC over the exact journal/report bytes,
their status, and their diagnostic identity. While SREV-24 owns a mutable
running journal, the wrapper keeps a separate HMAC lifecycle lease so a
same-process crash-safe resume can still use SREV-24's operation-ID recovery.
The token is never persisted. Reconnect/resume/read calls must provide the same
token, while a missing or rotated token and a changed journal/report fail
closed. A process restart cannot adopt a saved running lease without an
independent trusted owner/monotonic store, so running progress and resume fail
closed after restart. This is a diagnostic integrity boundary, not scientific
evidence.

The lifecycle revision is also held in a process-local monotonic anchor. This
rejects an old valid complete triplet replayed after a newer cancellation in
the same process. Owner-writable files cannot provide an anti-rollback root
across process restarts: a restarted process may inspect an authenticated
complete snapshot diagnostically, but that snapshot cannot authorize a new
resume or control operation without the current lifecycle anchor. It also
cannot adopt a saved running lease as a new owner. A durable external
monotonic store or owner service would be required for stronger cross-process
rollback protection or crash recovery; this component does not pretend that a
plain digest provides either.

The optional `session_context` request mapping carries the selected
`campaign_id`, `episode_id`, `source_revision`, `selection_revision`, and
`context_revision`. The delegated session ID also binds the canonical request
and recipe digests, and those digests are included in the browser
view/control envelope. Changing selected context, source identity, or recipe
cannot therefore reuse another session's durable results.

Complete output contains the delegated
`experiment-loop-report.v1`/`experiment-loop-session.v1` artifacts plus
`review-session.v1.json`, `review-session.v1.html`, the local browser module,
and the token-keyed integrity seal. Partial, failed, unavailable, and
cancelled results carry no complete artifacts, while the delegated
journal/report and settled-state seal remain available for diagnosis and an
explicit resume when SREV-24 permits it. Native SREV-24 admission records
retain their launcher
receipt/root/preservation shape; when a native complete session is read or
resumed, pass the launcher admission companion again for fresh receipt/root
validation. Injected fixture proofs use their separate measured shape and must
remain below the requested component base.

## Offline browser and local controls

The emitted HTML references only
`components/review_sessions/review_sessions.js`. The module has no CDN,
network, browser storage, scheduler, simulator, or arbitrary-code hook. It
renders the supplied snapshot and exposes start/stop/resume/progress/result
navigation as explicit actions. A local adapter must inject
`controlRequest(envelope)` and authenticate the envelope's exact loopback
origin and session token with `validate_control()` before invoking a session.
No browser callback is treated as execution authorization by itself.

```python
from robot_sf.render.review_sessions import make_control_handler

handler = make_control_handler(
    session,
    origin="http://127.0.0.1:8765",
    session_token="launcher-issued-token",
)
```

The origin may be `localhost`, `127.0.0.1`, or `::1` over HTTP(S), with no
path/query/fragment or embedded credentials. Tokens are compared in constant
time and controls carry the bound session ID, context revision, request digest,
and recipe digest. Remote origins,
missing tokens, mismatched origins, stale context, unknown actions, and
read-only controls fail closed. Standalone CLI `--stop` additionally requires
the request to carry the session-owned `origin` and `session_token`; a
non-empty command-line token alone is never sufficient. When
`--admission-config` is omitted, the CLI may recover the persisted native
launcher fields only as input to SREV-24's fresh receipt/root/preservation
validation. Injected or incomplete sessions fail closed rather than treating
their durable proof as native admission.

## Evidence and preservation boundary

The component records `evidence_boundary: diagnostic_only`,
`scientific_claim_allowed: false`, and
`dependent_family_status: standalone_fixture_only`. Source files are
immutable. Paths are relative and contained under the caller's base; output
collisions, traversal, symlinks, malformed/non-finite JSON, incompatible
versions, missing capabilities, and stale admission state are rejected or
reported unavailable. The SREV-24 journal is the crash-safe recovery
authority; this adapter does not copy or reinterpret its accounting. The
short-lived lifecycle lease is kept in a contained, hidden state directory
under the same base and is removed after each settled state.
