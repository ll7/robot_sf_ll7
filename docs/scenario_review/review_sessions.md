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
owner. `progress()` and `result_navigation()` only read the durable journal or
report and never acquire a dispatch lock.

Complete output contains the delegated
`experiment-loop-report.v1`/`experiment-loop-session.v1` artifacts plus
`review-session.v1.json`, `review-session.v1.html`, and the local browser
module. Partial, failed, unavailable, and cancelled results carry no complete
artifacts, while the delegated journal/report remain available for diagnosis
and an explicit resume when SREV-24 permits it.

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
path/query/fragment. Tokens are compared in constant time. Remote origins,
missing tokens, mismatched origins, unknown actions, and read-only controls
fail closed.

## Evidence and preservation boundary

The component records `evidence_boundary: diagnostic_only`,
`scientific_claim_allowed: false`, and
`dependent_family_status: standalone_fixture_only`. Source files are
immutable. Paths are relative and contained under the caller's base; output
collisions, traversal, symlinks, malformed/non-finite JSON, incompatible
versions, missing capabilities, and stale admission state are rejected or
reported unavailable. The SREV-24 journal is the crash-safe recovery
authority; this adapter does not copy or reinterpret its accounting.
