# CARLA native↔aligned parity: conservative claim-boundary report (#1511)

**Status:** conservative claim-boundary report (proposal-tier). **Parent umbrella:** #1491.
**Scope:** separate **fixture parity**, **runtime parity**, and **remaining transfer caveats**
from the existing CARLA replay-parity evidence. Running the parity bundle itself is out of scope
(#1510); this note does not run CARLA.

**Bottom line:** as of the most recent recorded attempt (2026-05-21), **metric-level
native↔aligned parity is NOT established.** Fixture staging and CARLA runtime connectivity were
demonstrated, but the live run failed closed before metric-producing oracle replay, so no positive
parity claim is licensed.

## Evidence reviewed

| Stage | Bundle | What it shows |
| --- | --- | --- |
| Setup smoke | `docs/context/evidence/issue_1111_carla_setup_smoke_2026-05-18/` | CARLA availability + certified #1111 payload staged |
| Live replay | `docs/context/evidence/issue_1169_carla_live_replay_2026-05-18/` | Live replay with #1329 static-geometry support (static-boundary case) |
| Live parity attempt | `docs/context/evidence/issue_1430_carla_live_parity_2026-05-21/` | CARLA 0.9.16 client↔server on `Town10HD_Opt`; run **failed before oracle-replay** |

## Parity, separated by layer

### 1. Fixture parity — **partial / staged**
The certified #1111 payload is staged and the setup smoke records CARLA availability. The fixture
inputs (payload, static geometry after #1329) are in place. This is staging/availability evidence,
**not** a metric-parity claim.

### 2. Runtime parity — **connectivity established, run failed closed**
The #1430 attempt connected a CARLA 0.9.16 Python client to a CARLA 0.9.16 server on
`Carla/Maps/Town10HD_Opt`. The earlier #1169 blocker (`T0 payload static obstacle replay is not
implemented`) was **not** the observed failure after #1329. However, the run failed closed before
metric-producing oracle replay:

```text
status: failed
mode: failed
reason: CARLA failed to spawn robot
```

So runtime *connectivity and prerequisites* are demonstrated, but a runtime that *reaches
oracle-replay* is not.

### 3. Metric parity — **UNAVAILABLE (no claim licensed)**
`issue_1430_carla_live_parity_2026-05-21/parity_report.json`
(`comparison_schema: carla_oracle_replay_parity_v1`) reports every metric — `success`, `collision`,
`ttc_min_s`, `min_distance_m`, `comfort`, `jerk` — with `status: unavailable` and
`reason: "CARLA replay mode/status is not native/comparable: failed"`. No `carla_value`, no
`delta`. **No native↔aligned metric parity has been demonstrated**, in either direction.

## Remaining transfer caveats

- **Host/route:** the preferred `imech036` route failed non-interactively (SSH host-key
  verification); the fallback `imech156-u` host was used. Parity evidence is host-dependent and not
  laptop-local-provable (a capable CARLA host is required).
- **Spawn-failure blocker:** the robot actor-spawn failure (follow-up **#1437**) blocks any
  metric-parity claim until diagnosed and resolved.
- **Fail-closed policy:** per the benchmark fallback policy, the failed/`unavailable` rows must
  **not** be treated as parity success, dry-run, or launch-packet evidence.

## Claim boundary (what may and may not be said today)

- **May say:** CARLA fixture staging and 0.9.16 client↔server runtime connectivity on `Town10HD_Opt`
  are demonstrated; the #1169 static-obstacle-replay blocker is resolved (#1329).
- **Must NOT say:** that Robot SF metrics have been validated against, or shown to transfer to/from,
  CARLA. Metric parity is unavailable; no success/collision/TTC/clearance/comfort/jerk parity exists.
- **Minimum artifact to strengthen:** a CARLA run that reaches `oracle-replay` and emits comparable
  metrics (blocked by #1437), executed via the parity bundle (#1510) on a capable host, with
  `carla_oracle_replay_parity_v1` deltas recorded.

## Related

- Parent umbrella: #1491. Parity bundle execution: #1510. Spawn-failure diagnosis: #1437.
- Earlier parent epic referenced in the evidence: #872.
- Benchmark fallback / fail-closed policy: `docs/context/issue_691_benchmark_fallback_policy.md`.
