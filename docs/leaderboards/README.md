# Static Leaderboards

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1866>

Static leaderboards are Markdown result surfaces for benchmark and smoke evidence that is already
tracked in this repository. They are intentionally small: each row must name its evidence URI,
status, benchmark track, and claim boundary so a scan-friendly table does not become an accidental
promotion claim.

## Pages

| Page | Scope | Population status |
| --- | --- | --- |
| [Smoke](smoke.md) | Policy-search and benchmark smoke rows with durable report or compact-summary evidence. | initial rows |
| [Nominal Sanity](nominal_sanity.md) | Nominal-sanity policy-search rows. | initial row |
| [AMV Actuation](amv_actuation.md) | Synthetic AMV actuation smoke and diagnostic rows; interpret through the [AMV actuation evidence ladder](../context/issue_2230_amv_actuation_evidence_ladder.md). | initial rows |
| [LiDAR 2D](lidar_2d.md) | LiDAR learned-policy smoke rows. | initial row |

## Row Contract

Every populated row should include these visible fields:

| Field | Meaning |
| --- | --- |
| `planner` | Planner, policy, or candidate identifier. |
| `suite` | Scenario suite, policy-search stage, or benchmark surface. |
| `success` | Mean success metric, or `not_recorded` when the durable evidence does not expose it. |
| `collision` | Collision metric from the evidence source, or `not_recorded`. |
| `near_miss` | Near-miss metric from the evidence source, or `not_recorded`. |
| `low_progress` | Low-progress count/rate when reported, or `not_recorded`. |
| `min_distance` | Minimum-distance metric when reported, or `not_recorded`. |
| `runtime` | Runtime or wall-clock metric when reported, or `not_recorded`. |
| `benchmark_track` | Named benchmark, policy-search stage, observation track, or `not_benchmark_evidence`. |
| `evidence_uri` | Repository-relative tracked evidence path. Do not point at worktree-local `output/`. |
| `status` | Evidence status such as `pass`, `revise`, `successful_evidence`, `not_available`, `excluded`, or `not_yet_populated`. |
| `claim_boundary` | One-line statement the row supports; sidecars validate it as `claim_wording` and also carry `claim_boundary` caveats. |

## Machine-Readable Row Claims

Each leaderboard page has a sidecar `<page>.rows.json` file containing row-level
`BenchmarkClaim.v1` profile records (`benchmark_row_claim.v1`) for every populated row. The
sidecar enforces:

- `suite_id`, `planner_id`, `planner_mode` (`native`/`adapter`/`fallback`/`degraded`/`not_available`)
- `seeds`, `metrics`, `row_status`, `exclusions`, `artifact_uri`, `claim_wording`,
  `evidence_tier`, `claim_boundary`
- `exclusions` is intentionally required; use `"none"` only for rows with complete evidence and
  no caveats.
- A fail-closed rule that rejects `output/` artifact URIs and missing tracked artifacts
- Deterministic wording policy that rejects claims above the row's evidence tier or status

Validate one sidecar:

```bash
uv run python -m robot_sf.benchmark.cli validate-row-claims \
  --sidecar docs/leaderboards/smoke.rows.json
```

Validate every leaderboard sidecar:

```bash
uv run python -m robot_sf.benchmark.cli validate-row-claims --all
```

Sidecar files are the source of truth for the row claim contract; the Markdown
 table is the human-readable rendering.

## Evidence Rules

- Use tracked reports, tracked compact summaries, release manifests, or other durable repository
  evidence as `evidence_uri`.
- Do not populate rows directly from worktree-local `output/` paths. If a report mentions an
  `output/` summary but no compact tracked evidence exists, link the tracked report and mark any
  missing metrics as `not_recorded`.
- Keep fallback, degraded, unavailable, excluded, and failed rows visible as caveats. They are not
  successful benchmark evidence unless the issue explicitly measures that mode.
- Keep smoke, diagnostic, training-smoke, and synthetic AMV rows separate from paper-facing
  benchmark claims.
- Prefer `not_yet_populated` over inventing a metric or copying an untracked artifact.
- Update the matching `<page>.rows.json` sidecar whenever a row is added, removed, or downgraded.
  The validator will reject Markdown rows that have no corresponding machine-readable claim record,
  but the sidecar is the enforcement surface.

## Maintenance

Update these pages when a tracked evidence bundle, policy-search report, benchmark release, or
claim-map decision changes the row status. For bulk population or automation, preserve the row
contract above so generated tables retain the same evidence boundary.

Run `uv run python scripts/validation/validate_platform_docs.py` before publishing leaderboard,
planner-zoo, policy-card, or benchmark-suite catalog updates.

Run `uv run python -m robot_sf.benchmark.cli validate-row-claims --all` to check that every
leaderboard sidecar still satisfies the `BenchmarkClaim.v1` contract.
