# Issue #4207 Interacting Certification-Transfer Smoke Evidence

> **Synthetic CPU-scale smoke, not a physics run.** This packet is generated in report-only mode
> from a hand-authored, deterministic episode fixture
> (`tests/benchmark/fixtures/issue_4207_interacting_smoke/interacting_smoke_episodes.jsonl`), **not**
> from a robot/pedestrian simulation. Its sole purpose is to exercise the interaction-validity
> guard's positive path (`model_sensitivity_exercised = true`) end-to-end over a committed
> interacting scenario family. See `SYNTHETIC_SMOKE_NOTICE.md`.

## Why this packet exists

The companion `francis2023_blind_corner` probe
(`docs/context/evidence/issue_4207_certification_transfer_probe_2026-07/`) recorded cells that were
all `non_interacting` (`robot_ped_within_5m_frac = 0`, min clearance ~20 m). Because
`social_force_default` (social-force model, SFM) and `hsfm_total_force_v1` (headed social-force
model, HSFM) only diverge inside the 5 m pedestrian near field,
that run's stable transfer statuses were **vacuous** and the guard reported
`model_sensitivity_exercised = false`.

This interacting smoke family drives every transfer cell into the near field so the guard reports
`model_sensitivity_exercised = true`, and includes a genuine interacting flip (the `ppo` arm passes
under `social_force_default` but fails under `hsfm_total_force_v1`) — model-assumption fragility of
the certification decision, exercised rather than vacuous.

The provisional release gates are not deployment approval. Missing gate metrics are `not_evaluable`,
never `pass`. Transfer flips are reported as model-assumption fragility in the gate decision, not as
a failed experiment.

## Counts

- Gate status counts: `{'pass': 5, 'fail': 3}`
- Transfer status counts: `{'stable_pass': 9, 'fragile_pass_to_fail': 1, 'conservative_fail_to_pass': 1, 'stable_fail': 5}`
- Interaction status counts: `{'interacting': 16}`
- Model sensitivity exercised: `True`
- Flip cases: `2` (both on the `ppo` arm; interacting, so non-vacuous)

## Reproduce

```bash
uv run python scripts/benchmark/run_certification_transfer_issue_4207.py \
  --config configs/benchmarks/issue_4207_interacting_smoke_probe.yaml \
  --gate-spec configs/benchmarks/release_gates/issue_4207_interacting_smoke_gates.yaml \
  --output-dir docs/context/evidence/issue_4207_interacting_smoke_2026-07 \
  --episodes-jsonl tests/benchmark/fixtures/issue_4207_interacting_smoke/interacting_smoke_episodes.jsonl \
  --generated-at 2026-07-03T00:00:00+00:00
```

## Files

- `summary.json`
- `metadata.json`
- `certification_gate_cells.csv`
- `certification_transfer_matrix.csv`
- `metric_deltas_by_model.csv`
- `flip_cases.csv`
- `claim_boundary.md`
- `SYNTHETIC_SMOKE_NOTICE.md`
- `SHA256SUMS`
