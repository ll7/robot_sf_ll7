# Reproduction — Package A decision packet (issue #3078)

- Execution date: 2026-07-03
- Base commit (origin/main): `79f6dab2dc498e61baf71bf033e1c0f838cad7db`
- Host: imech036 (CPU only; no compute submission, no campaign execution)
- All commands run via the worktree shared-venv wrapper
  (`scripts/dev/run_worktree_shared_venv.sh`), which pins `PYTHONPATH` to the
  worktree while reusing the primary checkout `.venv`.

## 1. Package A readiness checker (fail-closed)

```bash
uv run --extra analytics python scripts/validation/check_package_a_readiness.py --json
```

Result: `status: ready` — `missing_paths: []`, `issues: []`.

## 2. Held-out-family partition validator

```bash
uv run python scripts/tools/validate_heldout_transfer_partitions.py \
  configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml
```

Result: `validated held-out transfer partition manifest: …partitions.yaml`
(exit 0).

## 3. Seed-sufficiency CLI surface probe

```bash
uv run python scripts/tools/analyze_seed_sufficiency.py --help
```

Result: CLI present; usage printed. No frozen Package A protocol campaign root
with `reports/seed_variability_by_scenario.json` is retained, so no analysis
was produced (see README).

## 4. Package A decision packet (assembly)

```bash
uv run --extra analytics python scripts/validation/check_package_a_readiness.py \
  --decision-packet --json \
  --heldout-partition-manifest \
  configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml
```

Result: `classification: blocked_pending_package_a_evidence`
(`readiness_status: ready`). Reasons: no canonical campaign result store
supplied; no seed-sufficiency analysis report supplied. Partition manifest
validates (`ok: true`). Full output in
[`package_a_decision_packet.json`](package_a_decision_packet.json) (the
committed copy has its `manifest_path` normalized to a repo-relative path).

## Focused tests

```bash
uv run python -m pytest tests/validation/test_check_package_a_readiness.py -q
```

(See PR body for the recorded pass/fail line.)
