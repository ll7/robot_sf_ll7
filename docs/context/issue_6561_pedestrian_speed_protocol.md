# Issue #6561 pedestrian desired-speed protocol

This note records the frozen, check-only protocol for comparing three pedestrian desired-speed regimes on the same six scenarios, four planners, and 30 seeds.

## Status and claim boundary

Status: **current protocol-only preparation**, not benchmark evidence. No registered episode is stored or authorized by this packet. The production campaign remains gated on retrieval and integrity validation of the native #6102 robot-speed result bundle, followed by the separate activation and submission gates. Fallback, degraded, incomplete, duplicate, missing, provenance-invalid, and cap-inactive rows cannot support a result claim.

The tracked source of truth is [`configs/benchmarks/issue_6561_pedestrian_speed_protocol.yaml`](../../configs/benchmarks/issue_6561_pedestrian_speed_protocol.yaml). Its exact manifest is compiled without running an episode:

```bash
uv run python scripts/validation/check_issue_6561_pedestrian_speed_protocol.py --manifest
```

The checker must report 2,160 identities and manifest hash `371f1a0160ec7faf1ade531691f104e2a1c92f7c34857e887ba1ba539e1b5238`. The upstream robot-speed prerequisite is pinned separately as #6102 manifest hash `e32ce197149af62bf366f5ca95abbb42215b379fe7916d916ccdd544dce8666f`.

## Frozen design

- Six medium classic-interaction scenarios: head-on corridor, doorway, group crossing, merging, overtaking, and station platform. Each source file and the composed scenario matrix is SHA-256 pinned in the protocol.
- Four planners: `scenario_adaptive_hybrid_orca_v2_collision_guard`, PPO, ORCA, and `prediction_planner`. The three file-backed planner configurations are SHA-256 pinned; ORCA uses its native registry binding.
- Seeds `111–140`, horizon `600`, timestep `0.1 s`, native execution, and robot speed cap `2.0 m/s`.
- Three pedestrian regimes: released `legacy_default` with the speed tier unset; `slow_distributed` with mean `0.65 m/s`, standard deviation `0.2 m/s`; and `typical_distributed` with mean `1.3 m/s`, standard deviation `0.2 m/s`. Explicit regimes derive the desired-speed sampling seed from the episode seed.
- Spawn speed remains the released `0.5 m/s`; only desired speed changes. The protocol records configured and realized distributions, initial spawn speed, time-to-target, acceleration transient, and activation fraction.

The primary metrics are success, collision, and near-miss rates. Exposure and clearance metrics, typed collision rates, paired scenario-seed inference, one-sided harm margins, Holm–Bonferroni multiplicity, and the 2,000-replicate paired-seed-block bootstrap are frozen in the YAML. Ranking is descriptive only; collision frequency is not a physical-impact or real-world safety claim.

For non-reference regimes, activation requires at least 80% of pedestrians to reach the configured desired-speed mean within `0.20 m/s`, with a p95 time-to-target no greater than `2.0 s`. The declared transient window is recorded and excluded only as specified; failure is `intervention_not_activated`, which blocks a no-harm conclusion.

The required turnaround ledger is also frozen from decision through dissertation evidence-admission decision. Private scheduler topology, credentials, scratch paths, and job IDs do not belong in this tracked protocol note.

## Validation

```bash
uv run pytest -q tests/validation/test_check_issue_6561_pedestrian_speed_protocol.py
uv run python scripts/validation/check_issue_6561_pedestrian_speed_protocol.py --manifest
```

These checks prove protocol identity and manifest construction only. They do not prove runtime activation, native campaign execution, result integrity, or dissertation admission.
