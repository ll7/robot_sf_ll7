# Issue #923: Multi-Ped Adversarial Candidate Schema

## Goal

Issue #923 is the first bounded implementation slice under the larger #870 multi-pedestrian
adversarial environment epic. It adds a versioned schema for scripted multi-pedestrian adversarial
candidates before any reset/step, learned-adversary, or benchmark-runner integration exists.

## Scope Boundary

This schema is a development contract only:

- it can parse YAML into typed candidate entries,
- it validates pedestrian IDs, poses, timing, speed, seed, and minimum route length,
- it serializes to deterministic JSON-compatible dictionaries,
- it does not create or certify benchmark scenarios,
- it does not alter existing single-pedestrian adversarial search behavior.

Runtime generation, plausibility checks against concrete maps, episode attribution, and replay
integration remain future #870 work.

## Current Surfaces

- Schema code: `robot_sf/adversarial/config.py`
- Example config: `configs/adversarial/group_squeeze_multi_ped_example.yaml`
- Tests: `tests/adversarial/test_adversarial_search.py`

## Validation Path

Use targeted checks for this slice:

```bash
uv run pytest tests/adversarial/test_adversarial_search.py -q
uv run ruff check robot_sf/adversarial/config.py tests/adversarial/test_adversarial_search.py
```

Before PR handoff, run the repository readiness gate:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Caveats

Do not report this schema as multi-pedestrian adversarial benchmark support. The fail-closed
benchmark policy still applies: until runtime execution and scenario certification are proven, these
payloads are inputs for future development stress tooling only.
