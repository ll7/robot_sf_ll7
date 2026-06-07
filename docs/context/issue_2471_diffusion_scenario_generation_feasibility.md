# Issue #2471 Diffusion Scenario Generation Feasibility Scope (2026-06-06)

Status: scoped proposal and interface direction, not diffusion training or generated-realism evidence.

Related surfaces:
- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2471
- Parent roadmap issue: https://github.com/ll7/robot_sf_ll7/issues/2469
- Existing programmatic scenario generator: `robot_sf/benchmark/scenario_generator.py`
- GeneratedScenario dataclass: `robot_sf/benchmark/scenario_generator.py:57` (`GeneratedScenario`)
- Benchmark scenario JSON Schema: `robot_sf/benchmark/schema/scenarios.schema.json`
- Scenario certification: `robot_sf/scenario_certification/v1.py`
- Adversarial generation protocol: `docs/context/issue_1457_adversarial_generation_protocol.md`
- Adversarial failure archive: `docs/context/issue_1237_adversarial_failure_archive.md`
- Adversarial manifest smoke: `docs/context/issue_2562_adversarial_manifest_smoke.md`
- Adversarial manifest quality metrics: `docs/context/issue_2567_adversarial_manifest_quality.md`
- Learned-expansion gate: `docs/context/issue_2568_adversarial_expansion_gate.md`
- Failure mechanism classifier: `docs/context/issue_2012_failure_mechanism_classifier.md`
- Scenario contract docs: `docs/scenario_contracts.md`
- Scenario config archetypes: `configs/scenarios/archetypes/`
- SVG map pipeline: `robot_sf/nav/svg_map_parser.py`
- Pedestrian population logic: `robot_sf/ped_npc/ped_population.py`
- Trace export schema: `robot_sf/analysis_workbench/schemas/simulation_trace_export.v1.json`

## Result
Issue #2471 asks whether a diffusion-based scenario generation pipeline is feasible for Robot SF,
and what interface, data, and compute prerequisites would be needed before any training or
realism claim.

The current repository has three scenario-creation paths: (1) authored SVG maps with YAML manifests,
(2) a simple programmatic generator (`scenario_generator.py` with 4 flow patterns, 3 obstacle
layouts, 3 density levels in a 10m x 6m arena), and (3) an adversarial route/search framework that
perturbs existing map configurations. None of these paths use learned generation. A diffusion
scenario generator would be an entirely new capability with distinct data, compute, and validation
requirements.

This pass defines the failure-archive data format, the generated scenario manifest and schema,
validation gate, compute and data prerequisites, stop rule, first non-training spike, and
downstream propagation boundary. It does not train a diffusion model, generate realistic scenarios,
or change planner rankings.

## Failure-Archive Inputs
A diffusion scenario generator needs training data from archived planner failures. The failure
archive input format should be a deterministic, repo-addressable trace bundle with:

- **Source scenario**: `scenario_id`, `map_file`, `seed`, `planner_id`, `config_path`
- **State trajectory**: time-indexed sequence of robot pose, pedestrian states (position,
  velocity, goal, groups), and obstacle layout — either as a compacted `simulation_trace_export.v1`
  JSONL slice or a structured NPZ array
- **Failure label**: one or more mechanism-classified labels from the
  `issue_2012_failure_mechanism_classifier.md` vocabulary (collision, timeout, stuck, off-route,
  navigation_error, simulation_error)
- **Criticality context**: near-miss distances, clearance events, forced-braking intervals, and
  any space-occupancy conflicts
- **Metadata provenance**: generator version, scenario-contract hash, planner hash,
  observation-export schema version, commit SHA

A candidate reference path for archive curation is the existing adversarial failure archive pattern
(`docs/context/issue_1237_adversarial_failure_archive.md`) extended with per-timestep trace
snapshots and mechanism labels. A minimum viable archive should contain at least 50 distinct
failure traces across 3+ map geometries and 2+ planners before diffusion training is viable.

## Generated Scenario Manifest Schema And Validation Gate
A diffusion-generated scenario must produce a YAML manifest that validates against an extension
of the existing `scenarios.schema.json`. The proposed manifest schema adds a `diffusion_generator`
block:

```yaml
# docs/context/evidence/issue_2471/scenario_generation_manifest_schema.yaml
```

Required `diffusion_generator` fields:
- `generator_id`: unique model/checkpoint identifier
- `generator_commit`: training or export commit SHA
- `training_archive_hash`: SHA-256 of the failure archive used for training
- `training_seed`: deterministic seed for training
- `inference_seed`: deterministic seed for inference/sampling
- `sampling_params`: dict of diffusion sampling parameters (steps, guidance scale, etc.)
- `generated_scenario_id`: unique ID in namespace `diffusion_<hash>`
- `certification`: scenario certification classification (`certify_scenario_file` output)

The validation gate must pass all of:
1. **Schema validation**: manifest conforms to the extended scenarios.schema.json
2. **Scenario certification**: `certify_scenario_file` returns `valid` or `hard_but_solvable`
   (not `invalid`, `geometrically_infeasible`, or `kinodynamically_infeasible`)
3. **Determinism check**: re-running the same generator_id + inference_seed produces bit-identical
   manifest (within floating-point tolerance)
4. **Fail-closed archive lineage**: the `training_archive_hash` must match a registered archive
   in `model/registry.yaml` or a tracked evidence manifest
5. **Planner smoke**: at least one baseline planner (e.g., `orca`, `social_force`) runs to
   completion without `simulation_error`

Scenarios that fail any validation gate are rejected fail-closed and recorded as
`diffusion_generation_failed` — they must not enter benchmark scenario sets.

Before diffusion training or broad generated-scenario expansion starts, the proposed generated
manifest batches must also pass the Issue #2568 workflow gate: run an Issue #2562-style manifest
smoke, summarize Issue #2567 quality metrics, and reject or label as diagnostic any invalid,
duplicate, degenerate, fallback/degraded, or low-yield batch. This gate does not certify realism or
benchmark strength; it only prevents learned generation from proceeding on unproven manifest
behavior.

## Compute And Data Prerequisites
Before any diffusion training is attempted, the worktree must satisfy:

### Data
- [ ] Minimum 50 failure traces across 3+ map geometries and 2+ planners
- [ ] Traces exported in a format loadable by a PyTorch/TF Dataset (NPZ stack or
      `simulation_trace_export.v1` JSONL)
- [ ] Failure labels verified against the issue_2012 classifier vocabulary
- [ ] Normalized state representation (position deltas, obstacle occupancy grid, goal vectors)
- [ ] Train/validation split with explicit seeds and held-out map geometries

### Compute (estimate for a proof-of-concept simple diffusion)
- [ ] GPU with >= 8 GB VRAM (e.g., RTX 3070/4060 or better)
- [ ] ~50 GB scratch disk for trace caching, augmentation, and checkpoint intermediates
- [ ] Training runtime estimate: 2-8 hours for a small UNet/temporal-diffusion on 50-200 traces
      (single GPU, batch size 8-32)
- [ ] Inference runtime target: < 30 seconds per scenario on the same GPU
- [ ] CPU-only fallback: < 5 minutes per scenario (small model, T=50 diffusion steps)

These are feasibility estimates for a v0 prototype. Production scaling is out of scope until the
first spike passes.

## Stop Rule
Work on this issue is complete when all of the following hold:

1. This feasibility note is reviewed and merged.
2. The failure-archive manifest schema is defined (this note or a follow-up PR).
3. The generated-scenario manifest schema extension is defined (this note or a follow-up PR).
4. The validation gate steps are documented (this note or a follow-up PR).

Work **stops** at the feasibility/interface boundary. No diffusion model training, no scenario
realism claim, no benchmark improvement claim, no planner ranking — those are explicitly deferred
to follow-up issues.

## First Non-Training Spike
The recommended first executable spike (that does not require training) is:

1. **Define the failure-archive YAML manifest format** (a structured YAML with trace references).
2. **Write a schema validator** that checks the manifest against the extended schema.
3. **Curate 5 existing failure traces** from the adversarial search or benchmark debug directories
   into the archive format, stored as small YAML/JSON fixtures under
   `docs/context/evidence/issue_2471/`.
4. **Write a deterministic smoke test** that loads the fixture manifest and validates against the
   schema — this is the interface contract test.
5. **Prove fail-closed behavior**: a malformed manifest (missing field, bad hash) is rejected with
   a clear error.

This spike proves the data pipeline and validation gate work without any GPU, training, or
diffusion inference. It is the minimum viable interface proof.

## Downstream Propagation And Synthesis Target
When a working diffusion prototype eventually exists, the following surfaces must be updated:

- **`model/registry.yaml`**: add a diffusion generator entry with checkpoint path, training
  archive hash, and inference parameters
- **`configs/scenarios/`**: generated scenarios may be added to a `diffusion_` namespace,
  but only after per-scenario certification and a mechanism-aware bias audit
- **Benchmark scenario sets**: generated scenarios must be kept separate from authored maps
  in benchmark aggregate rows; label them as `diffusion_candidate` until mechanism-transfer
  evidence is reviewed
- **Failure mechanism synthesis** (`docs/context/issue_2220_failure_mechanism_taxonomy.md`):
  the synthesis target should classify whether generated scenarios reproduce, amplify, or
  cover new failure mechanisms versus the training archive
- **Paper-facing docs**: generated scenarios are `synthetic` and must be labeled as such;
  any benchmark claim using generated scenarios must explicitly name the generation source,

## Claim Boundary
This is proposal and interface evidence only.

It does **not**:
- Prove that a diffusion model can generate realistic or useful scenarios in Robot SF
- Prove that generated scenarios change planner rankings or benchmark outcomes
- Provide a trained diffusion model, checkpoint, or inference code
- Replace the existing programmatic generator, SVG map pipeline, or adversarial search
- Claim any benchmark improvement, planner robustness, or scenario coverage metric

A scenario manifest schema records a proposed interface; scenario certification, trace evidence,
and data-grounded validation remain separate gates that must be completed before any diffusion
claim is made.

The Issue #2568 gate is the current cross-method workflow gate for broad learned adversarial
expansion. Passing it is prerequisite hygiene, not evidence that diffusion produces realistic or
benchmark-useful scenarios.

## Validation
```bash
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```

If evidence fixtures are added:
```bash
uv run python -c "import yaml; yaml.safe_load(open('docs/context/evidence/issue_2471/scenario_generation_manifest_schema.yaml'))"
```

## Follow-Up Boundary
The recommended next issue is an executable failure-archive interface spike: define the YAML
manifest format, write the schema validator, curate 5 existing failure traces as fixtures, and
prove fail-closed validation. Stop there before any diffusion training, scenario-realism claim,
or planner comparison.

The issue after that (if the spike passes) is a data-curation issue: expand from 5 to 50+
failure traces across 3+ maps and 2+ planners, with mechanism labels verified, normalized
state representations, and a train/validation split. Only then should a diffusion-training
issue be opened.
