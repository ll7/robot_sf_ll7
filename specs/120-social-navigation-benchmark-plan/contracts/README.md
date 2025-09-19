# Contracts Directory

This folder will contain JSON Schema definitions and (later) OpenAPI-like contracts for benchmark artifacts.

Planned schema files (Phase 1 output):
- episode.schema.v1.json
- aggregate.schema.v1.json
- scenario-matrix.schema.v1.json
- snqi-weights.schema.v1.json
- resume-manifest.schema.v1.json

Each schema must:
1. Declare `$schema` (draft 2020-12) and `title`.
2. Provide `type` at root and `required` arrays.
3. For objects, restrict additionalProperties where appropriate.
4. Include `version` constant field where applicable.

Validation tests will assert compliance by loading sample artifacts.
