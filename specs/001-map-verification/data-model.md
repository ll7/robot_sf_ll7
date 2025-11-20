# Data Model — Map Verification Workflow

## Entities

### MapRecord
- **Fields**: `map_id` (str), `file_path` (Path), `tags` (set[str]), `ci_enabled` (bool), `metadata` (dict with spawn zones, goals, pedestrian flags), `last_modified` (datetime)
- **Relationships**: References zero or more `VerificationResult` entries.
- **Validation Rules**: `map_id` unique; metadata must include spawn + goal definitions unless marked `pedestrian_only=True` and referencing alt schema.

### VerificationResult
- **Fields**: `map_id`, `status` (pass|fail|warn), `rule_ids` (list[str]), `duration_ms` (float), `factory_used` (robot|pedestrian), `message` (str), `timestamp` (datetime)
- **Relationships**: Belongs to exactly one `VerificationRunSummary`.
- **Validation Rules**: `status` must align with rule severities; `rule_ids` non-empty when status != pass; `duration_ms` > 0.

### VerificationRunSummary
- **Fields**: `run_id` (uuid/str), `git_sha`, `total_maps`, `passed`, `failed`, `warned`, `slow_maps` (list[map_id]), `artifact_path`, `started_at`, `finished_at`
- **Relationships**: Aggregates many `VerificationResult` entries.
- **Validation Rules**: `passed + failed + warned == total_maps`; `slow_maps` subset of analyzed maps; timestamps monotonic.
