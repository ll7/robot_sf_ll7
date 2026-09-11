# Lineage Index

- Schema: `robot_sf.lineage_index.v1` | Status: `ok`
- Rows: 1 | Records: 14 | Findings: 0

Sanitized cross-record lineage projection only: rows join declared semantic identities from compact public records and an optional sanitized private projection. A missing link is a custody observation, not a scientific, benchmark, or evidence verdict.

## Findings

- none

## Rows

| Lineage key | Attempt | Relation | Predecessor | Missing | Joined records |
| --- | --- | --- | --- | --- | --- |
| `job:job-1` | 1 | `initial` | `-` | - | analysis_ids=(analysis-1); artifact_ids=(art-compact-1,art-raw-1); campaign_ids=(camp-newline); checkpoint_ids=(ckpt-1); claim_ids=(claim-1); commit_ids=(c0ffee1); config_ids=(cfg-newline); environment_ids=(env-1); issue_ids=(8897); job_ids=(job-1); manifest_ids=(manifest-a); model_ids=(model-1); pull_request_ids=(9012) |
