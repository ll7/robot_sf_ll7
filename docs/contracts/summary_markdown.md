# Markdown Summary Contract — Verify Feature Extractor Training Flow

| Section | Required Content |
|---------|------------------|
| Header | Run ID, creation timestamp, output directory, hardware overview table |
| Extractor Table | One row per extractor with columns: Extractor Name, Status, Worker Mode, Duration (s), Best Metric (configurable key), Notes |
| Failures & Skips | Bullet list explaining any non-success statuses |
| Aggregated Metrics | Table or bullet list mirroring keys from `aggregate_metrics` in `summary.json` |
| Reproducibility Footer | Paths to JSON summary and per-extractor artifact directories |

**Formatting Rules**
- Status values must match the enum in `training_summary.schema.json` (`success`, `failed`, `skipped`).
- Durations rounded to one decimal place.
- Metrics displayed with explicit units where applicable (e.g., `reward`, `collision_rate`).
- Hardware overview table must include platform, architecture, GPU model (if present), CUDA version (if present), and worker count.
