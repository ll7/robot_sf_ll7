# Data Model: CI System Package Installation Optimization

## Overview
This feature optimizes the CI system package installation performance. The data model captures CI job execution, package installation metrics, and performance tracking.

## Entities

### CIJob
Represents a single CI pipeline execution with timing and status tracking.

**Fields**:
- `id`: string (UUID) - Unique identifier for the CI run
- `start_time`: datetime - When the CI job started
- `end_time`: datetime - When the CI job completed (nullable for running jobs)
- `status`: enum [pending, running, completed, failed] - Current job status
- `package_installation_time`: float (seconds) - Time taken for system package installation step
- `total_duration`: float (seconds) - Total CI job duration
- `github_run_id`: string - GitHub Actions run identifier
- `branch`: string - Git branch being tested

**Relationships**:
- `packages`: one-to-many with SystemPackage
- `metrics`: one-to-many with PerformanceMetric

**Validation Rules**:
- `package_installation_time` must be >= 0
- `total_duration` must be >= 0
- `status` transitions: pending → running → completed/failed
- `package_installation_time` should be < 73 seconds (50% reduction target)

### SystemPackage
Represents a required Ubuntu system package for headless testing.

**Fields**:
- `id`: string (UUID) - Unique identifier
- `name`: string - Package name (e.g., "ffmpeg", "libglib2.0-0")
- `version`: string - Installed package version (nullable)
- `installation_status`: enum [pending, installed, failed] - Installation result
- `download_size`: integer (bytes) - Approximate download size
- `ci_job_id`: string (UUID) - Reference to parent CI job

**Relationships**:
- `ci_job`: many-to-one with CIJob

**Validation Rules**:
- `name` must be one of: ffmpeg, libglib2.0-0, libgl1, fonts-dejavu-core
- `installation_status` must be "installed" for successful CI runs
- `download_size` must be >= 0

### PerformanceMetric
Tracks performance measurements and thresholds for CI optimization.

**Fields**:
- `id`: string (UUID) - Unique identifier
- `name`: string - Metric name (e.g., "package_installation_time", "cache_hit_rate")
- `value`: float - Measured value
- `unit`: string - Unit of measurement (e.g., "seconds", "percentage")
- `threshold`: float - Target threshold value
- `ci_job_id`: string (UUID) - Reference to parent CI job

**Relationships**:
- `ci_job`: many-to-one with CIJob

**Validation Rules**:
- `value` must be >= 0
- `threshold` must be >= 0
- For "package_installation_time": threshold = 73.0, unit = "seconds"

## State Transitions

### CIJob State Machine
```
pending → running (on job start)
running → completed (on successful finish)
running → failed (on error)
```

### SystemPackage State Machine
```
pending → installed (on successful installation)
pending → failed (on installation error)
```

## Data Flow
1. CI job starts → CIJob created with status "pending"
2. Package installation begins → CIJob status → "running"
3. Each package installation tracked → SystemPackage records created
4. Installation completes → PerformanceMetric recorded
5. CI job ends → CIJob status updated, total_duration calculated

## Constraints
- All packages must achieve "installed" status for CI success
- Package installation time must meet performance threshold
- Data retention: CI job records kept for 30 days for performance analysis