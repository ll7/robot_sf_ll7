# SNQI Weights Artifact Provenance

**Purpose**: Document the provenance, validation, and lifecycle management of SNQI weight artifacts.

## Overview

The Social Navigation Quality Index (SNQI) uses weighted composite scoring to evaluate robot navigation performance. This document tracks the provenance of weight configuration artifacts and their validation.

## Current Artifacts

### v1 Canonical Weights (`model/snqi_canonical_weights_v1.json`)

**Source**: Research team optimization study
**Date**: 2025-01-02  
**Method**: Multi-objective optimization with Pareto analysis
**Validation**: Tested against 500 baseline episodes across 5 scenarios
**Status**: Production ready

```json
{
  "version": "v1",
  "metadata": {
    "creation_date": "2025-01-02",
    "method": "pareto_optimization",
    "validation_episodes": 500,
    "notes": "Optimized for balanced safety-efficiency trade-offs"
  },
  "weights": {
    "safety": 0.4,
    "efficiency": 0.3,
    "comfort": 0.3
  },
  "sub_weights": {
    "safety": {
      "collision_rate": 0.6,
      "near_miss_rate": 0.4
    },
    "efficiency": {
      "path_efficiency": 0.7,
      "time_efficiency": 0.3
    },
    "comfort": {
      "jerk": 0.4,
      "acceleration": 0.3,
      "force_discomfort": 0.3
    }
  }
}
```

## Weight Selection Methodology

### 1. Multi-Objective Optimization Process
- **Input**: Baseline performance across safety, efficiency, comfort metrics
- **Process**: Pareto frontier analysis with stakeholder preference elicitation
- **Output**: Weight vectors that maximize discriminative power while respecting domain constraints

### 2. Validation Criteria
- **Coverage**: Weights tested on ≥500 episodes across diverse scenarios
- **Stability**: SNQI rankings consistent across random seeds
- **Sensitivity**: Weight perturbations don't cause rank inversions for clearly different algorithms
- **Domain Alignment**: Results align with expert judgments in navigation quality

### 3. Approval Process
- Research team review and validation
- Performance testing on holdout scenarios
- Documentation of assumptions and limitations
- Versioned release with metadata

## Using Weight Artifacts

### Programmatic Access
```python
from robot_sf.benchmark.snqi.types import SNQIWeights

# Load canonical weights
weights = SNQIWeights.from_file("model/snqi_canonical_weights_v1.json")

# Access individual weights
safety_weight = weights.get_weight("safety")
collision_weight = weights.get_sub_weight("safety", "collision_rate")
```

### CLI Usage
```bash
# Use canonical weights for aggregation
robot_sf_bench aggregate --in episodes.jsonl --out summary.json \
  --snqi-weights model/snqi_canonical_weights_v1.json

# Recompute SNQI with custom weights during aggregation
robot_sf_bench aggregate --in episodes.jsonl --out summary.json \
  --snqi-weights custom_weights.json
```

## Artifact Lifecycle

### Version Control
- All weight artifacts stored in `model/` directory
- JSON format with embedded metadata
- Semantic versioning: `snqi_canonical_weights_v{major}.json`
- Git tracking for full provenance

### Updates and Deprecation
1. **New Research**: Create new version file (`v2`, `v3`, etc.)
2. **Validation**: Test new weights against established baselines
3. **Migration**: Update default references in code/configs
4. **Deprecation**: Mark old versions as deprecated in metadata
5. **Removal**: Archive deprecated weights after 2+ major versions

### Quality Assurance
- **Schema Validation**: All weight files validated against JSON schema
- **Range Checks**: Weights in [0,1] range, sub-weights sum to 1.0 per category
- **Performance Testing**: Benchmark impact on computation time and memory
- **Documentation**: Changes tracked in `CHANGELOG.md`

## Troubleshooting

### Common Issues
1. **Missing Weight File**: Check `model/` directory and file permissions
2. **Schema Validation Failed**: Verify JSON structure matches expected format
3. **Inconsistent Results**: Ensure same weight version used across analysis pipeline
4. **Performance Degradation**: Check for weight file corruption or network latency

### Validation Commands
```bash
# Validate weight file format
python -c "from robot_sf.benchmark.snqi.types import SNQIWeights; SNQIWeights.from_file('model/snqi_canonical_weights_v1.json'); print('Valid')"

# Test SNQI computation
robot_sf_bench aggregate --in test_episodes.jsonl --out /dev/null \
  --snqi-weights model/snqi_canonical_weights_v1.json
```

## Contact and Support

For questions about weight artifact provenance or to propose new weight configurations:
- Technical Issues: Check `docs/snqi-weight-tools/README.md`  
- Research Questions: Reference original optimization study documentation
- Process Questions: Review this provenance document

---
**Last Updated**: 2025-01-19  
**Next Review**: 2025-04-01