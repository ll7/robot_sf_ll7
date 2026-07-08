# Research: Fix Benchmark Placeholder Outputs

**Date**: 2025-09-24
**Feature**: 133-all-generated-plots
**Status**: Complete

## Research Questions & Findings

### Q1: Why are plots currently placeholders?
**Decision**: Plots use hardcoded placeholder images instead of computing from actual benchmark data
**Rationale**: Original implementation was incomplete - focused on benchmark execution but not visualization
**Alternatives considered**: 
- Generate plots from synthetic data (rejected: defeats purpose of real benchmarking)
- Skip plots entirely (rejected: violates visualization requirements)
**Implementation**: Need to extract metrics from JSONL episode data and generate matplotlib plots

### Q2: Why are videos dummy placeholders?
**Decision**: Videos are static placeholder MP4 files instead of rendered simulation replays
**Rationale**: Video rendering infrastructure exists but wasn't integrated into benchmark pipeline
**Alternatives considered**:
- Use screen recording during simulation (rejected: unreliable, platform-dependent)
- Generate animated GIFs (rejected: MP4 requirement, poorer quality)
**Implementation**: Use environment_factory + sim_view methods to replay episodes and render videos

### Q3: What metrics are available for plotting?
**Decision**: Use existing benchmark metrics (collisions, success rates, SNQI scores, trajectory data)
**Rationale**: Metrics are already computed and stored in episode JSONL files
**Alternatives considered**: Add new metrics (rejected: scope creep, existing metrics sufficient)
**Implementation**: Parse metrics from episode records and create distribution plots

### Q4: How to integrate video rendering into benchmark pipeline?
**Decision**: Add video rendering step after benchmark execution, before completion
**Rationale**: Videos need episode data, so must run after benchmark but as part of same process
**Alternatives considered**:
- Render videos during benchmark execution (rejected: performance impact)
- Separate video generation script (rejected: user expectation of integrated workflow)
**Implementation**: Extend benchmark orchestrator to call video rendering functions

### Q5: What plot types should be generated?
**Decision**: Generate standard benchmark visualization plots (metric distributions, trajectories, performance comparisons)
**Rationale**: Follows existing documentation and research paper standards
**Alternatives considered**: Custom plots per user request (rejected: standardization needed)
**Implementation**: Create reusable plotting functions for common benchmark visualizations

## Technical Dependencies Analysis

### Matplotlib Integration
- **Status**: Already available in environment
- **Usage**: Core plotting library for PDF generation
- **Best practices**: Use vector outputs (PDF), consistent styling, proper axis labels

### MoviePy Integration  
- **Status**: Listed in pyproject.toml dependencies
- **Usage**: Video rendering and composition
- **Best practices**: Frame-by-frame rendering, proper encoding, error handling for missing frames

### Episode Data Processing
- **Status**: JSONL parsing already implemented
- **Usage**: Extract metrics and trajectories for visualization
- **Best practices**: Validate data completeness, handle missing fields gracefully

## Integration Patterns

### Benchmark Pipeline Extension
**Pattern**: Extend existing benchmark orchestrator to include post-processing steps
**Benefits**: Maintains single-command execution, ensures data consistency
**Implementation**: Add visualization phase after episode execution completes

### Error Handling Strategy
**Pattern**: Fail gracefully with clear error messages for missing dependencies or data
**Benefits**: Better user experience, easier debugging
**Implementation**: Check dependencies at startup, validate data before plotting

## Risk Assessment

### Low Risk Items
- Matplotlib plotting (well-established, already used elsewhere)
- JSONL data parsing (already implemented for aggregation)
- PDF output generation (standard matplotlib functionality)

### Medium Risk Items  
- MoviePy video rendering (new integration, but library is mature)
- Performance impact of video rendering (may slow down benchmark completion)

### Mitigation Strategies
- Add dependency checks and clear error messages
- Make video rendering optional with fallback to placeholder
- Profile performance impact and optimize if needed

## Conclusion

All research questions resolved. Implementation approach is clear:
1. Extract real metrics from episode data for plotting
2. Use environment replay methods for video rendering  
3. Integrate both into benchmark pipeline
4. Add proper error handling and validation

No blocking unknowns remain. Ready to proceed to Phase 1 design.