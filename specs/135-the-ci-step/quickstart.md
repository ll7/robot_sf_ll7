# Quickstart: Testing CI System Package Installation Optimization

## Overview
This guide provides steps to validate that the CI system package installation optimization meets performance and reliability requirements.

## Prerequisites
- GitHub repository with CI workflow
- Access to trigger CI runs
- Monitoring access to CI performance metrics

## Test Scenarios

### Scenario 1: Performance Validation
**Given** the optimized CI workflow is deployed
**When** a CI run executes the system package installation step
**Then** the step completes in under 73 seconds (50% reduction target)

**Steps**:
1. Push a commit to trigger CI
2. Monitor the "System packages for headless" step duration
3. Verify duration < 73 seconds
4. Record measurement for trend analysis

### Scenario 2: Reliability Validation
**Given** the optimized CI workflow is deployed
**When** 10 consecutive CI runs execute
**Then** all package installations succeed (zero failure rate)

**Steps**:
1. Trigger 10 CI runs (can use workflow_dispatch)
2. Monitor all runs complete successfully
3. Verify no installation failures
4. Calculate success rate = 100%

### Scenario 3: Functionality Preservation
**Given** packages are installed via optimized method
**When** subsequent test steps run
**Then** headless GUI testing works correctly

**Steps**:
1. CI run completes package installation
2. Monitor pygame and matplotlib-dependent tests
3. Verify all GUI tests pass in headless mode
4. Confirm no import or runtime errors

### Scenario 4: Cache Effectiveness (if caching implemented)
**Given** package caching is enabled
**When** consecutive CI runs execute on same runner
**Then** cache hits provide faster installation

**Steps**:
1. Run CI twice in succession
2. Compare first run (cache miss) vs second run (cache hit)
3. Verify cache hit is significantly faster
4. Monitor cache hit rate over time

## Validation Commands

### Local Testing (limited)
```bash
# Test package availability (requires sudo)
sudo apt-get update && sudo apt-get install -y ffmpeg libglib2.0-0 libgl1 fonts-dejavu-core

# Verify packages work
ffmpeg -version | head -1
pkg-config --modversion glib-2.0

# Test headless GUI
SDL_VIDEODRIVER=dummy MPLBACKEND=Agg python -c "
import pygame
pygame.init()
print('pygame OK')
pygame.quit()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure()
plt.close()
print('matplotlib OK')
"
```

### CI Performance Monitoring
```bash
# Check recent CI run times (requires GitHub CLI)
gh run list --limit 5 --json headSha,databaseId,status,createdAt,updatedAt

# Get detailed timing for specific run
gh run view <run-id> --json jobs
```

## Success Criteria
- [ ] Average package installation time < 73 seconds across 5 runs
- [ ] 100% success rate across 10 runs
- [ ] All headless GUI tests pass
- [ ] No performance regressions introduced
- [ ] Cache hit rate > 80% (if caching implemented)

## Troubleshooting

### Performance Issues
- Check network connectivity during CI runs
- Verify caching is working (check cache keys)
- Monitor Ubuntu runner load

### Functionality Issues
- Confirm all required packages are still installed
- Check for version conflicts
- Validate headless environment variables are set

### Reliability Issues
- Review CI logs for error patterns
- Check for Ubuntu version compatibility
- Verify sudo permissions in CI

## Expected Results
- Package installation: ~45-60 seconds (with optimizations)
- Cache hit scenarios: ~10-20 seconds
- Success rate: 100%
- No breaking changes to existing functionality