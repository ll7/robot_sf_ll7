# Research: CI System Package Installation Optimization

## Current State Analysis

**Current CI Step**:
```yaml
- name: System packages for headless
  run: sudo apt-get update && sudo apt-get install -y ffmpeg libglib2.0-0 libgl1 fonts-dejavu-core
```

**Performance Issue**: Takes 2min 26sec, which is ~50% of total CI time (5min).

**Required Packages**: ffmpeg, libglib2.0-0, libgl1, fonts-dejavu-core (for headless GUI testing with pygame/matplotlib).

## Optimization Approaches Evaluated

### Approach 1: Faster Package Manager (apt-fast)
**Description**: apt-fast uses aria2 for parallel downloads and multiple connections.

**Implementation**:
```bash
# Install apt-fast
sudo apt-get install -y aria2
wget https://raw.githubusercontent.com/ilikenwf/apt-fast/master/apt-fast
sudo mv apt-fast /usr/local/bin/
sudo chmod +x /usr/local/bin/apt-fast

# Use apt-fast instead of apt-get
sudo apt-fast update && sudo apt-fast install -y ffmpeg libglib2.0-0 libgl1 fonts-dejavu-core
```

**Pros**: 
- Can reduce download time by 2-3x through parallel connections
- Drop-in replacement for apt-get
- No changes to workflow structure

**Cons**:
- Adds installation time for apt-fast itself (~10-20sec)
- May not work reliably in all network conditions
- Additional dependency management

**Expected Performance**: Could reduce to 1min-1min 30sec

### Approach 2: Pre-built Docker Image
**Description**: Use a custom Docker container with packages pre-installed.

**Implementation**:
```yaml
jobs:
  ci:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/your-org/robot-sf-ci:latest
      # Pre-built with packages
```

**Pros**:
- Eliminates package installation entirely
- Consistent environment
- Potentially faster overall CI

**Cons**:
- Requires maintaining custom Docker image
- More complex setup (build and push image)
- May affect caching of .venv and uv cache
- Need to ensure Python/uv compatibility

**Expected Performance**: Near-instant (seconds instead of minutes)

### Approach 3: System Package Caching
**Description**: Cache downloaded packages between runs using actions/cache.

**Implementation**:
```yaml
- name: Cache apt packages
  uses: actions/cache@v4
  with:
    path: /var/cache/apt/archives
    key: apt-${{ runner.os }}-${{ hashFiles('.github/workflows/ci.yml') }}

- name: System packages for headless
  run: sudo apt-get update && sudo apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 fonts-dejavu-core
```

**Pros**:
- Significant speedup on cache hits
- Simple to implement
- Works with existing workflow

**Cons**:
- Cache misses still slow
- Cache size limits (GitHub has 10GB limit)
- May not work perfectly with Ubuntu version changes

**Expected Performance**: ~30sec on cache hit, 2min 26sec on miss

### Approach 4: Combined apt-get update/install
**Description**: Combine update and install in one command to avoid double metadata download.

**Implementation**:
```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 fonts-dejavu-core
```

**Pros**: Simple change
**Cons**: Minimal impact (saves ~10-20sec at most)
**Expected Performance**: 2min 16sec - 2min 6sec

### Approach 5: Hybrid - apt-fast + caching
**Description**: Combine apt-fast with caching for best of both worlds.

**Implementation**: Install apt-fast + cache /var/cache/apt/archives

**Pros**: Fast downloads + cache persistence
**Cons**: More complex setup
**Expected Performance**: ~45sec on cache hit with apt-fast

## Decision: Recommended Approach

**Primary Recommendation**: Approach 3 (System Package Caching) + Approach 1 (apt-fast)

**Rationale**:
- Caching provides immediate benefit and works reliably
- apt-fast adds parallel download speedup
- Both are low-risk changes to existing workflow
- Combined should achieve 50%+ reduction target
- No major workflow restructuring required

**Alternative Considered**: Pre-built Docker image
- Rejected because: Adds complexity of image maintenance, potential conflicts with uv/Python setup, overkill for this specific optimization

**Implementation Plan**:
1. Add apt package caching
2. Install and use apt-fast
3. Test performance improvement
4. Fallback to standard apt-get if issues arise

## Success Metrics
- Target: < 1min 13sec (50% reduction from 2min 26sec)
- Success Criteria: Consistent achievement across multiple CI runs
- Failure Criteria: > 2min or reliability issues introduced