# Performance Notes - Social Navigation Benchmark

**Purpose**: Document performance baselines, benchmarks, and optimization targets for the Social Navigation Benchmark platform.

## Performance Baselines

### Environment Performance (as of 2025-01-19)

**Environment Creation**:
- Target: < 2.0 seconds
- Measured: ~1.16 seconds
- Status: ✅ PASS

**Environment Reset**:
- Target: > 1 reset/second
- Measured: ~1,745 resets/second (0.6ms/reset)
- Status: ✅ PASS

### Historical Performance Targets

From `dev_guide.md` expected ranges:
- **Environment creation**: < 1 second
- **Model loading**: 1–5 seconds  
- **Simulation performance**: ~22 steps/second (~45ms/step)
- **Build time**: 2–3 minutes (first time)
- **Test suite**: 2–3 minutes (≈170 tests)

## Performance Validation

### Smoke Test Suite
Location: `scripts/validation/performance_smoke_test.py`

**Validation Criteria**:
- Environment creation < 3.0s (allows headroom)
- Reset performance > 0.5 resets/sec (minimum acceptable)
- Consistent performance across multiple runs

**Usage**:
```bash
# Run performance smoke test
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/performance_smoke_test.py

# Results saved to: results/performance_smoke_test.json
```

### Benchmark Runner Performance
Location: `robot_sf/benchmark/runner.py`

**Parallel Processing**:
- Multi-worker support for episode generation
- Process-based parallelism with resume capabilities
- Manifest-driven episode deduplication

**Expected Throughput**:
- Single worker: ~1-2 episodes/second (varies by scenario complexity)
- 4 workers: ~4-6 episodes/second (near-linear scaling)
- Resume overhead: minimal (manifest lookup < 1ms per episode)

## Performance Monitoring

### Key Metrics to Track
1. **Environment Initialization**: Creation + first reset time
2. **Episode Generation**: Episodes per second in benchmark runs  
3. **Memory Usage**: Peak RSS during multi-worker runs
4. **I/O Performance**: JSONL write throughput for large episode batches

### Regression Detection
- Performance degradation > 50% should trigger investigation
- Compare against baseline measurements from this document
- Use `results/performance_smoke_test.json` for automated monitoring

## Optimization Notes

### Known Performance Bottlenecks
1. **pygame/SDL initialization**: ~1s for headless setup
2. **FastPysf compilation**: JIT overhead on first use  
3. **Large episode JSON**: Serialization cost grows with trajectory length
4. **File I/O**: JSONL append becomes slow with very large files

### Optimization Strategies
1. **Environment Reuse**: Keep environments alive across episodes when possible
2. **Batch Processing**: Group similar scenarios to reduce setup overhead
3. **Memory Management**: Monitor RSS growth in long-running processes
4. **Disk I/O**: Use SSD storage for results directories

## Hardware Dependencies

### Development Environment Specs
- **CPU**: M-series Apple Silicon (ARM64)
- **Memory**: 16+ GB recommended for multi-worker runs
- **Storage**: SSD for results/ directory  
- **Display**: Headless mode using SDL_VIDEODRIVER=dummy

### Performance Scaling
- **Single-core**: Environment creation and reset dominated by setup costs
- **Multi-core**: Near-linear scaling up to 4-8 workers depending on scenario complexity
- **Memory**: ~500MB per worker process typical

## Validation Scripts Status

### Available Validation Scripts
- ✅ `test_basic_environment.sh`: Environment creation smoke test
- ✅ `test_model_prediction.sh`: Model loading and inference test  
- ✅ `performance_smoke_test.py`: Performance baseline measurement
- ⏳ `test_complete_simulation.sh`: Full episode simulation (may timeout on complex scenarios)

### Usage in CI/CD
```bash
# Run all validation scripts
./scripts/validation/test_basic_environment.sh
./scripts/validation/test_model_prediction.sh
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/performance_smoke_test.py
```

## Performance History

### 2025-01-19 - Initial Baseline
- Environment creation: 1.16s (target: < 2.0s)
- Environment reset: 1,745 resets/sec (target: > 1/sec)  
- Platform: macOS ARM64, Python 3.13, headless mode
- Status: All performance targets met

---
**Monitoring Schedule**: Monthly performance validation recommended  
**Next Review**: 2025-02-19  
**Contact**: Reference dev team for performance regression issues